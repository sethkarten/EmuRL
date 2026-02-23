#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Detection for Pokemon Red RL.

Detects when the agent visits states that are outside the speedrun training
distribution. Uses VQ-VAE reconstruction error as the primary OOD signal.

Usage:
    from ood_detector import OODDetector

    detector = OODDetector.from_tokenizer("pretrained_tokenizer.pkl")
    ood_scores = detector.compute_ood_scores(frames)

    # Penalize rewards for OOD states
    adjusted_rewards = rewards * detector.get_reward_multiplier(ood_scores)
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class OODConfig:
    """Configuration for OOD detection."""
    # Reconstruction thresholds (calibrated on speedrun data)
    reconstruction_threshold: float = 0.05   # MSE above this is suspicious
    reconstruction_clip: float = 0.2         # Cap the OOD score at this MSE

    # Dynamics (temporal) thresholds
    dynamics_threshold: float = 0.3          # Token accuracy below this is suspicious
    dynamics_weight: float = 0.7             # Weight for dynamics vs reconstruction OOD

    # Reward adjustment
    min_reward_multiplier: float = 0.1       # Minimum reward for high-OOD states
    ood_penalty_scale: float = 5.0           # How quickly rewards drop with OOD score


class OODDetector:
    """
    Detects out-of-distribution states using VQ-VAE reconstruction error.

    The key insight is that the VQ-VAE was trained on speedrun videos, so:
    - Low reconstruction error = frame looks like speedrun gameplay
    - High reconstruction error = frame looks different (menus, glitches, etc.)

    We use this to penalize rewards for states that speedrunners never visit,
    encouraging the agent to stay on the "optimal path" through the game.
    """

    def __init__(
        self,
        tokenizer_model,
        tokenizer_params,
        config: Optional[OODConfig] = None,
        dynamics_model=None,
        dynamics_params=None,
    ):
        self.tokenizer = tokenizer_model
        self.params = tokenizer_params
        self.config = config or OODConfig()
        self.dynamics_model = dynamics_model
        self.dynamics_params = dynamics_params

        # JIT-compile the forward pass
        @jax.jit
        def _compute_recon_error(params, frames):
            """Compute reconstruction MSE for a batch of frames."""
            # Forward pass through tokenizer
            recon, tokens, z, codebook_loss, commitment_loss = tokenizer_model.apply(
                params, frames, train=False
            )

            # Normalize frames to [0, 1] for comparison
            frames_norm = frames.astype(jnp.float32) / 255.0

            # MSE per frame (mean over H, W, C)
            mse = jnp.mean((recon - frames_norm) ** 2, axis=(1, 2, 3))

            # Also return quantization error (distance from encoder output to codebook)
            quant_error = jnp.mean((z - jax.lax.stop_gradient(z)) ** 2, axis=(1, 2, 3))

            return mse, quant_error, recon, tokens

        self._compute_recon_error = _compute_recon_error

        # JIT-compile dynamics prediction if available
        if dynamics_model is not None:
            @jax.jit
            def _compute_dynamics_error(tok_params, dyn_params, frames_t, frames_t1):
                """Compute dynamics prediction error for frame pairs."""
                # Encode frames
                _, tokens_t, z_t, _, _ = tokenizer_model.apply(tok_params, frames_t, train=False)
                _, tokens_t1, _, _, _ = tokenizer_model.apply(tok_params, frames_t1, train=False)

                # Get token embeddings for frame t
                codebook = tok_params['params']['quantizer']['codebook']
                B, H, W = tokens_t.shape
                token_embeds = codebook[tokens_t.reshape(-1)].reshape(B, H * W, -1)

                # Predict next tokens
                logits = dynamics_model.apply(dyn_params, token_embeds, train=False)

                # Compare predicted tokens to actual next tokens
                predicted_tokens = jnp.argmax(logits, axis=-1)  # (B, H*W)
                actual_tokens = tokens_t1.reshape(B, -1)  # (B, H*W)

                # Token accuracy per frame
                accuracy = jnp.mean(predicted_tokens == actual_tokens, axis=1)

                return accuracy

            self._compute_dynamics_error = _compute_dynamics_error

        # Statistics for adaptive thresholding
        self.running_mean = 0.0
        self.running_var = 0.01
        self.count = 0

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer_path: str,
        dynamics_path: Optional[str] = None,
        config: Optional[OODConfig] = None
    ):
        """Load OOD detector from pretrained tokenizer and optionally dynamics model."""
        from pretrain_visual import Tokenizer, DynamicsModel, PretrainConfig

        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)

        # Get config
        if 'config' in data:
            tok_config = PretrainConfig(**data['config'])
        else:
            tok_config = PretrainConfig()

        # Create tokenizer model
        tokenizer = Tokenizer(config=tok_config)
        tok_params = data['params']

        # Load dynamics model if provided
        dynamics_model = None
        dynamics_params = None
        if dynamics_path is not None:
            with open(dynamics_path, 'rb') as f:
                dyn_data = pickle.load(f)

            dynamics_model = DynamicsModel(
                vocab_size=tok_config.vocab_size,
                embed_dim=tok_config.embed_dim,
                num_layers=tok_config.num_layers,
                num_heads=tok_config.num_heads,
                mlp_ratio=tok_config.mlp_ratio,
            )
            dynamics_params = dyn_data['params']

        return cls(tokenizer, tok_params, config, dynamics_model, dynamics_params)

    def compute_ood_scores(self, frames: jnp.ndarray) -> jnp.ndarray:
        """
        Compute OOD scores for a batch of frames using reconstruction error.

        Args:
            frames: (B, H, W, C) uint8 frames

        Returns:
            ood_scores: (B,) float scores in [0, 1], higher = more OOD
        """
        mse, quant_error, recon, tokens = self._compute_recon_error(self.params, frames)

        # Convert MSE to OOD score in [0, 1]
        # Score of 0 = perfectly in-distribution
        # Score of 1 = very out-of-distribution
        threshold = self.config.reconstruction_threshold
        clip = self.config.reconstruction_clip

        # Linear ramp from threshold to clip
        ood_scores = jnp.clip((mse - threshold) / (clip - threshold), 0.0, 1.0)

        return ood_scores

    def compute_temporal_ood_scores(
        self,
        frames_t: jnp.ndarray,
        frames_t1: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute OOD scores using dynamics model (temporal anomaly detection).

        Low prediction accuracy = the transition is unlike speedrun transitions.

        Args:
            frames_t: (B, H, W, C) current frames
            frames_t1: (B, H, W, C) next frames

        Returns:
            ood_scores: (B,) float scores in [0, 1], higher = more OOD
        """
        if self.dynamics_model is None:
            # Fall back to reconstruction-only OOD
            return self.compute_ood_scores(frames_t1)

        # Get dynamics prediction accuracy
        accuracy = self._compute_dynamics_error(
            self.params, self.dynamics_params, frames_t, frames_t1
        )

        # Convert accuracy to OOD score
        # High accuracy = low OOD (normal transition)
        # Low accuracy = high OOD (unexpected transition)
        threshold = self.config.dynamics_threshold

        # Score of 0 if accuracy >= threshold
        # Score of 1 if accuracy = 0
        ood_scores = jnp.clip((threshold - accuracy) / threshold, 0.0, 1.0)

        return ood_scores

    def compute_combined_ood_scores(
        self,
        frames_t: jnp.ndarray,
        frames_t1: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute combined OOD scores using both reconstruction and dynamics.

        Args:
            frames_t: (B, H, W, C) current frames
            frames_t1: (B, H, W, C) next frames

        Returns:
            ood_scores: (B,) float scores in [0, 1], higher = more OOD
        """
        # Reconstruction OOD on next frame
        recon_ood = self.compute_ood_scores(frames_t1)

        if self.dynamics_model is None:
            return recon_ood

        # Dynamics OOD on transition
        dyn_ood = self.compute_temporal_ood_scores(frames_t, frames_t1)

        # Weighted combination
        w = self.config.dynamics_weight
        combined = w * dyn_ood + (1 - w) * recon_ood

        return combined

    def compute_detailed_ood(self, frames: jnp.ndarray):
        """
        Compute detailed OOD metrics.

        Returns:
            dict with:
                - ood_scores: (B,) normalized OOD scores
                - reconstruction_mse: (B,) raw MSE values
                - reconstructed_frames: (B, H, W, C) reconstructed frames
                - tokens: (B, H, W) token indices
        """
        mse, quant_error, recon, tokens = self._compute_recon_error(self.params, frames)

        threshold = self.config.reconstruction_threshold
        clip = self.config.reconstruction_clip
        ood_scores = jnp.clip((mse - threshold) / (clip - threshold), 0.0, 1.0)

        return {
            'ood_scores': ood_scores,
            'reconstruction_mse': mse,
            'reconstructed_frames': recon,
            'tokens': tokens,
        }

    def get_reward_multiplier(self, ood_scores: jnp.ndarray) -> jnp.ndarray:
        """
        Convert OOD scores to reward multipliers.

        High OOD = low multiplier (penalize exploring unseen states)
        Low OOD = high multiplier (encourage staying on speedrun path)

        Args:
            ood_scores: (B,) scores in [0, 1]

        Returns:
            multipliers: (B,) values in [min_multiplier, 1.0]
        """
        scale = self.config.ood_penalty_scale
        min_mult = self.config.min_reward_multiplier

        # Exponential decay: multiplier = 1.0 at ood=0, min_mult at ood=1
        # multiplier = (1 - min_mult) * exp(-scale * ood) + min_mult
        multipliers = (1.0 - min_mult) * jnp.exp(-scale * ood_scores) + min_mult

        return multipliers

    def update_statistics(self, mse_values: np.ndarray):
        """Update running statistics for adaptive thresholding."""
        batch_mean = np.mean(mse_values)
        batch_var = np.var(mse_values)
        batch_count = len(mse_values)

        # Welford's online algorithm
        new_count = self.count + batch_count
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_count / new_count
        self.running_var = (
            (self.count * self.running_var + batch_count * batch_var) / new_count +
            (delta ** 2) * self.count * batch_count / (new_count ** 2)
        )
        self.count = new_count

    def calibrate_on_speedruns(self, frames_path: str, num_samples: int = 5000):
        """
        Calibrate OOD thresholds using speedrun training data.

        Sets thresholds such that:
        - 95% of speedrun frames have OOD score < 0.5
        - 99% of speedrun frames have OOD score < 1.0
        """
        from pathlib import Path

        frames_dir = Path(frames_path)
        all_frames = []

        # Load sample frames
        video_dirs = sorted(frames_dir.iterdir())
        for video_dir in video_dirs[:10]:  # Sample from first 10 videos
            if not video_dir.is_dir():
                continue
            npy_files = sorted(video_dir.glob("*.npy"))[:num_samples // 10]
            for npy_file in npy_files:
                frame = np.load(npy_file)
                all_frames.append(frame)

        if not all_frames:
            print("No frames found for calibration")
            return

        # Compute MSE on all frames
        frames = jnp.array(all_frames[:num_samples])
        mse_values = []

        batch_size = 64
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            mse, _, _ = self._compute_recon_error(self.params, batch)
            mse_values.extend(np.array(mse).tolist())

        mse_values = np.array(mse_values)

        # Set thresholds based on percentiles
        p95 = np.percentile(mse_values, 95)
        p99 = np.percentile(mse_values, 99)

        print(f"Calibration results on {len(mse_values)} frames:")
        print(f"  Mean MSE: {np.mean(mse_values):.4f}")
        print(f"  Std MSE:  {np.std(mse_values):.4f}")
        print(f"  P50 MSE:  {np.percentile(mse_values, 50):.4f}")
        print(f"  P95 MSE:  {np.percentile(mse_values, 95):.4f}")
        print(f"  P99 MSE:  {np.percentile(mse_values, 99):.4f}")

        # Update config
        self.config.reconstruction_threshold = float(p95)
        self.config.reconstruction_clip = float(p99 * 2)

        print(f"\nUpdated thresholds:")
        print(f"  threshold: {self.config.reconstruction_threshold:.4f}")
        print(f"  clip:      {self.config.reconstruction_clip:.4f}")

        return mse_values


def visualize_ood_examples(
    detector: OODDetector,
    speedrun_dir: str,
    agent_frames: Optional[np.ndarray] = None,
    output_path: str = "ood_examples.png"
):
    """
    Visualize OOD detection on example frames.

    Shows:
    - Top row: In-distribution frames (speedrun)
    - Bottom row: Out-of-distribution frames (agent exploration)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Load some speedrun frames
    speedrun_path = Path(speedrun_dir)
    speedrun_frames = []

    for video_dir in sorted(speedrun_path.iterdir())[:3]:
        if video_dir.is_dir():
            npy_files = sorted(video_dir.glob("*.npy"))
            for npy_file in npy_files[100:104]:  # Skip early frames
                frame = np.load(npy_file)
                speedrun_frames.append(frame)

    speedrun_frames = np.array(speedrun_frames[:8])

    # Compute OOD for speedrun frames
    speedrun_details = detector.compute_detailed_ood(speedrun_frames)

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))

    for i in range(min(8, len(speedrun_frames))):
        # Original frame
        axes[0, i].imshow(speedrun_frames[i])
        axes[0, i].set_title(f"OOD: {speedrun_details['ood_scores'][i]:.3f}\nMSE: {speedrun_details['reconstruction_mse'][i]:.4f}")
        axes[0, i].axis('off')

        # Reconstruction
        recon = np.array(speedrun_details['reconstructed_frames'][i] * 255).astype(np.uint8)
        axes[1, i].imshow(recon)
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')

    plt.suptitle("OOD Detection: Speedrun Frames", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOD Detection")
    parser.add_argument("--tokenizer", type=str, default="pretrained_tokenizer.pkl")
    parser.add_argument("--dynamics", type=str, default=None, help="Dynamics model for temporal OOD")
    parser.add_argument("--speedrun-frames", type=str, default="data/speedruns/frames")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate thresholds")
    parser.add_argument("--visualize", action="store_true", help="Visualize examples")
    parser.add_argument("--test-temporal", action="store_true", help="Test temporal OOD detection")
    args = parser.parse_args()

    print("Loading OOD detector...")
    detector = OODDetector.from_tokenizer(args.tokenizer, args.dynamics)

    if args.calibrate:
        print("\nCalibrating on speedrun data...")
        detector.calibrate_on_speedruns(args.speedrun_frames)

    if args.visualize:
        print("\nVisualizing examples...")
        visualize_ood_examples(detector, args.speedrun_frames)

    if args.test_temporal and args.dynamics:
        print("\nTesting temporal OOD detection...")
        from pathlib import Path

        speedrun_dir = Path(args.speedrun_frames)

        # Test on speedrun frame pairs (should have low OOD)
        print("\n1. Speedrun consecutive frames (expected: low OOD)")
        for video_dir in sorted(speedrun_dir.iterdir())[:2]:
            if not video_dir.is_dir():
                continue
            npy_files = sorted(video_dir.glob("*.npy"))
            # Sample consecutive pairs
            frames_t = []
            frames_t1 = []
            for i in range(100, min(120, len(npy_files) - 1)):
                frames_t.append(np.load(npy_files[i]))
                frames_t1.append(np.load(npy_files[i + 1]))

            frames_t = np.array(frames_t)
            frames_t1 = np.array(frames_t1)

            temporal_ood = detector.compute_temporal_ood_scores(frames_t, frames_t1)
            print(f"  {video_dir.name}: temporal OOD mean={np.mean(temporal_ood):.4f}, max={np.max(temporal_ood):.4f}")

        # Test on random frame pairs (should have high OOD)
        print("\n2. Random frame pairs (expected: high OOD)")
        all_frames = []
        for video_dir in sorted(speedrun_dir.iterdir())[:3]:
            if video_dir.is_dir():
                for npy in sorted(video_dir.glob("*.npy"))[:50]:
                    all_frames.append(np.load(npy))

        # Random pairs (non-consecutive)
        np.random.seed(42)
        indices = np.random.permutation(len(all_frames))
        frames_t = np.array([all_frames[i] for i in indices[:20]])
        frames_t1 = np.array([all_frames[i] for i in indices[50:70]])

        temporal_ood = detector.compute_temporal_ood_scores(frames_t, frames_t1)
        print(f"  Random pairs: temporal OOD mean={np.mean(temporal_ood):.4f}, max={np.max(temporal_ood):.4f}")
