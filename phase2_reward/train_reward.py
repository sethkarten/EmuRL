#!/usr/bin/env python3
"""
Train reward model using VQ-VAE backbone from pretraining.

Uses Bradley-Terry pairwise ranking: learns to predict which of two frames
represents more game progress, using temporal ordering in speedrun videos
as the training signal.

Architecture:
  Frame -> VQ Encoder (frozen) -> Token Embeddings -> Reward Head -> Score

Usage:
    uv run python train_reward.py \
        --data data/speedruns/frames \
        --tokenizer pretrained_tokenizer.pkl \
        --epochs 50
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from pathlib import Path
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, List
import random
import time
import pickle

# Enable bf16 training
jax.config.update("jax_default_matmul_precision", "bfloat16")


@dataclass
class RewardConfig:
    # Architecture (matching pretrain_visual.py)
    vocab_size: int = 512
    embed_dim: int = 192
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 192)
    token_grid: Tuple[int, int] = (9, 10)

    # Reward head
    hidden_dim: int = 256
    num_hidden_layers: int = 2

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 50
    samples_per_epoch: int = 10000
    margin: float = 0.0  # Margin for pairwise ranking

    # Data
    frame_shape: Tuple[int, int, int] = (144, 160, 3)
    min_time_gap: float = 5.0   # Minimum seconds between compared frames
    max_time_gap: float = 300.0  # Maximum seconds


# ============================================================================
# VQ-VAE ENCODER (matching pretrain_visual.py)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""
    channels: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        h = nn.GroupNorm(num_groups=min(8, self.channels))(x)
        h = nn.silu(h)
        h = nn.Conv(self.channels, (3, 3), padding='SAME')(h)
        h = nn.GroupNorm(num_groups=min(8, self.channels))(h)
        h = nn.silu(h)
        h = nn.Conv(self.channels, (3, 3), padding='SAME')(h)

        if x.shape[-1] != self.channels:
            x = nn.Conv(self.channels, (1, 1))(x)

        return x + h


class Encoder(nn.Module):
    """Encode frame to spatial feature grid."""
    channels: Tuple[int, ...] = (32, 64, 128, 192)
    embed_dim: int = 192

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0

        for ch in self.channels:
            x = nn.Conv(ch, (4, 4), strides=(2, 2), padding='SAME')(x)
            x = ResBlock(ch)(x, train)

        x = nn.Conv(self.embed_dim, (1, 1))(x)
        return x  # (B, 9, 10, embed_dim)


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through gradient."""
    vocab_size: int = 512
    embed_dim: int = 192

    @nn.compact
    def __call__(self, z, train: bool = True):
        B, H, W, D = z.shape

        codebook = self.param(
            'codebook',
            nn.initializers.variance_scaling(1.0, 'fan_in', 'uniform'),
            (self.vocab_size, self.embed_dim)
        )

        z_flat = z.reshape(-1, D)
        z_sq = jnp.sum(z_flat ** 2, axis=1, keepdims=True)
        e_sq = jnp.sum(codebook ** 2, axis=1, keepdims=True).T
        distances = z_sq + e_sq - 2 * jnp.dot(z_flat, codebook.T)

        indices = jnp.argmin(distances, axis=1).reshape(B, H, W)
        z_q = codebook[indices.reshape(-1)].reshape(B, H, W, D)

        # Straight-through
        z_q = z + jax.lax.stop_gradient(z_q - z)

        return z_q, indices


# ============================================================================
# REWARD MODEL
# ============================================================================

class RewardHead(nn.Module):
    """MLP head that maps token embeddings to scalar reward."""
    hidden_dim: int = 256
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, token_embeds, train: bool = True):
        # token_embeds: (B, H*W, D)
        # Mean pool over tokens
        x = jnp.mean(token_embeds, axis=1)  # (B, D)

        # MLP
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)

        # Output scalar score
        score = nn.Dense(1)(x).squeeze(-1)  # (B,)

        return score


class RewardModel(nn.Module):
    """Full reward model: Frame -> VQ Encoder -> Token Embeddings -> Score."""
    config: RewardConfig
    freeze_encoder: bool = True

    def setup(self):
        self.encoder = Encoder(
            channels=self.config.encoder_channels,
            embed_dim=self.config.embed_dim,
        )
        self.quantizer = VectorQuantizer(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
        )
        self.reward_head = RewardHead(
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
        )

    def __call__(self, frames, train: bool = True):
        # Encode frames
        z = self.encoder(frames, train=False if self.freeze_encoder else train)

        # Quantize
        z_q, tokens = self.quantizer(z, train=False if self.freeze_encoder else train)

        # Flatten to sequence
        B, H, W, D = z_q.shape
        token_embeds = z_q.reshape(B, H * W, D)

        # Get reward score
        if self.freeze_encoder:
            # Stop gradients to encoder
            token_embeds = jax.lax.stop_gradient(token_embeds)

        score = self.reward_head(token_embeds, train)

        return score


# ============================================================================
# DATASET
# ============================================================================

class PairwiseFrameDataset:
    """Dataset for pairwise reward model training."""

    def __init__(self, data_dir: Path, config: RewardConfig):
        self.config = config
        self.data_dir = Path(data_dir)

        # Load all videos with timestamps
        self.videos = []
        for index_path in sorted(self.data_dir.glob("*_index.json")):
            with open(index_path) as f:
                index = json.load(f)
                frames = []
                for entry in index['frames']:
                    frame_path = self.data_dir / entry['path']
                    if frame_path.exists():
                        frames.append({
                            'path': frame_path,
                            'timestamp': entry.get('timestamp', entry['frame_idx']),
                            'progress': entry.get('progress', entry['frame_idx']),
                        })
                if len(frames) > 20:  # Need enough frames for pairs
                    self.videos.append(frames)

        total_frames = sum(len(v) for v in self.videos)
        print(f"Loaded {len(self.videos)} videos with {total_frames:,} frames")

    def sample_pairs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample pairs of frames where later frame should have higher score.

        Returns:
            frames_early: (B, H, W, C) earlier frames
            frames_later: (B, H, W, C) later frames
            margins: (B,) time gaps (for weighted ranking loss)
        """
        frames_early = []
        frames_later = []
        margins = []

        for _ in range(batch_size):
            # Pick random video
            video = random.choice(self.videos)

            # Pick two frames with sufficient time gap
            max_attempts = 10
            for _ in range(max_attempts):
                idx1 = random.randint(0, len(video) - 2)
                idx2 = random.randint(idx1 + 1, len(video) - 1)

                t1 = video[idx1]['timestamp']
                t2 = video[idx2]['timestamp']
                gap = t2 - t1

                if self.config.min_time_gap <= gap <= self.config.max_time_gap:
                    break
            else:
                # Use what we have
                idx1 = random.randint(0, len(video) - 2)
                idx2 = random.randint(idx1 + 1, len(video) - 1)
                gap = video[idx2]['timestamp'] - video[idx1]['timestamp']

            frames_early.append(np.load(video[idx1]['path']))
            frames_later.append(np.load(video[idx2]['path']))
            margins.append(gap)

        return np.stack(frames_early), np.stack(frames_later), np.array(margins)


# ============================================================================
# TRAINING
# ============================================================================

def create_reward_state(config: RewardConfig, key, tokenizer_params=None, freeze_encoder=False):
    """Create reward model training state."""
    model = RewardModel(config, freeze_encoder=freeze_encoder)

    dummy = jnp.zeros((1, *config.frame_shape), dtype=jnp.uint8)
    params = model.init(key, dummy)

    # Load pretrained encoder weights if provided
    if tokenizer_params is not None:
        import flax
        params = flax.core.unfreeze(params)
        params['params']['encoder'] = tokenizer_params['params']['encoder']
        params['params']['quantizer'] = tokenizer_params['params']['quantizer']
        params = flax.core.freeze(params)
        print("  Loaded pretrained encoder from tokenizer")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=0.01),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), model


def train_reward(config: RewardConfig, dataset: PairwiseFrameDataset,
                 tokenizer_params: dict = None,
                 output_path: str = "reward_model.pkl",
                 freeze_encoder: bool = False):
    """Train reward model with Bradley-Terry pairwise ranking."""
    print("\n" + "=" * 70)
    print("STAGE 2: Training Reward Model (Pairwise Ranking)")
    print("=" * 70)
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Hidden layers: {config.num_hidden_layers}")
    encoder_status = "frozen" if freeze_encoder else "trainable (finetuning)"
    if tokenizer_params:
        encoder_status += " (pretrained init)"
    else:
        encoder_status += " (random init)"
    print(f"  Encoder: {encoder_status}")

    key = jax.random.PRNGKey(44)
    key, init_key = jax.random.split(key)

    state, model = create_reward_state(config, init_key, tokenizer_params, freeze_encoder)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Total parameters: {num_params:,}")

    # JIT compile training step
    @jax.jit
    def train_step(state, frames_early, frames_later, key):
        """Bradley-Terry ranking loss: later frames should score higher."""
        def loss_fn(params):
            # Get scores for both frames
            scores_early = model.apply(params, frames_early, train=True, rngs={'dropout': key})
            key2 = jax.random.fold_in(key, 1)
            scores_later = model.apply(params, frames_later, train=True, rngs={'dropout': key2})

            # Bradley-Terry loss: log sigmoid(score_later - score_early)
            # Higher score_later should give lower loss
            diff = scores_later - scores_early - config.margin
            loss = -jnp.mean(jax.nn.log_sigmoid(diff))

            # Accuracy: how often do we correctly rank the pair?
            accuracy = jnp.mean((scores_later > scores_early).astype(jnp.float32))

            return loss, accuracy

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss, acc

    # Inference function for evaluation
    @jax.jit
    def get_score(params, frames):
        return model.apply(params, frames, train=False)

    steps_per_epoch = config.samples_per_epoch // config.batch_size

    print(f"\nTraining for {config.epochs} epochs ({steps_per_epoch} steps/epoch)")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Loss':>10} | {'Accuracy':>10} | {'Time':>5}")
    print("-" * 70)

    for epoch in range(config.epochs):
        t0 = time.time()
        epoch_loss = 0
        epoch_acc = 0

        for step in range(steps_per_epoch):
            frames_early, frames_later, _ = dataset.sample_pairs(config.batch_size)

            key, step_key = jax.random.split(key)
            state, loss, acc = train_step(
                state,
                jnp.array(frames_early),
                jnp.array(frames_later),
                step_key
            )

            epoch_loss += float(loss)
            epoch_acc += float(acc)

        epoch_loss /= steps_per_epoch
        epoch_acc /= steps_per_epoch
        epoch_time = time.time() - t0

        print(f"{epoch+1:5d} | {epoch_loss:10.4f} | {epoch_acc:10.2%} | {epoch_time:4.1f}s")

        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            save_reward_model(state.params, config, output_path)
            print(f"  -> Saved checkpoint to {output_path}")

    print("=" * 70)
    print("Reward model training complete!")

    # Final evaluation
    print("\nEvaluating on 1000 random pairs...")
    correct = 0
    for _ in range(1000):
        frames_early, frames_later, _ = dataset.sample_pairs(1)
        score_early = float(get_score(state.params, jnp.array(frames_early))[0])
        score_later = float(get_score(state.params, jnp.array(frames_later))[0])
        if score_later > score_early:
            correct += 1
    print(f"  Pairwise accuracy: {correct / 10:.1f}%")

    return state.params


def save_reward_model(params, config: RewardConfig, path: str):
    """Save reward model."""
    with open(path, 'wb') as f:
        pickle.dump({
            'params': params,
            'config': asdict(config),
            'model_type': 'vq_reward',  # Mark as new VQ-based model
        }, f)


def load_tokenizer(path: str):
    """Load tokenizer params from pretrain_visual.py output."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data.get('config', {})


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train reward model with VQ backbone")
    parser.add_argument("--data", type=str, default="data/speedruns/frames")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to pretrained tokenizer (for encoder initialization)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights (only train MLP head)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output", type=str, default="reward_vq.pkl")
    args = parser.parse_args()

    config = RewardConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
    )

    print("=" * 70)
    print("REWARD MODEL TRAINING (VQ BACKBONE)")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Tokenizer: {args.tokenizer or '(random init)'}")

    # Load tokenizer if provided
    tokenizer_params = None
    if args.tokenizer:
        print(f"\nLoading tokenizer from {args.tokenizer}")
        tokenizer_params, tok_config = load_tokenizer(args.tokenizer)
        # Override config with tokenizer settings
        if 'vocab_size' in tok_config:
            config.vocab_size = tok_config['vocab_size']
        if 'embed_dim' in tok_config:
            config.embed_dim = tok_config['embed_dim']
        if 'encoder_channels' in tok_config:
            config.encoder_channels = tuple(tok_config['encoder_channels'])

    print(f"\nConfig:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")

    print("\nLoading dataset...")
    dataset = PairwiseFrameDataset(args.data, config)

    if len(dataset.videos) == 0:
        print("ERROR: No training data found!")
        return

    train_reward(config, dataset, tokenizer_params, args.output, freeze_encoder=args.freeze_encoder)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {args.output}")
    print(f"\nNext: Run RL with this reward model:")
    print(f"  uv run python train_with_cnn_reward.py \\")
    print(f"      --reward-model {args.output} \\")
    print(f"      --pretrained-encoder pretrained_encoder.pkl")


if __name__ == "__main__":
    main()
