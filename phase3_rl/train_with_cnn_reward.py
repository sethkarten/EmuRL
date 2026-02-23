#!/usr/bin/env python3
"""
Fast RL training using CNN reward model instead of VLM.

This is ~100x faster than VLM-based training since CNN inference
takes <1ms vs ~10s for VLM calls.

Usage:
    python train_with_cnn_reward.py --reward-model reward_cnn.pkl --iterations 50000
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
import optax
import time
import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple

# Enable bf16 training for speed
jax.config.update("jax_default_matmul_precision", "bfloat16")

# Multi-GPU setup
NUM_DEVICES = jax.local_device_count()
print(f"JAX devices: {NUM_DEVICES} GPUs available")


# ============================================================================
# BLANK FRAME DETECTION
# ============================================================================

def is_blank_frame(frame: np.ndarray, threshold: float = 20.0) -> bool:
    """Check if a single frame is blank (all black, all white, or very low variance)."""
    if frame.max() - frame.min() < threshold:
        return True
    if frame.var() < 100:
        return True
    return False


def get_blank_mask(frames: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """Get boolean mask of blank frames. Shape: (N,)"""
    # frames: (N, H, W, C)
    # Check variance per frame
    var_per_frame = frames.var(axis=(1, 2, 3))
    range_per_frame = frames.max(axis=(1, 2, 3)) - frames.min(axis=(1, 2, 3))
    blank_mask = (var_per_frame < 100) | (range_per_frame < threshold)
    return blank_mask


BLANK_FRAME_SCORE = -10.0  # Score to assign to blank frames


# ============================================================================
# GAMEBOY BUTTON MAPPING
# ============================================================================
# The emulator uses a bitmask for buttons, where:
# Bit 0 = Right, Bit 1 = Left, Bit 2 = Up, Bit 3 = Down
# Bit 4 = A, Bit 5 = B, Bit 6 = Select, Bit 7 = Start

# Map action indices 0-7 to proper button bitmasks
ACTION_TO_BUTTON = np.array([
    1,    # 0: Right (bit 0)
    2,    # 1: Left (bit 1)
    4,    # 2: Up (bit 2)
    8,    # 3: Down (bit 3)
    16,   # 4: A (bit 4)
    32,   # 5: B (bit 5)
    64,   # 6: Select (bit 6)
    128,  # 7: Start (bit 7)
], dtype=np.uint8)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Environment
    num_envs: int = 32
    frame_skip: int = 30

    # Trajectory (can be much longer without VLM bottleneck)
    trajectory_length: int = 64  # Reduced for GPU memory
    reward_interval: int = 64  # Compute reward every 64 steps (~32s game time)

    # Policy network (~1M params)
    embed_dim: int = 128
    vision_layers: int = 3
    decoder_layers: int = 1
    num_heads: int = 4
    patch_size: int = 16  # Larger patches = fewer tokens
    num_actions: int = 8

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    minibatch_size: int = 64  # Reduced for GPU memory
    time_penalty: float = 0.001  # Cost per step to encourage faster progress

    # Exploration bonus
    novelty_coef: float = 0.5  # Bonus for reaching new score levels
    novelty_bucket_size: float = 0.5  # Score bucket size for novelty tracking

    # Progress checkpointing (for curriculum learning)
    checkpoint_progress_threshold: float = 0.5  # Save state when progress exceeds this delta
    max_checkpoints: int = 100  # Maximum number of checkpoints to keep
    reset_to_checkpoint_prob: float = 0.1  # Probability of resetting env to a checkpoint
    episode_length: int = 2048  # Max steps before soft reset (0 = no limit)

    # Logging
    wandb_project: str = "pokemon-red-vlm"
    log_interval: int = 10
    checkpoint_interval: int = 10_000_000  # Only save final checkpoint


# ============================================================================
# PROGRESS CHECKPOINTER (for curriculum learning)
# ============================================================================

class ProgressCheckpointer:
    """
    Saves game states at progress milestones for curriculum learning.

    Maintains a globally-ordered set of checkpoints, evenly distributed across
    the progress range. Only saves new checkpoints when they represent genuine
    new progress milestones. Also stores frames for visual review.
    """

    def __init__(self, config: Config):
        self.config = config
        # Ordered list of (progress_score, state_id, frame) - sorted by progress
        # frame is a numpy array (144, 160, 3) uint8
        self.checkpoints = []
        self.global_best_progress = -float('inf')
        self.steps_since_reset = {}  # env_idx -> steps

    def update(self, env, progress_scores: np.ndarray, frames: np.ndarray, env_indices: list = None):
        """
        Check if any environment has reached a new global progress milestone.

        Only saves checkpoints that:
        1. Exceed the current global best, OR
        2. Fill a gap in the progress distribution

        Args:
            env: VecGameBoy environment
            progress_scores: Array of progress scores, one per environment
            frames: Current frames from environments, shape (num_envs, 144, 160, 3)
            env_indices: Which env indices these scores correspond to (default: all)
        """
        if env_indices is None:
            env_indices = list(range(len(progress_scores)))

        new_checkpoints = 0
        threshold = self.config.checkpoint_progress_threshold

        for i, env_idx in enumerate(env_indices):
            score = float(progress_scores[i])
            frame = frames[env_idx]

            # Skip blank frames - don't save them as checkpoints
            if is_blank_frame(frame):
                continue

            # IMPORTANT: Only save states with positive progress scores
            # This prevents saving intro/title states as checkpoints
            # Late gameplay scores around +3 to +6, intro around -3 to +3
            # We use 2.0 as threshold to ensure we only save actual gameplay
            MIN_CHECKPOINT_SCORE = 2.0
            if score < MIN_CHECKPOINT_SCORE:
                continue

            # Check if this represents a new milestone
            should_save = False

            if score > self.global_best_progress + threshold:
                # New global best - always save
                should_save = True
                self.global_best_progress = score
            elif len(self.checkpoints) < self.config.max_checkpoints:
                # Room for more checkpoints - check if this fills a gap
                should_save = self._fills_gap(score, threshold)

            if should_save:
                state_id = env.save_state(env_idx)
                frame_copy = frame.copy()  # Store a copy of the frame
                self._insert_checkpoint(score, state_id, frame_copy)
                new_checkpoints += 1

        return new_checkpoints

    def _fills_gap(self, score: float, threshold: float) -> bool:
        """Check if this score fills a gap in the checkpoint distribution."""
        if not self.checkpoints:
            return True

        # Find where this score would be inserted
        for existing_score, _, _ in self.checkpoints:
            if abs(score - existing_score) < threshold:
                # Too close to an existing checkpoint
                return False

        return True

    def _insert_checkpoint(self, score: float, state_id: int, frame: np.ndarray):
        """Insert checkpoint maintaining sorted order by progress score."""
        # Remove any checkpoint too close to this score (dedup)
        threshold = self.config.checkpoint_progress_threshold
        self.checkpoints = [
            (s, sid, f) for s, sid, f in self.checkpoints
            if abs(s - score) >= threshold * 0.5  # Keep if sufficiently different
        ]

        # Insert in sorted order
        insert_idx = 0
        for i, (existing_score, _, _) in enumerate(self.checkpoints):
            if score > existing_score:
                insert_idx = i + 1

        self.checkpoints.insert(insert_idx, (score, state_id, frame))

        # Prune if too many - keep evenly distributed across progress range
        if len(self.checkpoints) > self.config.max_checkpoints:
            self._prune_checkpoints()

    def _prune_checkpoints(self):
        """Prune checkpoints to keep them evenly distributed across progress range."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return

        # Always keep the best (last) and worst (first) checkpoints
        # Remove checkpoints that are closest to their neighbors
        while len(self.checkpoints) > self.config.max_checkpoints:
            min_gap = float('inf')
            remove_idx = 1  # Don't remove first or last

            for i in range(1, len(self.checkpoints) - 1):
                prev_score = self.checkpoints[i - 1][0]
                curr_score = self.checkpoints[i][0]
                next_score = self.checkpoints[i + 1][0]

                # Gap created if we remove this checkpoint
                gap_before = curr_score - prev_score
                gap_after = next_score - curr_score
                min_neighbor_gap = min(gap_before, gap_after)

                if min_neighbor_gap < min_gap:
                    min_gap = min_neighbor_gap
                    remove_idx = i

            self.checkpoints.pop(remove_idx)

    def save_checkpoint_frames(self, output_dir: str = "checkpoint_frames"):
        """
        Save all checkpoint frames as images for visual review.

        Creates a grid image and individual frames sorted by progress score.
        """
        if not self.checkpoints:
            print("No checkpoints to save")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Save individual frames
        for i, (score, state_id, frame) in enumerate(self.checkpoints):
            from PIL import Image
            img = Image.fromarray(frame)
            # Scale up 3x for better visibility
            img = img.resize((160 * 3, 144 * 3), Image.NEAREST)
            img.save(f"{output_dir}/ckpt_{i:02d}_prog_{score:.2f}.png")

        # Create a grid image showing all checkpoints
        n = len(self.checkpoints)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        from PIL import Image, ImageDraw, ImageFont
        cell_w, cell_h = 160 * 2, 144 * 2 + 20  # Space for label
        grid = Image.new('RGB', (cols * cell_w, rows * cell_h), (40, 40, 40))
        draw = ImageDraw.Draw(grid)

        for i, (score, state_id, frame) in enumerate(self.checkpoints):
            row, col = i // cols, i % cols
            x, y = col * cell_w, row * cell_h

            # Resize and paste frame
            img = Image.fromarray(frame)
            img = img.resize((160 * 2, 144 * 2), Image.NEAREST)
            grid.paste(img, (x, y))

            # Add progress label
            label = f"#{i} prog={score:.1f}"
            draw.text((x + 5, y + 144 * 2 + 2), label, fill=(255, 255, 255))

        grid.save(f"{output_dir}/checkpoint_grid.png")
        print(f"Saved {n} checkpoint frames to {output_dir}/")

    def maybe_reset_envs(self, env, obs: np.ndarray, rng: np.random.Generator, steps_this_iter: int = 1):
        """
        Probabilistically reset some environments to checkpoints.

        This implements curriculum learning by allowing agents to practice
        from various points in the game, not just the beginning.

        Args:
            env: VecGameBoy environment
            obs: Observation buffer to update
            rng: Random number generator
            steps_this_iter: Number of steps taken this iteration (for episode length tracking)

        Returns:
            Number of environments reset
        """
        if not self.checkpoints:
            return 0

        num_reset = 0
        # Get number of environments (handle both property and method)
        if hasattr(env, 'num_envs'):
            ne = env.num_envs
            num_envs = ne() if callable(ne) else ne
        else:
            num_envs = obs.shape[0]

        for env_idx in range(num_envs):
            # Update step counter
            self.steps_since_reset[env_idx] = self.steps_since_reset.get(env_idx, 0) + steps_this_iter

            should_reset = False

            # Check episode length limit
            if self.config.episode_length > 0:
                if self.steps_since_reset[env_idx] >= self.config.episode_length:
                    should_reset = True

            # Random checkpoint reset (low probability per iteration)
            if rng.random() < self.config.reset_to_checkpoint_prob:
                should_reset = True

            if should_reset:
                # Sample a checkpoint (biased toward higher progress)
                checkpoint = self._sample_checkpoint(rng)
                if checkpoint is not None:
                    score, state_id, _ = checkpoint  # Ignore frame when loading
                    env.load_state(env_idx, state_id)
                    self.steps_since_reset[env_idx] = 0
                    num_reset += 1

        # Update observations after any resets
        if num_reset > 0:
            env.render(obs)

        return num_reset

    def _sample_checkpoint(self, rng: np.random.Generator):
        """Sample a checkpoint, biased toward higher progress scores."""
        if not self.checkpoints:
            return None

        n = len(self.checkpoints)

        # Checkpoints are sorted by progress (ascending), so higher indices = more progress
        # Use weighted sampling biased toward higher progress (later in list)
        # Weight = index + 1 (so first checkpoint has weight 1, last has weight n)
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights = weights / weights.sum()

        idx = rng.choice(n, p=weights)
        return self.checkpoints[idx]

    def get_stats(self):
        """Get checkpoint statistics for logging."""
        if not self.checkpoints:
            return {
                'num_checkpoints': 0,
                'best_progress': self.global_best_progress,
                'progress_range': (0, 0),
            }
        scores = [c[0] for c in self.checkpoints]
        return {
            'num_checkpoints': len(self.checkpoints),
            'best_progress': self.global_best_progress,
            'progress_range': (min(scores), max(scores)),
        }


# ============================================================================
# NOVELTY TRACKER (for exploration bonus)
# ============================================================================

class NoveltyTracker:
    """
    Tracks visited score levels and provides novelty bonus for reaching new ones.

    This encourages exploration by rewarding the agent for reaching score levels
    it hasn't seen before. Uses score buckets to group similar scores together.
    """

    def __init__(self, bucket_size: float = 0.5):
        self.bucket_size = bucket_size
        self.visited_buckets = set()  # Global set of visited score buckets
        self.env_last_bucket = {}  # Per-env tracking of last bucket

    def _get_bucket(self, score: float) -> int:
        """Convert score to bucket index."""
        return int(score / self.bucket_size)

    def compute_novelty_bonus(self, scores: np.ndarray, env_indices: list = None) -> np.ndarray:
        """
        Compute novelty bonus for each environment based on score.

        Returns bonus of 1.0 for reaching a new global bucket, 0.0 otherwise.
        """
        if env_indices is None:
            env_indices = list(range(len(scores)))

        bonuses = np.zeros(len(scores), dtype=np.float32)

        for i, env_idx in enumerate(env_indices):
            score = float(scores[i])
            bucket = self._get_bucket(score)

            # Check if this bucket is new (globally)
            if bucket not in self.visited_buckets:
                bonuses[i] = 1.0
                self.visited_buckets.add(bucket)

            # Update per-env tracking
            self.env_last_bucket[env_idx] = bucket

        return bonuses

    def get_stats(self):
        """Get novelty tracking statistics."""
        if not self.visited_buckets:
            return {'visited_buckets': 0, 'bucket_range': (0, 0)}
        buckets = list(self.visited_buckets)
        return {
            'visited_buckets': len(buckets),
            'bucket_range': (min(buckets) * self.bucket_size, max(buckets) * self.bucket_size),
        }


# ============================================================================
# CNN REWARD MODEL (ResNet from train_reward_cnn.py)
# ============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""
    channels: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        y = nn.Conv(self.channels, (3, 3), strides=(self.strides, self.strides), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if self.strides != 1 or x.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, (1, 1), strides=(self.strides, self.strides), use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """Configurable ResNet architecture for reward prediction."""
    blocks_per_stage: tuple = (2, 2, 2, 2)
    channels: tuple = (64, 128, 256, 512)
    num_classes: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0

        x = nn.Conv(self.channels[0], (7, 7), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for stage_idx, (num_blocks, ch) in enumerate(zip(self.blocks_per_stage, self.channels)):
            for block_idx in range(num_blocks):
                strides = 2 if block_idx == 0 and stage_idx > 0 else 1
                x = ResidualBlock(ch, strides=strides)(x, train)

        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)

        return x.squeeze(-1)


# Model configurations
MODEL_CONFIGS = {
    'tiny': {'blocks_per_stage': (1,1,1,1), 'channels': (32,64,128,256)},
    'small': {'blocks_per_stage': (1,1,1,1), 'channels': (64,128,256,512)},
    'medium': {'blocks_per_stage': (2,2,2,2), 'channels': (32,64,128,256)},
    '18': {'blocks_per_stage': (2,2,2,2), 'channels': (64,128,256,512)},
    '34': {'blocks_per_stage': (3,4,6,3), 'channels': (64,128,256,512)},
}


def load_reward_model(path: str):
    """Load trained ResNet reward model."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Get model config
    model_name = data['config'].get('model_name', '18')
    if model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
        model = ResNet(
            blocks_per_stage=cfg['blocks_per_stage'],
            channels=cfg['channels'],
        )
    else:
        # Fallback for old models
        model = ResNet()

    return model, data['params'], data['batch_stats']


def load_sequence_reward_model(path: str):
    """Load trained sequence reward model (transformer-based)."""
    from train_reward_sequence import SequenceRewardModel, SequenceRewardConfig

    with open(path, 'rb') as f:
        data = pickle.load(f)

    cfg = data['config']
    config = SequenceRewardConfig(
        seq_len=cfg.get('seq_len', 8),
        vocab_size=cfg.get('vocab_size', 512),
        embed_dim=cfg.get('embed_dim', 192),
        encoder_channels=tuple(cfg.get('encoder_channels', (32, 64, 128, 192))),
        num_layers=cfg.get('num_layers', 4),
        num_heads=cfg.get('num_heads', 4),
    )

    model = SequenceRewardModel(config)

    # Handle nested params structure
    params = data['params']
    if 'params' in params and 'params' in params['params']:
        # Double nested - flatten one level
        params = params['params']

    return model, params, config


# ============================================================================
# POLICY MODEL
# ============================================================================

# ============================================================================
# VQ POLICY (matching pretrain_visual.py architecture exactly)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm (matches pretrain_visual.py)."""
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
    """VQ-VAE encoder (matches pretrain_visual.py exactly)."""
    channels: tuple = (32, 64, 128, 192)
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
    """Vector quantization (matches pretrain_visual.py exactly)."""
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


class TransformerBlock(nn.Module):
    """Transformer block (matches pretrain_visual.py exactly)."""
    embed_dim: int
    num_heads: int = 4
    mlp_ratio: int = 3

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
        )(y, y)
        y = nn.Dropout(0.1, deterministic=not train)(y)
        x = x + y

        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.embed_dim * self.mlp_ratio)(y)
        y = nn.gelu(y)
        y = nn.Dropout(0.1, deterministic=not train)(y)
        y = nn.Dense(self.embed_dim)(y)
        y = nn.Dropout(0.1, deterministic=not train)(y)
        x = x + y

        return x


class VQPolicy(nn.Module):
    """
    Policy using pretrained VQ-VAE encoder + dynamics transformer backbone.

    Architecture (matches pretrain_visual.py):
      Frame -> Encoder -> VectorQuantizer -> pos_embed -> TransformerBlocks -> LayerNorm
           -> Action/Value heads (replacing vocab prediction head)

    All components (encoder, quantizer, transformer) are finetuned with RL.
    """
    vocab_size: int = 512
    embed_dim: int = 192
    encoder_channels: tuple = (32, 64, 128, 192)
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 3
    num_actions: int = 8
    num_tokens: int = 90  # 9 * 10

    @nn.compact
    def __call__(self, frames, train: bool = True):
        B = frames.shape[0]

        # === ENCODER (from tokenizer) ===
        z = Encoder(
            channels=self.encoder_channels,
            embed_dim=self.embed_dim,
        )(frames, train)

        # === QUANTIZER (from tokenizer) ===
        z_q, tokens = VectorQuantizer(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )(z, train)

        # Flatten to sequence: (B, 9, 10, D) -> (B, 90, D)
        token_embeds = z_q.reshape(B, self.num_tokens, self.embed_dim)

        # === TRANSFORMER (from dynamics model) ===
        # Add positional embeddings
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, self.num_tokens, self.embed_dim)
        )
        x = token_embeds + pos_embed

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(x, train)

        # Final LayerNorm (same as dynamics model)
        x = nn.LayerNorm()(x)

        # === POLICY/VALUE HEADS (replacing Dense(vocab)) ===
        pooled = jnp.mean(x, axis=1)  # (B, embed_dim)
        policy_logits = nn.Dense(self.num_actions, name='policy_head')(pooled)
        value = nn.Dense(1, name='value_head')(pooled).squeeze(-1)

        return policy_logits, value


# Legacy ViT-based model (kept for backward compatibility)
class PatchEmbed(nn.Module):
    patch_size: int = 8
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = nn.Conv(
            self.embed_dim,
            (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID'
        )(x)
        return x.reshape(B, -1, self.embed_dim)


class LegacyTransformerBlock(nn.Module):
    """Legacy ViT transformer block (kept for backward compatibility)."""
    embed_dim: int = 256
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            deterministic=not train,
        )(y, y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.embed_dim * 4)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim)(y)
        return x + y


class VisionEncoder(nn.Module):
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    patch_size: int = 8

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0
        x = PatchEmbed(self.patch_size, self.embed_dim)(x)
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02),
                               (1, x.shape[1], self.embed_dim))
        x = x + pos_embed
        for _ in range(self.num_layers):
            x = LegacyTransformerBlock(self.embed_dim, self.num_heads)(x, train)
        return nn.LayerNorm()(x)


class LanguageDecoder(nn.Module):
    embed_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4

    @nn.compact
    def __call__(self, vision_features, train: bool = True):
        B = vision_features.shape[0]
        action_query = self.param('action_query', nn.initializers.normal(0.02),
                                  (1, 1, self.embed_dim))
        x = jnp.broadcast_to(action_query, (B, 1, self.embed_dim))

        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.embed_dim,
                deterministic=not train)(y, y)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.embed_dim,
                deterministic=not train)(y, vision_features)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.embed_dim * 4)(y)
            y = nn.gelu(y)
            y = nn.Dense(self.embed_dim)(y)
            x = x + y

        return nn.LayerNorm()(x)[:, 0, :]


class SmallVLMPolicy(nn.Module):
    """Legacy ViT-based policy (kept for backward compatibility)."""
    embed_dim: int = 256
    vision_layers: int = 6
    decoder_layers: int = 2
    num_heads: int = 4
    num_actions: int = 8
    patch_size: int = 8

    @nn.compact
    def __call__(self, frames, train: bool = True):
        vision_features = VisionEncoder(
            self.embed_dim, self.vision_layers, self.num_heads, self.patch_size
        )(frames, train)
        decoder_output = LanguageDecoder(
            self.embed_dim, self.decoder_layers, self.num_heads
        )(vision_features, train)
        policy_logits = nn.Dense(self.num_actions)(decoder_output)
        value = nn.Dense(1)(decoder_output).squeeze(-1)
        return policy_logits, value


# ============================================================================
# TRAINING
# ============================================================================

def create_train_state(config: Config, key, pretrained_tokenizer=None, pretrained_dynamics=None):
    """
    Create training state with policy model.

    If pretrained_tokenizer is provided, loads encoder + quantizer weights.
    If pretrained_dynamics is also provided, loads transformer weights.
    Otherwise, uses legacy SmallVLMPolicy (ViT-based).
    """
    dummy = jnp.zeros((1, 144, 160, 3), dtype=jnp.uint8)

    if pretrained_tokenizer is not None:
        # Get config from tokenizer
        tok_config = pretrained_tokenizer.get('config', {})
        vocab_size = tok_config.get('vocab_size', 512)
        embed_dim = tok_config.get('embed_dim', 192)
        encoder_channels = tuple(tok_config.get('encoder_channels', (32, 64, 128, 192)))
        num_tokens = tok_config.get('token_grid', (9, 10))
        if isinstance(num_tokens, (list, tuple)):
            num_tokens = num_tokens[0] * num_tokens[1]

        model = VQPolicy(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_channels=encoder_channels,
            num_layers=4,
            num_heads=4,
            mlp_ratio=3,
            num_actions=config.num_actions,
            num_tokens=num_tokens,
        )
        params = model.init(key, dummy)

        # Load pretrained weights
        params = flax.core.unfreeze(params)
        tok_params = pretrained_tokenizer['params']['params']

        # Load encoder and quantizer from tokenizer
        params['params']['Encoder_0'] = tok_params['encoder']
        params['params']['VectorQuantizer_0'] = tok_params['quantizer']
        print("  Loaded encoder + quantizer from tokenizer")

        # Load transformer from dynamics if provided
        if pretrained_dynamics is not None:
            dyn_params = pretrained_dynamics['params']['params']
            # Copy pos_embed, TransformerBlocks, LayerNorm (skip Dense_0)
            params['params']['pos_embed'] = dyn_params['pos_embed']
            for i in range(4):
                params['params'][f'TransformerBlock_{i}'] = dyn_params[f'TransformerBlock_{i}']
            params['params']['LayerNorm_0'] = dyn_params['LayerNorm_0']
            print("  Loaded transformer from dynamics model")

        params = flax.core.freeze(params)
        print(f"  VQPolicy ready: vocab={vocab_size}, embed={embed_dim}, tokens={num_tokens}")
    else:
        # Use legacy ViT-based policy
        model = SmallVLMPolicy(
            embed_dim=config.embed_dim,
            vision_layers=config.vision_layers,
            decoder_layers=config.decoder_layers,
            num_heads=config.num_heads,
            num_actions=config.num_actions,
            patch_size=config.patch_size,
        )
        params = model.init(key, dummy)
        print("  Using ViT-based policy (no pretrained weights)")

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(config.learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-model", type=str, required=True,
                        help="Path to trained reward model (CNN or sequence)")
    parser.add_argument("--sequence-reward", action="store_true",
                        help="Use sequence-based reward model (transformer)")
    parser.add_argument("--pretrained-tokenizer", type=str, default=None,
                        help="Path to pretrained tokenizer (encoder + quantizer)")
    parser.add_argument("--pretrained-dynamics", type=str, default=None,
                        help="Path to pretrained dynamics model (transformer)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--rom", default="roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb")
    parser.add_argument("--iterations", type=int, default=12000,
                        help="Total iterations (default 12000 ≈ 12 hours)")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--return-filter", type=float, default=0.0,
                        help="Only train on top X%% of returns (0 = no filtering, 0.8 = top 80%%)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default=None,
                        help="Wandb run ID to resume")
    args = parser.parse_args()

    config = Config(num_envs=args.num_envs)

    # Load reward model
    print(f"Loading reward model: {args.reward_model}")
    use_sequence_reward = args.sequence_reward
    seq_reward_config = None

    if use_sequence_reward:
        reward_model, reward_params, seq_reward_config = load_sequence_reward_model(args.reward_model)
        seq_len = seq_reward_config.seq_len
        frame_skip_reward = seq_reward_config.frame_skip
        print(f"  Sequence reward model: seq_len={seq_len}, frame_skip={frame_skip_reward}")

        # Initialize frame buffer for sequences
        frame_buffer = np.zeros(
            (config.num_envs, seq_len, 144, 160, 3),
            dtype=np.uint8
        )
        frame_buffer_idx = 0

        @jax.jit
        def compute_sequence_rewards(sequences):
            """Compute progress scores for sequences. Shape: (B, seq_len, H, W, C)"""
            # reward_params is already {'params': {...}}, so pass directly
            return reward_model.apply(reward_params, sequences, train=False)

        # Test
        test_seq = jnp.zeros((1, seq_len, 144, 160, 3), dtype=jnp.uint8)
        test_score = compute_sequence_rewards(test_seq)
        print(f"Reward model test score: {float(test_score[0]):.4f}")

        # Wrapper for compatibility
        def compute_rewards(frames):
            """For sequence model, build sequences from recent frames."""
            # This is called with single frames, so we need to handle buffering
            # For now, just duplicate the frame to make a sequence
            B = frames.shape[0]
            seq = jnp.broadcast_to(frames[:, None], (B, seq_len, 144, 160, 3))
            return compute_sequence_rewards(seq)

    else:
        reward_model, reward_params, reward_batch_stats = load_reward_model(args.reward_model)

        @jax.jit
        def compute_rewards(frames):
            """Compute progress scores for a batch of frames."""
            return reward_model.apply(
                {'params': reward_params, 'batch_stats': reward_batch_stats},
                frames, train=False
            )

        # Test reward model
        test_frame = jnp.zeros((1, 144, 160, 3), dtype=jnp.uint8)
        test_score = compute_rewards(test_frame)
        print(f"Reward model test score: {float(test_score[0]):.4f}")

    # Initialize wandb with full config
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"vq-policy-{config.num_envs}env-{int(time.time())}"
            wandb_config = {
                **asdict(config),
                'reward_model': args.reward_model,
                'pretrained_tokenizer': args.pretrained_tokenizer,
                'pretrained_dynamics': args.pretrained_dynamics,
                'return_filter': args.return_filter,
                'total_iterations': args.iterations,
                'architecture': 'VQPolicy' if args.pretrained_tokenizer else 'SmallVLMPolicy',
            }
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=["pokemon-red", "vq-policy", "ppo", "pretrained"],
                resume="allow" if args.wandb_resume else None,
                id=args.wandb_resume,
            )
            print(f"Wandb: {wandb_run.url}")
        except Exception as e:
            print(f"Wandb init failed: {e}")

    # Load emulator
    from emurust import VecGameBoy
    with open(args.rom, 'rb') as f:
        rom_data = np.frombuffer(f.read(), dtype=np.uint8)

    env = VecGameBoy(rom_data, config.num_envs, frame_skip=config.frame_skip)
    obs = np.zeros(env.obs_shape(), dtype=np.uint8)

    # Initialize progress checkpointer for curriculum learning
    checkpointer = ProgressCheckpointer(config)
    checkpoint_rng = np.random.default_rng(42)

    # Initialize novelty tracker for exploration bonus
    novelty_tracker = NoveltyTracker(bucket_size=config.novelty_bucket_size)

    # Initialize policy
    key = jax.random.PRNGKey(int(time.time()))

    # Load pretrained weights
    pretrained_tokenizer = None
    pretrained_dynamics = None

    if args.pretrained_tokenizer:
        print(f"Loading pretrained tokenizer: {args.pretrained_tokenizer}")
        with open(args.pretrained_tokenizer, 'rb') as f:
            pretrained_tokenizer = pickle.load(f)

    if args.pretrained_dynamics:
        print(f"Loading pretrained dynamics: {args.pretrained_dynamics}")
        with open(args.pretrained_dynamics, 'rb') as f:
            pretrained_dynamics = pickle.load(f)

    state = create_train_state(config, key, pretrained_tokenizer, pretrained_dynamics)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Policy parameters: {num_params:,}")

    # Resume from checkpoint if provided
    start_iteration = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        with open(args.resume, 'rb') as f:
            checkpoint = pickle.load(f)
        state = state.replace(params=checkpoint['params'])
        start_iteration = checkpoint.get('iteration', 0)
        print(f"  Resumed at iteration {start_iteration}")

    # JIT compile policy functions - capture apply_fn in closure
    apply_fn = state.apply_fn

    @jax.jit
    def _select_action_impl(params, frames, key):
        logits, value = apply_fn(params, frames, train=False)
        # Clip logits for numerical stability (consistent with ppo_update)
        logits = jnp.clip(logits, -20.0, 20.0)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(len(action)), action]
        return action, log_prob, value

    def _ppo_update_single(state, frames, actions, returns, old_log_probs, rng_key):
        """Single-device PPO update step."""
        def loss_fn(params):
            logits, values = state.apply_fn(params, frames, train=True, rngs={'dropout': rng_key})

            # Clip logits to prevent numerical instability (extreme softmax outputs)
            logits = jnp.clip(logits, -20.0, 20.0)

            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = log_probs[jnp.arange(len(actions)), actions]

            # Clip log prob ratio to prevent extreme importance weights
            log_ratio = jnp.clip(action_log_probs - old_log_probs, -20.0, 2.0)
            ratio = jnp.exp(log_ratio)

            advantages = returns - values
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            # Safe normalization: use max(std, 1.0) to avoid tiny divisors
            # This effectively skips normalization when std is very small
            safe_std = jnp.maximum(adv_std, 1.0)
            advantages = (advantages - adv_mean) / safe_std
            clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
            policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = ((values - returns) ** 2).mean()

            # Stable entropy: use softmax with numerical stability
            # entropy = -sum(p * log(p)) where p = softmax(logits)
            probs = jax.nn.softmax(logits)
            # Clamp log_probs from below to avoid -inf * 0 = NaN
            log_probs_clamped = jnp.maximum(log_probs, -20.0)
            entropy = -(probs * log_probs_clamped).sum(-1).mean()

            total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            return total_loss, {'policy': policy_loss, 'value': value_loss, 'entropy': entropy}

        grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        metrics['grad_norm'] = grad_norm
        return state.apply_gradients(grads=grads), metrics

    # Single GPU training (multi-GPU disabled for stability)
    ppo_update = jax.jit(_ppo_update_single)
    use_pmap = False

    def select_action(state, frames, key):
        return _select_action_impl(state.params, frames, key)

    env.reset(obs)

    # Fast-forward through intro screens to actual gameplay
    # This skips: Game Freak logo, title screen, Oak intro, naming
    print("Fast-forwarding through intro screens...")
    INTRO_STEPS = 6000  # ~3 minutes at 30fps frame_skip
    for step in range(INTRO_STEPS):
        # Alternate Start and A to skip through menus
        if step % 3 == 0:
            buttons = np.full(config.num_envs, 128, dtype=np.uint8)  # Start
        else:
            buttons = np.full(config.num_envs, 16, dtype=np.uint8)   # A
        env.step(buttons, obs)
    print(f"  Skipped {INTRO_STEPS} intro steps, now in gameplay")

    print(f"\nTraining config:")
    print(f"  Iterations: {start_iteration} -> {args.iterations}")
    print(f"  Envs: {config.num_envs}")
    print(f"  Trajectory: {config.trajectory_length} steps")
    print(f"  Reward computed every {config.reward_interval} steps")
    print("=" * 70)

    start_time = time.time()
    total_frames = 0

    for iteration in range(start_iteration, args.iterations):
        t0 = time.time()

        # === ROLLOUT ===
        all_frames = []
        all_actions = []
        all_log_probs = []

        for step in range(config.trajectory_length):
            key, subkey = jax.random.split(key)
            actions, log_probs, _ = select_action(state, obs, subkey)
            actions_np = np.array(actions, dtype=np.uint8)

            all_frames.append(obs.copy())
            all_actions.append(actions_np)
            all_log_probs.append(np.array(log_probs))

            # Map action indices to proper button bitmasks
            button_np = ACTION_TO_BUTTON[actions_np]
            env.step(button_np, obs)

        rollout_time = time.time() - t0
        total_frames += config.num_envs * config.trajectory_length * config.frame_skip

        # === COMPUTE REWARDS (CNN - fast!) ===
        t1 = time.time()

        # Compute progress scores for frames at reward_interval
        reward_indices = list(range(0, config.trajectory_length, config.reward_interval))
        if reward_indices[-1] != config.trajectory_length - 1:
            reward_indices.append(config.trajectory_length - 1)

        # Batch all frames for reward computation
        reward_frames = []
        for idx in reward_indices:
            reward_frames.append(all_frames[idx])  # (num_envs, H, W, C)
        reward_frames = np.concatenate(reward_frames, axis=0)  # (num_indices * num_envs, H, W, C)

        # Detect blank frames and handle them specially
        blank_mask = get_blank_mask(reward_frames)
        non_blank_frames = reward_frames[~blank_mask]

        # Get progress scores for non-blank frames only
        if len(non_blank_frames) > 0:
            non_blank_scores = compute_rewards(jnp.array(non_blank_frames))
            non_blank_scores = np.array(non_blank_scores)
        else:
            non_blank_scores = np.array([])

        # Reconstruct full score array with blank frames getting BLANK_FRAME_SCORE
        all_scores = np.full(len(reward_frames), BLANK_FRAME_SCORE, dtype=np.float32)
        all_scores[~blank_mask] = non_blank_scores

        progress_scores = all_scores.reshape(len(reward_indices), config.num_envs)

        # Update checkpointer with final progress scores (last frame of trajectory)
        final_progress = progress_scores[-1]  # (num_envs,)
        final_frames = all_frames[-1]  # (num_envs, 144, 160, 3)
        new_checkpoints = checkpointer.update(env, final_progress, final_frames)

        # Compute novelty bonus for reaching new score buckets
        novelty_bonuses = np.zeros((len(reward_indices), config.num_envs), dtype=np.float32)
        for t_idx in range(len(reward_indices)):
            novelty_bonuses[t_idx] = novelty_tracker.compute_novelty_bonus(
                progress_scores[t_idx]
            )

        # Convert to per-step rewards via interpolation
        all_rewards = np.zeros((config.num_envs, config.trajectory_length), dtype=np.float32)
        for env_idx in range(config.num_envs):
            scores = progress_scores[:, env_idx]
            full_scores = np.interp(
                range(config.trajectory_length),
                reward_indices,
                scores
            )
            # Progress delta (score change)
            delta = full_scores[1:] - full_scores[:-1]

            # IMPORTANT: Don't penalize going through low-score regions (exploration)
            # Only reward positive progress, ignore negative progress
            # This allows agent to explore through "valleys" to reach peaks
            positive_delta = np.maximum(0, delta)

            # Interpolate novelty bonus to per-step
            novelty = novelty_bonuses[:, env_idx]
            full_novelty = np.interp(
                range(config.trajectory_length),
                reward_indices,
                novelty
            )

            # Reward = positive progress + novelty bonus - time penalty
            all_rewards[env_idx, 1:] = positive_delta + config.novelty_coef * full_novelty[1:] - config.time_penalty

        reward_time = time.time() - t1

        # === TRAINING ===
        t2 = time.time()

        frames_batch = np.stack(all_frames, axis=1)
        actions_batch = np.stack(all_actions, axis=1)
        log_probs_batch = np.stack(all_log_probs, axis=1)

        # Compute returns
        returns_batch = np.zeros_like(all_rewards)
        running_return = np.zeros(config.num_envs)
        for t in reversed(range(config.trajectory_length)):
            running_return = all_rewards[:, t] + config.gamma * running_return
            returns_batch[:, t] = running_return

        # Flatten
        B = config.num_envs * config.trajectory_length
        frames_flat = frames_batch.reshape(B, 144, 160, 3)
        actions_flat = actions_batch.reshape(B)
        log_probs_flat = log_probs_batch.reshape(B)
        returns_flat = returns_batch.reshape(B)

        # PPO updates with minibatching
        # Skip training if returns have no variance (no learning signal)
        returns_std = returns_flat.std()
        if returns_std < 1e-6:
            # No progress detected in this rollout - skip training
            if iteration % config.log_interval == 0:
                print(f"[{iteration:5d}] SKIP - no variance in returns (rewards all zero)")
            continue

        # Apply return filtering if specified (only train on top X% of returns)
        if args.return_filter > 0:
            threshold = np.percentile(returns_flat, (1 - args.return_filter) * 100)
            filter_mask = returns_flat >= threshold
            frames_flat = frames_flat[filter_mask]
            actions_flat = actions_flat[filter_mask]
            returns_flat = returns_flat[filter_mask]
            log_probs_flat = log_probs_flat[filter_mask]
            B = len(returns_flat)

            if B < config.minibatch_size:
                if iteration % config.log_interval == 0:
                    print(f"[{iteration:5d}] SKIP - too few samples after filtering ({B})")
                continue

        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, B)

        metrics_list = []
        for mb_idx, start in enumerate(range(0, B, config.minibatch_size)):
            end = min(start + config.minibatch_size, B)
            idx = perm[start:end]
            key, dropout_key = jax.random.split(key)

            if use_pmap:
                # Reshape data for multi-GPU: (num_devices, batch_per_device, ...)
                mb_size = len(idx)
                # Pad to be divisible by NUM_DEVICES
                pad_size = (NUM_DEVICES - mb_size % NUM_DEVICES) % NUM_DEVICES
                if pad_size > 0:
                    idx = jnp.concatenate([idx, idx[:pad_size]])
                mb_size = len(idx)
                per_device = mb_size // NUM_DEVICES

                frames_mb = frames_flat[idx].reshape(NUM_DEVICES, per_device, 144, 160, 3)
                actions_mb = actions_flat[idx].reshape(NUM_DEVICES, per_device)
                returns_mb = returns_flat[idx].reshape(NUM_DEVICES, per_device)
                log_probs_mb = log_probs_flat[idx].reshape(NUM_DEVICES, per_device)
                dropout_keys = jax.random.split(dropout_key, NUM_DEVICES)

                state, metrics = ppo_update(
                    state, frames_mb, actions_mb, returns_mb, log_probs_mb, dropout_keys
                )
                # Average metrics across devices
                metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
            else:
                state, metrics = ppo_update(
                    state,
                    frames_flat[idx],
                    actions_flat[idx],
                    returns_flat[idx],
                    log_probs_flat[idx],
                    dropout_key
                )
            metrics_list.append(metrics)

        training_time = time.time() - t2
        iter_time = time.time() - t0

        # Maybe reset some environments to checkpoints (curriculum learning)
        steps_this_iter = config.trajectory_length * config.frame_skip
        num_resets = checkpointer.maybe_reset_envs(env, obs, checkpoint_rng, steps_this_iter)

        # Logging
        if iteration % config.log_interval == 0 and len(metrics_list) > 0:
            elapsed = time.time() - start_time
            fps = total_frames / elapsed if elapsed > 0 else 0
            avg_reward = all_rewards.mean()
            avg_entropy = np.mean([float(m['entropy']) for m in metrics_list])
            avg_grad_norm = np.mean([float(m['grad_norm']) for m in metrics_list])
            ckpt_stats = checkpointer.get_stats()
            novelty_stats = novelty_tracker.get_stats()

            prog_range = ckpt_stats['progress_range']
            print(f"[{iteration:5d}] reward={avg_reward:+.4f} | "
                  f"ent={avg_entropy:.3f} | "
                  f"ckpts={ckpt_stats['num_checkpoints']} prog=[{prog_range[0]:.1f},{prog_range[1]:.1f}] | "
                  f"novelty={novelty_stats['visited_buckets']} | "
                  f"{fps/1000:.0f}K fps")

            if wandb_run:
                wandb_run.log({
                    "timing/rollout_sec": rollout_time,
                    "timing/reward_sec": reward_time,
                    "timing/training_sec": training_time,
                    "timing/iter_sec": iter_time,
                    "timing/fps": fps,
                    "reward/mean": avg_reward,
                    "policy/entropy": avg_entropy,
                    "progress/iteration": iteration,
                    "progress/total_frames": total_frames,
                    "checkpoints/num_checkpoints": ckpt_stats['num_checkpoints'],
                    "checkpoints/best_progress": ckpt_stats['best_progress'],
                    "checkpoints/min_progress": prog_range[0],
                    "checkpoints/max_progress": prog_range[1],
                    "checkpoints/resets_this_iter": num_resets,
                }, step=iteration)

        # Checkpoints (save every checkpoint_interval iterations)
        if iteration % config.checkpoint_interval == 0 and iteration > start_iteration:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/policy_{iteration}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'config': asdict(config),
                    'iteration': iteration,
                }, f)
            print(f"  → Saved {checkpoint_path}")

            # Save checkpoint frames for visual review
            frame_dir = f"checkpoints/frames_{iteration}"
            checkpointer.save_checkpoint_frames(frame_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"  Time: {elapsed/3600:.2f} hours")
    print(f"  Frames: {total_frames:,}")
    print(f"  Avg FPS: {total_frames/elapsed:,.0f}")

    # Save final checkpoint frames
    checkpointer.save_checkpoint_frames("checkpoints/frames_final")

    # Save final policy
    # If using pmap, extract single copy of params (they're replicated)
    params_to_save = state.params
    if use_pmap:
        params_to_save = jax.tree_util.tree_map(lambda x: x[0], state.params)

    with open("checkpoints/policy_final.pkl", 'wb') as f:
        pickle.dump({
            'params': params_to_save,
            'config': asdict(config),
            'iteration': args.iterations,
        }, f)
    print(f"  → Saved checkpoints/policy_final.pkl")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
