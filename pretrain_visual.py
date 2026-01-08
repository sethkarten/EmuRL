#!/usr/bin/env python3
"""
Pretrain visual encoder on speedrun frames (DreamerV4-style).

No actions needed - just learns visual representations from frame sequences.

Approaches:
1. Autoencoder - reconstruct frames
2. Next frame prediction - predict t+1 from t
3. Contrastive (temporal) - frames close in time should be similar

Usage:
    python pretrain_visual.py --data data/speedruns/frames --epochs 50
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
from dataclasses import dataclass
from typing import List, Tuple
import random
import time
import pickle


@dataclass
class PretrainConfig:
    # Model
    embed_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    patch_size: int = 16

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 50
    samples_per_epoch: int = 10000

    # Contrastive
    temperature: float = 0.1
    temporal_window: int = 30  # Frames within this window are "positive"

    # Data
    frame_shape: Tuple[int, int, int] = (144, 160, 3)


# ============================================================================
# MODEL COMPONENTS (matching policy network architecture)
# ============================================================================

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID'
        )(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block."""
    embed_dim: int
    num_heads: int

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

        # FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.embed_dim * 4)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim)(y)
        y = nn.Dropout(0.1, deterministic=not train)(y)
        x = x + y

        return x


class VisionEncoder(nn.Module):
    """ViT-style encoder - matches policy network."""
    embed_dim: int
    num_layers: int
    num_heads: int
    patch_size: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Patch embedding
        x = PatchEmbed(self.patch_size, self.embed_dim)(x)

        # Add positional embedding
        num_patches = x.shape[1]
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, num_patches, self.embed_dim)
        )
        x = x + pos_embed

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads)(x, train)

        x = nn.LayerNorm()(x)

        # Global average pooling -> single vector
        x = x.mean(axis=1)

        return x


class FrameDecoder(nn.Module):
    """Decode latent back to frame (for reconstruction loss)."""
    embed_dim: int
    patch_size: int
    output_shape: Tuple[int, int, int] = (144, 160, 3)

    @nn.compact
    def __call__(self, z, train: bool = True):
        H, W, C = self.output_shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Project to patch tokens
        x = nn.Dense(num_patches * self.embed_dim)(z)
        x = x.reshape(-1, num_patches, self.embed_dim)

        # Transformer blocks for decoding
        for _ in range(2):
            x = TransformerBlock(self.embed_dim, 4)(x, train)

        # Project each patch to pixels
        patch_pixels = self.patch_size * self.patch_size * C
        x = nn.Dense(patch_pixels)(x)

        # Reshape to image
        x = x.reshape(-1, num_patches_h, num_patches_w, self.patch_size, self.patch_size, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape(-1, H, W, C)

        return x


class VisualPretrainModel(nn.Module):
    """Combined model for pretraining."""
    config: PretrainConfig

    @nn.compact
    def __call__(self, frames, train: bool = True):
        # Encode
        encoder = VisionEncoder(
            self.config.embed_dim,
            self.config.num_layers,
            self.config.num_heads,
            self.config.patch_size
        )
        z = encoder(frames, train)

        # Project for contrastive learning
        z_proj = nn.Dense(self.config.embed_dim)(z)
        z_proj = nn.relu(z_proj)
        z_proj = nn.Dense(self.config.embed_dim)(z_proj)
        z_proj = z_proj / (jnp.linalg.norm(z_proj, axis=-1, keepdims=True) + 1e-8)

        # Decode for reconstruction
        decoder = FrameDecoder(self.config.embed_dim, self.config.patch_size)
        recon = decoder(z, train)

        return z, z_proj, recon


# ============================================================================
# DATASET
# ============================================================================

class SpeedrunFrameDataset:
    """Dataset for pretraining on speedrun frames."""

    def __init__(self, data_dir: Path, config: PretrainConfig):
        self.config = config
        self.data_dir = Path(data_dir)

        # Load all videos with frame sequences
        self.videos = []
        for index_path in sorted(self.data_dir.glob("*_index.json")):
            with open(index_path) as f:
                index = json.load(f)
                frames = []
                for entry in index['frames']:
                    frame_path = self.data_dir / entry['path']
                    if frame_path.exists():
                        frames.append(frame_path)
                if len(frames) > config.temporal_window * 2:
                    self.videos.append(frames)

        total_frames = sum(len(v) for v in self.videos)
        print(f"Loaded {len(self.videos)} videos with {total_frames} frames")

    def sample_contrastive_batch(self, batch_size: int):
        """
        Sample pairs for contrastive learning.

        Returns:
            anchor_frames: (B, H, W, C)
            positive_frames: (B, H, W, C) - temporally close to anchor
            negative_frames: (B, H, W, C) - from different video/far away
        """
        anchors = []
        positives = []
        negatives = []

        for _ in range(batch_size):
            # Pick random video and anchor frame
            video = random.choice(self.videos)
            n = len(video)
            anchor_idx = random.randint(self.config.temporal_window, n - self.config.temporal_window - 1)

            # Positive: within temporal window
            pos_offset = random.randint(1, self.config.temporal_window)
            if random.random() < 0.5:
                pos_offset = -pos_offset
            pos_idx = anchor_idx + pos_offset

            # Negative: different video or far away
            if random.random() < 0.5 and len(self.videos) > 1:
                # Different video
                neg_video = random.choice([v for v in self.videos if v != video])
                neg_idx = random.randint(0, len(neg_video) - 1)
                neg_frame = np.load(neg_video[neg_idx])
            else:
                # Same video, far away
                far_indices = list(range(0, anchor_idx - self.config.temporal_window * 2)) + \
                              list(range(anchor_idx + self.config.temporal_window * 2, n))
                if far_indices:
                    neg_idx = random.choice(far_indices)
                    neg_frame = np.load(video[neg_idx])
                else:
                    neg_video = random.choice(self.videos)
                    neg_idx = random.randint(0, len(neg_video) - 1)
                    neg_frame = np.load(neg_video[neg_idx])

            anchors.append(np.load(video[anchor_idx]))
            positives.append(np.load(video[pos_idx]))
            negatives.append(neg_frame)

        return np.stack(anchors), np.stack(positives), np.stack(negatives)

    def sample_reconstruction_batch(self, batch_size: int):
        """Sample random frames for reconstruction."""
        frames = []
        for _ in range(batch_size):
            video = random.choice(self.videos)
            idx = random.randint(0, len(video) - 1)
            frames.append(np.load(video[idx]))
        return np.stack(frames)

    def sample_sequence_batch(self, batch_size: int, seq_len: int = 4):
        """Sample frame sequences for next-frame prediction."""
        sequences = []
        for _ in range(batch_size):
            video = random.choice(self.videos)
            start = random.randint(0, len(video) - seq_len)
            seq = [np.load(video[start + i]) for i in range(seq_len)]
            sequences.append(np.stack(seq))
        return np.stack(sequences)  # (B, T, H, W, C)


# ============================================================================
# TRAINING
# ============================================================================

def create_train_state(config: PretrainConfig, key):
    """Initialize model and optimizer."""
    model = VisualPretrainModel(config)

    dummy = jnp.zeros((1, *config.frame_shape), dtype=jnp.uint8)
    params = model.init(key, dummy)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=0.01),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state, anchors, positives, negatives, key):
    """
    Combined contrastive + reconstruction loss.
    """
    def loss_fn(params):
        # Forward pass on all frames
        all_frames = jnp.concatenate([anchors, positives, negatives], axis=0)
        z, z_proj, recon = state.apply_fn(params, all_frames, train=True, rngs={'dropout': key})

        B = anchors.shape[0]
        z_anchor = z_proj[:B]
        z_pos = z_proj[B:2*B]
        z_neg = z_proj[2*B:]

        # Contrastive loss (InfoNCE style)
        # Positive similarity
        pos_sim = jnp.sum(z_anchor * z_pos, axis=-1) / 0.1  # temperature
        # Negative similarity
        neg_sim = jnp.sum(z_anchor * z_neg, axis=-1) / 0.1

        # Combined logits: [pos, neg]
        logits = jnp.stack([pos_sim, neg_sim], axis=-1)
        labels = jnp.zeros(B, dtype=jnp.int32)  # Positive is index 0
        contrastive_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        # Reconstruction loss (only on anchor frames)
        recon_anchor = recon[:B]
        target = anchors.astype(jnp.float32) / 255.0
        recon_loss = jnp.mean((recon_anchor - target) ** 2)

        # Total loss
        total_loss = contrastive_loss + 0.1 * recon_loss

        # Metrics
        accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()

        return total_loss, {
            'contrastive_loss': contrastive_loss,
            'recon_loss': recon_loss,
            'accuracy': accuracy,
        }

    grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


def extract_encoder_params(full_params):
    """Extract just the encoder parameters for transfer to policy."""
    # The encoder is under 'VisionEncoder_0' in the params tree
    encoder_params = full_params['params']['VisionEncoder_0']
    return encoder_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/speedruns/frames")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="pretrained_encoder.pkl")
    args = parser.parse_args()

    config = PretrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print("Loading dataset...")
    dataset = SpeedrunFrameDataset(args.data, config)

    if len(dataset.videos) == 0:
        print("No training data found!")
        return

    print(f"\nInitializing model...")
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    state = create_train_state(config, init_key)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}")

    print(f"\nPretraining for {config.epochs} epochs...")
    print("=" * 70)
    print(f"{'Epoch':>5} | {'Contrast':>9} | {'Recon':>9} | {'Accuracy':>8} | {'Time':>6}")
    print("-" * 70)

    steps_per_epoch = config.samples_per_epoch // config.batch_size

    for epoch in range(config.epochs):
        t0 = time.time()
        epoch_metrics = {'contrastive_loss': 0, 'recon_loss': 0, 'accuracy': 0}

        for step in range(steps_per_epoch):
            anchors, positives, negatives = dataset.sample_contrastive_batch(config.batch_size)

            key, step_key = jax.random.split(key)
            state, metrics = train_step(
                state,
                jnp.array(anchors),
                jnp.array(positives),
                jnp.array(negatives),
                step_key
            )

            for k, v in metrics.items():
                epoch_metrics[k] += float(v)

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= steps_per_epoch

        epoch_time = time.time() - t0

        print(f"{epoch+1:5d} | {epoch_metrics['contrastive_loss']:9.4f} | "
              f"{epoch_metrics['recon_loss']:9.4f} | {epoch_metrics['accuracy']:8.1%} | "
              f"{epoch_time:5.1f}s")

    print("=" * 70)
    print("Pretraining complete!")

    # Save encoder weights
    encoder_params = extract_encoder_params(state.params)

    with open(args.output, 'wb') as f:
        pickle.dump({
            'encoder_params': encoder_params,
            'config': {
                'embed_dim': config.embed_dim,
                'num_layers': config.num_layers,
                'num_heads': config.num_heads,
                'patch_size': config.patch_size,
            }
        }, f)

    print(f"Saved pretrained encoder to: {args.output}")


if __name__ == "__main__":
    main()
