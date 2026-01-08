#!/usr/bin/env python3
"""
Dreamer 4-style visual pretraining on speedrun frames.

Two-stage training (no actions needed):
  Stage 1A: Tokenizer (VQ-VAE) - encode frames to discrete token grids
  Stage 1B: Dynamics Model - predict next token grid from current

The tokenizer converts 144x160 frames into a 9x10 grid of discrete tokens,
similar to how language models tokenize text. This allows the dynamics
model to predict visual sequences in a discrete token space.

Usage:
    # Train both stages
    uv run python pretrain_visual.py --data data/speedruns/frames --epochs 100

    # Train only tokenizer
    uv run python pretrain_visual.py --data data/speedruns/frames --stage 1a --epochs 100

    # Train dynamics on existing tokenizer
    uv run python pretrain_visual.py --data data/speedruns/frames --stage 1b \
        --tokenizer pretrained_tokenizer.pkl --epochs 50
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
from typing import Tuple
import random
import time
import pickle

# Enable bf16 training
jax.config.update("jax_default_matmul_precision", "bfloat16")


@dataclass
class PretrainConfig:
    # Tokenizer architecture (compact ~5M params for encoder+decoder)
    vocab_size: int = 512           # Codebook size (number of discrete tokens)
    embed_dim: int = 192            # Dimension of each codebook entry (compact)
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 192)  # 4 downsamples

    # Dynamics model architecture (compact ~2M params)
    num_layers: int = 4             # Transformer layers
    num_heads: int = 4
    mlp_ratio: int = 3

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 100
    samples_per_epoch: int = 10000
    use_bf16: bool = True           # Use bfloat16 for faster training

    # Loss weights
    commitment_weight: float = 0.25  # VQ commitment loss weight

    # Data
    frame_shape: Tuple[int, int, int] = (144, 160, 3)
    token_grid: Tuple[int, int] = (9, 10)  # After 4x downsample: 144/16=9, 160/16=10


# ============================================================================
# STAGE 1A: VQ-VAE TOKENIZER
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
        # Normalize to [0, 1]
        x = x.astype(jnp.float32) / 255.0

        # Downsample blocks: 144x160 -> 72x80 -> 36x40 -> 18x20 -> 9x10
        for i, ch in enumerate(self.channels):
            x = nn.Conv(ch, (4, 4), strides=(2, 2), padding='SAME')(x)
            x = ResBlock(ch)(x, train)

        # Project to embedding dimension
        x = nn.Conv(self.embed_dim, (1, 1))(x)

        return x  # Shape: (B, 9, 10, embed_dim)


class Decoder(nn.Module):
    """Decode from spatial embeddings to frame."""
    channels: Tuple[int, ...] = (192, 128, 64, 32)
    output_channels: int = 3

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Upsample blocks: 9x10 -> 18x20 -> 36x40 -> 72x80 -> 144x160
        for i, ch in enumerate(self.channels):
            x = nn.ConvTranspose(ch, (4, 4), strides=(2, 2), padding='SAME')(x)
            x = ResBlock(ch)(x, train)

        # Final conv to RGB
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.silu(x)
        x = nn.Conv(self.output_channels, (3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)  # Output in [0, 1]

        return x


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer (VQ-VAE style).

    Maps continuous encoder outputs to nearest codebook entries.
    Uses straight-through gradient estimator.
    """
    vocab_size: int = 512
    embed_dim: int = 192

    @nn.compact
    def __call__(self, z, train: bool = True):
        # z shape: (B, H, W, D)
        B, H, W, D = z.shape

        # Codebook: (vocab_size, embed_dim)
        codebook = self.param(
            'codebook',
            nn.initializers.variance_scaling(1.0, 'fan_in', 'uniform'),
            (self.vocab_size, self.embed_dim)
        )

        # Flatten spatial dims for quantization
        z_flat = z.reshape(-1, D)  # (B*H*W, D)

        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        z_sq = jnp.sum(z_flat ** 2, axis=1, keepdims=True)  # (B*H*W, 1)
        e_sq = jnp.sum(codebook ** 2, axis=1, keepdims=True).T  # (1, vocab_size)
        distances = z_sq + e_sq - 2 * jnp.dot(z_flat, codebook.T)  # (B*H*W, vocab_size)

        # Get nearest codebook indices
        indices = jnp.argmin(distances, axis=1)  # (B*H*W,)
        indices = indices.reshape(B, H, W)  # (B, H, W)

        # Get quantized embeddings
        z_q = codebook[indices.reshape(-1)]  # (B*H*W, D)
        z_q = z_q.reshape(B, H, W, D)

        # Compute losses
        # Codebook loss: move codebook entries toward encoder outputs
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z) - z_q) ** 2)
        # Commitment loss: keep encoder outputs close to codebook
        commitment_loss = jnp.mean((z - jax.lax.stop_gradient(z_q)) ** 2)

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + jax.lax.stop_gradient(z_q - z)

        return z_q, indices, codebook_loss, commitment_loss


class Tokenizer(nn.Module):
    """VQ-VAE tokenizer: frame -> discrete token grid -> reconstructed frame."""
    config: PretrainConfig

    def setup(self):
        self.encoder = Encoder(
            channels=self.config.encoder_channels,
            embed_dim=self.config.embed_dim,
        )
        self.quantizer = VectorQuantizer(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
        )
        self.decoder = Decoder(
            channels=self.config.encoder_channels[::-1],
            output_channels=3,
        )

    def __call__(self, x, train: bool = True):
        # Encode to continuous embeddings
        z = self.encoder(x, train)

        # Quantize to discrete tokens
        z_q, tokens, codebook_loss, commitment_loss = self.quantizer(z, train)

        # Decode to reconstructed frame
        recon = self.decoder(z_q, train)

        return recon, tokens, z, codebook_loss, commitment_loss

    def encode(self, x, train: bool = False):
        """Encode frame to token grid."""
        z = self.encoder(x, train)
        z_q, tokens, _, _ = self.quantizer(z, train)
        return tokens  # (B, H, W) integer tokens

    def decode_tokens(self, tokens, train: bool = False):
        """Decode token grid to frame."""
        # Get codebook
        codebook = self.quantizer.variables['params']['codebook']
        # Look up embeddings
        z_q = codebook[tokens]  # (B, H, W, D)
        # Decode
        return self.decoder(z_q, train)

    def get_embeddings(self, tokens):
        """Get continuous embeddings for tokens (for dynamics model input)."""
        codebook = self.quantizer.variables['params']['codebook']
        return codebook[tokens]  # (B, H, W, D)


# ============================================================================
# STAGE 1B: DYNAMICS MODEL (Transformer over tokens)
# ============================================================================

class TransformerBlock(nn.Module):
    """Standard transformer block."""
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

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


class DynamicsModel(nn.Module):
    """
    Transformer that predicts next token grid from current.

    Input: token embeddings (B, H*W, D)
    Output: logits for next tokens (B, H*W, vocab_size)
    """
    vocab_size: int = 512
    embed_dim: int = 192
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 3
    num_tokens: int = 90  # 9 * 10

    @nn.compact
    def __call__(self, token_embeds, train: bool = True):
        B, N, D = token_embeds.shape

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

        # Output projection to vocab logits
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)  # (B, N, vocab_size)

        return logits


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
                if len(frames) > 10:
                    self.videos.append(frames)

        total_frames = sum(len(v) for v in self.videos)
        print(f"Loaded {len(self.videos)} videos with {total_frames:,} frames")

    def sample_frames(self, batch_size: int) -> np.ndarray:
        """Sample random frames for tokenizer training."""
        frames = []
        for _ in range(batch_size):
            video = random.choice(self.videos)
            idx = random.randint(0, len(video) - 1)
            frames.append(np.load(video[idx]))
        return np.stack(frames)

    def sample_frame_pairs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample consecutive frame pairs for dynamics training."""
        frames_t = []
        frames_t1 = []
        for _ in range(batch_size):
            video = random.choice(self.videos)
            idx = random.randint(0, len(video) - 2)
            frames_t.append(np.load(video[idx]))
            frames_t1.append(np.load(video[idx + 1]))
        return np.stack(frames_t), np.stack(frames_t1)


# ============================================================================
# TRAINING STAGE 1A: TOKENIZER
# ============================================================================

def create_tokenizer_state(config: PretrainConfig, key):
    """Initialize tokenizer and optimizer."""
    model = Tokenizer(config)

    dummy = jnp.zeros((1, *config.frame_shape), dtype=jnp.uint8)
    params = model.init(key, dummy)

    # Use lower LR for codebook stability
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=0.01),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), model


def train_tokenizer(config: PretrainConfig, dataset: SpeedrunFrameDataset,
                    output_path: str = "pretrained_tokenizer.pkl"):
    """Train Stage 1A: VQ-VAE Tokenizer."""
    print("\n" + "=" * 70)
    print("STAGE 1A: Training VQ-VAE Tokenizer")
    print("=" * 70)
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Token grid: {config.token_grid[0]}x{config.token_grid[1]} = {config.token_grid[0] * config.token_grid[1]} tokens/frame")

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)

    state, tokenizer = create_tokenizer_state(config, init_key)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Parameters: {num_params:,}")

    # JIT compile training step
    @jax.jit
    def train_step(state, frames):
        def loss_fn(params):
            recon, tokens, z, codebook_loss, commitment_loss = tokenizer.apply(
                params, frames, train=True
            )

            # Reconstruction loss
            target = frames.astype(jnp.float32) / 255.0
            recon_loss = jnp.mean((recon - target) ** 2)

            # Total VQ-VAE loss
            total_loss = recon_loss + codebook_loss + config.commitment_weight * commitment_loss

            return total_loss, {
                'recon': recon_loss,
                'codebook': codebook_loss,
                'commit': commitment_loss,
                'tokens': tokens,  # Return tokens for usage calculation outside JIT
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    steps_per_epoch = config.samples_per_epoch // config.batch_size

    print(f"\nTraining for {config.epochs} epochs ({steps_per_epoch} steps/epoch)")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Recon':>8} | {'VQ':>8} | {'Commit':>8} | {'Usage':>6} | {'Time':>5}")
    print("-" * 70)

    for epoch in range(config.epochs):
        t0 = time.time()
        epoch_metrics = {'recon': 0, 'codebook': 0, 'commit': 0}
        all_tokens = []

        for step in range(steps_per_epoch):
            frames = dataset.sample_frames(config.batch_size)
            state, metrics = train_step(state, jnp.array(frames))

            epoch_metrics['recon'] += float(metrics['recon'])
            epoch_metrics['codebook'] += float(metrics['codebook'])
            epoch_metrics['commit'] += float(metrics['commit'])

            # Collect tokens for usage calculation (only every 10 steps to save memory)
            if step % 10 == 0:
                all_tokens.append(np.array(metrics['tokens']).flatten())

        for k in epoch_metrics:
            epoch_metrics[k] /= steps_per_epoch

        # Calculate codebook usage
        if all_tokens:
            all_tokens = np.concatenate(all_tokens)
            usage = len(np.unique(all_tokens)) / config.vocab_size
        else:
            usage = 0.0

        epoch_time = time.time() - t0

        print(f"{epoch+1:5d} | {epoch_metrics['recon']:8.5f} | "
              f"{epoch_metrics['codebook']:8.5f} | {epoch_metrics['commit']:8.5f} | "
              f"{usage:5.1%} | {epoch_time:4.1f}s")

        if (epoch + 1) % 20 == 0 or epoch == config.epochs - 1:
            save_tokenizer(state.params, config, output_path)
            print(f"  → Saved checkpoint to {output_path}")

    print("=" * 70)
    print("Tokenizer training complete!")

    return state.params


def save_tokenizer(params, config: PretrainConfig, path: str):
    """Save tokenizer weights."""
    with open(path, 'wb') as f:
        pickle.dump({
            'params': params,
            'config': asdict(config),
        }, f)


def load_tokenizer(path: str) -> Tuple[dict, PretrainConfig]:
    """Load tokenizer weights."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    config = PretrainConfig(**{
        k: v for k, v in data['config'].items()
        if k in PretrainConfig.__dataclass_fields__
    })
    return data['params'], config


# ============================================================================
# TRAINING STAGE 1B: DYNAMICS MODEL
# ============================================================================

def create_dynamics_state(config: PretrainConfig, key):
    """Initialize dynamics model and optimizer."""
    num_tokens = config.token_grid[0] * config.token_grid[1]

    model = DynamicsModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        num_tokens=num_tokens,
    )

    dummy = jnp.zeros((1, num_tokens, config.embed_dim))
    params = model.init(key, dummy)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=0.01),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), model


def train_dynamics(config: PretrainConfig, dataset: SpeedrunFrameDataset,
                   tokenizer_params: dict,
                   output_path: str = "pretrained_dynamics.pkl"):
    """Train Stage 1B: Dynamics Model."""
    print("\n" + "=" * 70)
    print("STAGE 1B: Training Dynamics Model (Transformer)")
    print("=" * 70)
    num_tokens = config.token_grid[0] * config.token_grid[1]
    print(f"  Transformer: {config.num_layers} layers, {config.num_heads} heads")
    print(f"  Sequence length: {num_tokens} tokens")

    key = jax.random.PRNGKey(43)
    key, init_key = jax.random.split(key)

    # Create frozen tokenizer for encoding
    tokenizer = Tokenizer(config)

    # Create dynamics model
    state, dynamics = create_dynamics_state(config, init_key)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Parameters: {num_params:,}")

    # Get codebook from tokenizer
    codebook = tokenizer_params['params']['quantizer']['codebook']

    # JIT compile encoding (frozen tokenizer)
    @jax.jit
    def encode_to_tokens(frames):
        """Encode frames to token indices."""
        _, tokens, _, _, _ = tokenizer.apply(tokenizer_params, frames, train=False)
        return tokens

    @jax.jit
    def tokens_to_embeddings(tokens, codebook):
        """Convert token indices to embeddings."""
        B, H, W = tokens.shape
        embeds = codebook[tokens.reshape(-1)]
        return embeds.reshape(B, H * W, -1)

    # JIT compile training step
    @jax.jit
    def train_step(state, embeds_t, tokens_t1, key):
        """
        Train to predict next frame's tokens from current frame's embeddings.
        """
        def loss_fn(params):
            logits = dynamics.apply(params, embeds_t, train=True, rngs={'dropout': key})
            # Cross-entropy loss for each token position
            targets = tokens_t1.reshape(tokens_t1.shape[0], -1)  # (B, H*W)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            loss = jnp.mean(loss)

            # Accuracy
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.mean(preds == targets)

            return loss, acc

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, acc

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
            frames_t, frames_t1 = dataset.sample_frame_pairs(config.batch_size)

            # Encode both frames to tokens
            tokens_t = encode_to_tokens(jnp.array(frames_t))
            tokens_t1 = encode_to_tokens(jnp.array(frames_t1))

            # Get embeddings for current frame
            embeds_t = tokens_to_embeddings(tokens_t, codebook)

            key, step_key = jax.random.split(key)
            state, loss, acc = train_step(state, embeds_t, tokens_t1, step_key)
            epoch_loss += float(loss)
            epoch_acc += float(acc)

        epoch_loss /= steps_per_epoch
        epoch_acc /= steps_per_epoch
        epoch_time = time.time() - t0

        print(f"{epoch+1:5d} | {epoch_loss:10.4f} | {epoch_acc:10.2%} | {epoch_time:4.1f}s")

        if (epoch + 1) % 20 == 0 or epoch == config.epochs - 1:
            save_dynamics(state.params, config, output_path)
            print(f"  → Saved checkpoint to {output_path}")

    print("=" * 70)
    print("Dynamics training complete!")

    return state.params


def save_dynamics(params, config: PretrainConfig, path: str):
    """Save dynamics model weights."""
    with open(path, 'wb') as f:
        pickle.dump({
            'params': params,
            'config': asdict(config),
        }, f)


# ============================================================================
# EXPORT FOR RL TRAINING
# ============================================================================

def export_encoder_for_rl(tokenizer_params: dict, config: PretrainConfig,
                          output_path: str = "pretrained_encoder.pkl"):
    """
    Export encoder + codebook for use in RL policy.

    The RL policy can use this to get token embeddings as state representation.
    """
    encoder_params = tokenizer_params['params']['encoder']
    quantizer_params = tokenizer_params['params']['quantizer']

    with open(output_path, 'wb') as f:
        pickle.dump({
            'encoder_params': encoder_params,
            'quantizer_params': quantizer_params,
            'config': {
                'vocab_size': config.vocab_size,
                'embed_dim': config.embed_dim,
                'encoder_channels': config.encoder_channels,
                'token_grid': config.token_grid,
                'frame_shape': config.frame_shape,
            }
        }, f)

    print(f"Exported encoder for RL to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dreamer 4-style visual pretraining")
    parser.add_argument("--data", type=str, default="data/speedruns/frames")
    parser.add_argument("--stage", type=str, default="both", choices=["1a", "1b", "both"])
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to pretrained tokenizer (for stage 1b)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-tokenizer", type=str, default="pretrained_tokenizer.pkl")
    parser.add_argument("--output-dynamics", type=str, default="pretrained_dynamics.pkl")
    parser.add_argument("--output-encoder", type=str, default="pretrained_encoder.pkl")
    args = parser.parse_args()

    config = PretrainConfig(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
    )

    print("=" * 70)
    print("DREAMER 4-STYLE VISUAL PRETRAINING")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Stage: {args.stage}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embed dim: {config.embed_dim}")
    print(f"Token grid: {config.token_grid}")

    print("\nLoading dataset...")
    dataset = SpeedrunFrameDataset(args.data, config)

    if len(dataset.videos) == 0:
        print("ERROR: No training data found!")
        return

    tokenizer_params = None

    # Stage 1A: Train Tokenizer
    if args.stage in ["1a", "both"]:
        tokenizer_params = train_tokenizer(config, dataset, args.output_tokenizer)
        export_encoder_for_rl(tokenizer_params, config, args.output_encoder)

    # Stage 1B: Train Dynamics
    if args.stage in ["1b", "both"]:
        if tokenizer_params is None:
            if args.tokenizer is None:
                print("ERROR: --tokenizer required for stage 1b")
                return
            print(f"\nLoading tokenizer from {args.tokenizer}")
            tokenizer_params, loaded_config = load_tokenizer(args.tokenizer)
            config = loaded_config

        train_dynamics(config, dataset, tokenizer_params, args.output_dynamics)

    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    if args.stage in ["1a", "both"]:
        print(f"  Tokenizer: {args.output_tokenizer}")
        print(f"  Encoder: {args.output_encoder}")
    if args.stage in ["1b", "both"]:
        print(f"  Dynamics: {args.output_dynamics}")

    print(f"\nNext: Run RL with pretrained encoder:")
    print(f"  uv run python train_with_cnn_reward.py \\")
    print(f"      --reward-model reward_resnet34.pkl \\")
    print(f"      --pretrained-encoder {args.output_encoder}")


if __name__ == "__main__":
    main()
