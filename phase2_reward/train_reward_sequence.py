#!/usr/bin/env python3
"""
Train reward model using transformer backbone with SEQUENCE context.

Key improvements over single-frame reward models:
1. Takes sequence of frames (e.g., 8-16 frames) as input
2. Uses transformer to reason over temporal context
3. Can distinguish "moving toward goal" from "walking in circles"
4. Massive combinatorial sampling for global progress understanding

Architecture:
  Frames -> VQ Encoder -> Token Sequences -> Transformer -> Pool -> Score

Usage:
    uv run python train_reward_sequence.py \
        --data data/speedruns/frames \
        --tokenizer pretrained_tokenizer.pkl \
        --dynamics pretrained_dynamics.pkl \
        --seq-len 8 \
        --pairs-per-epoch 100000 \
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
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import random
import time
import pickle

jax.config.update("jax_default_matmul_precision", "bfloat16")


@dataclass
class SequenceRewardConfig:
    # Architecture (matching pretrain_visual.py)
    vocab_size: int = 512
    embed_dim: int = 192
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 192)
    token_grid: Tuple[int, int] = (9, 10)

    # Transformer
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 3

    # Sequence
    seq_len: int = 8  # Number of frames in sequence
    frame_skip: int = 10  # Frames between samples (at 2fps = 5 seconds)

    # Reward head
    hidden_dim: int = 256

    # Training
    batch_size: int = 32  # Smaller batch due to sequence
    learning_rate: float = 1e-4
    epochs: int = 50
    pairs_per_epoch: int = 100000  # MASSIVE combinatorial sampling

    # Margin loss
    use_margin: bool = True
    margin_scale: float = 0.1

    # Hard negatives
    hard_negative_ratio: float = 0.2

    # Data
    frame_shape: Tuple[int, int, int] = (144, 160, 3)


# ============================================================================
# VQ-VAE ENCODER (matching pretrain_visual.py)
# ============================================================================

class ResBlock(nn.Module):
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
    channels: Tuple[int, ...] = (32, 64, 128, 192)
    embed_dim: int = 192

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0
        for ch in self.channels:
            x = nn.Conv(ch, (4, 4), strides=(2, 2), padding='SAME')(x)
            x = ResBlock(ch)(x, train)
        x = nn.Conv(self.embed_dim, (1, 1))(x)
        return x


class VectorQuantizer(nn.Module):
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
        z_q = z + jax.lax.stop_gradient(z_q - z)
        return z_q, indices


# ============================================================================
# TRANSFORMER (matching pretrain_visual.py dynamics model)
# ============================================================================

class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 3

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
        )(x, x)
        x = x + attn_out
        x = nn.LayerNorm()(x)

        # MLP
        mlp_dim = self.embed_dim * self.mlp_ratio
        h = nn.Dense(mlp_dim)(x)
        h = nn.gelu(h)
        h = nn.Dropout(0.1, deterministic=not train)(h)
        h = nn.Dense(self.embed_dim)(h)
        h = nn.Dropout(0.1, deterministic=not train)(h)
        x = x + h
        x = nn.LayerNorm()(x)

        return x


class SequenceTransformer(nn.Module):
    """Transformer that processes sequences of frame tokens."""
    embed_dim: int = 192
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 3
    tokens_per_frame: int = 90  # 9x10

    @nn.compact
    def __call__(self, token_embeds, train: bool = True):
        """
        Args:
            token_embeds: (B, seq_len, tokens_per_frame, embed_dim)
        Returns:
            pooled: (B, embed_dim) sequence representation
        """
        B, S, T, D = token_embeds.shape

        # Flatten sequence dimension: (B, seq_len * tokens_per_frame, embed_dim)
        x = token_embeds.reshape(B, S * T, D)

        # Add positional encoding
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, S * T, D)
        )
        x = x + pos_embed

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(x, train)

        # Global average pooling
        pooled = jnp.mean(x, axis=1)  # (B, D)

        return pooled


# ============================================================================
# SEQUENCE REWARD MODEL
# ============================================================================

class SequenceRewardModel(nn.Module):
    """Reward model that takes sequence of frames."""
    config: SequenceRewardConfig

    def setup(self):
        self.encoder = Encoder(
            channels=self.config.encoder_channels,
            embed_dim=self.config.embed_dim,
        )
        self.quantizer = VectorQuantizer(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
        )
        self.transformer = SequenceTransformer(
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            tokens_per_frame=self.config.token_grid[0] * self.config.token_grid[1],
        )
        self.reward_head = nn.Dense(1)

    def __call__(self, frames, train: bool = True):
        """
        Args:
            frames: (B, seq_len, H, W, C) sequence of frames
        Returns:
            scores: (B,) reward scores
        """
        B, S, H, W, C = frames.shape

        # Encode each frame
        frames_flat = frames.reshape(B * S, H, W, C)
        z = self.encoder(frames_flat, train=False)  # Freeze encoder
        z_q, _ = self.quantizer(z, train=False)

        # Reshape to sequence
        _, Hf, Wf, D = z_q.shape
        token_embeds = z_q.reshape(B, S, Hf * Wf, D)

        # Stop gradients to encoder
        token_embeds = jax.lax.stop_gradient(token_embeds)

        # Transformer processing
        pooled = self.transformer(token_embeds, train)

        # Reward score
        score = self.reward_head(pooled).squeeze(-1)

        return score


# ============================================================================
# DATASET WITH MASSIVE COMBINATORIAL SAMPLING
# ============================================================================

def is_blank_frame(frame: np.ndarray) -> bool:
    if frame.max() - frame.min() < 20:
        return True
    if frame.var() < 100:
        return True
    return False


class SequencePairDataset:
    """Dataset that samples SEQUENCE pairs with massive combinatorial coverage."""

    def __init__(self, data_dir: str, config: SequenceRewardConfig, split: str = 'train'):
        self.config = config
        self.data_dir = Path(data_dir)

        # Load videos
        self.videos = []
        self._load_videos(split)

        # Load hard negatives
        self._load_hard_negatives()

    def _load_videos(self, split: str, test_ratio: float = 0.1):
        """Load video frame paths."""
        all_videos = []

        for video_dir in sorted(self.data_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            frame_files = sorted([f for f in video_dir.iterdir() if f.suffix == '.npy'])
            if len(frame_files) < 100:
                continue

            frames = []
            for f in frame_files:
                name = f.stem
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        frame_idx = int(parts[1])
                        frames.append({'path': str(f), 'idx': frame_idx})
                    except ValueError:
                        continue

            if len(frames) > 100:
                all_videos.append({
                    'frames': sorted(frames, key=lambda x: x['idx']),
                    'video_id': video_dir.name,
                })

        n_test = max(1, int(len(all_videos) * test_ratio))
        if split == 'train':
            self.videos = all_videos[n_test:]
        else:
            self.videos = all_videos[:n_test]

        total_frames = sum(len(v['frames']) for v in self.videos)
        print(f"[{split}] Loaded {len(self.videos)} videos with {total_frames:,} frames")

    def _load_hard_negatives(self):
        """Load intro frames as hard negatives."""
        self.intro_frames = []
        # data_dir is data/speedruns/frames, we need data/
        base_dir = self.data_dir.parent.parent

        for subdir in ['intro_frames', 'intro_frames_expanded', 'intro_frames_comprehensive']:
            intro_dir = base_dir / subdir
            if intro_dir.exists():
                for f in intro_dir.iterdir():
                    if f.suffix == '.npy':
                        self.intro_frames.append(str(f))

        self.name_screens = []
        name_dir = base_dir / 'name_screens'
        if name_dir.exists():
            for f in name_dir.iterdir():
                if f.suffix == '.npy':
                    self.name_screens.append(str(f))

        print(f"  Intro frames (hard negative): {len(self.intro_frames):,}")
        print(f"  Name screens (hard negative): {len(self.name_screens):,}")

    def _sample_sequence(self, video: dict, start_idx: int) -> Optional[np.ndarray]:
        """Sample a sequence of frames starting at given index."""
        frames = video['frames']
        n = len(frames)

        # Get sequence with frame_skip spacing
        seq_frames = []
        for i in range(self.config.seq_len):
            idx = start_idx + i * self.config.frame_skip
            if idx >= n:
                return None

            frame = np.load(frames[idx]['path'])
            if is_blank_frame(frame):
                return None
            seq_frames.append(frame)

        return np.stack(seq_frames)

    def sample_pair(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Sample pair of sequences for comparison.

        Returns:
            seq_a: (seq_len, H, W, C) earlier sequence
            seq_b: (seq_len, H, W, C) later sequence
            label: 1.0 if b > a
            gap: temporal distance
        """
        r = random.random()

        # 80% combinatorial sampling
        if r < 0.80:
            return self._sample_combinatorial()

        # 20% hard negative
        if self.intro_frames or self.name_screens:
            return self._sample_hard_negative()

        return self._sample_combinatorial()

    def _sample_combinatorial(self, max_retries: int = 20):
        """TRUE combinatorial: pick ANY two sequences from video."""
        for _ in range(max_retries):
            video = random.choice(self.videos)
            n = len(video['frames'])

            # Need room for two sequences
            max_start = n - self.config.seq_len * self.config.frame_skip
            if max_start < 10:
                continue

            # Pick two random starting points
            start_a = random.randint(0, max_start)
            start_b = random.randint(0, max_start)

            if start_a == start_b:
                continue

            seq_a = self._sample_sequence(video, start_a)
            seq_b = self._sample_sequence(video, start_b)

            if seq_a is None or seq_b is None:
                continue

            gap = abs(start_b - start_a)

            if start_b > start_a:
                return seq_a, seq_b, 1.0, gap
            else:
                return seq_a, seq_b, 0.0, gap

        # Fallback
        return seq_a, seq_b, 1.0 if start_b > start_a else 0.0, gap

    def _sample_hard_negative(self, max_retries: int = 20):
        """Sample intro sequence vs late gameplay sequence."""
        for _ in range(max_retries):
            # Build intro sequence from random intro frames
            all_intro = self.intro_frames + self.name_screens
            if len(all_intro) < self.config.seq_len:
                return self._sample_combinatorial()

            intro_paths = random.sample(all_intro, self.config.seq_len)
            intro_frames = []
            valid = True
            for p in intro_paths:
                frame = np.load(p)
                if is_blank_frame(frame):
                    valid = False
                    break
                intro_frames.append(frame)

            if not valid:
                continue

            intro_seq = np.stack(intro_frames)

            # Get late gameplay sequence
            video = random.choice(self.videos)
            n = len(video['frames'])
            late_start = n // 2
            max_start = n - self.config.seq_len * self.config.frame_skip

            if max_start <= late_start:
                continue

            start = random.randint(late_start, max_start)
            gameplay_seq = self._sample_sequence(video, start)

            if gameplay_seq is None:
                continue

            # Gameplay should be higher than intro
            return intro_seq, gameplay_seq, 1.0, 100000

        return self._sample_combinatorial()

    def sample_batch(self, batch_size: int):
        """Sample a batch of sequence pairs."""
        seqs_a, seqs_b, labels, gaps = [], [], [], []

        for _ in range(batch_size):
            sa, sb, label, gap = self.sample_pair()
            seqs_a.append(sa)
            seqs_b.append(sb)
            labels.append(label)
            gaps.append(gap)

        return (
            np.stack(seqs_a),
            np.stack(seqs_b),
            np.array(labels, dtype=np.float32),
            np.array(gaps, dtype=np.float32),
        )


# ============================================================================
# TRAINING
# ============================================================================

def margin_loss(score_diff, labels, gaps, margin_scale=0.1):
    """Bradley-Terry loss with margin based on gap."""
    logits = score_diff * (2 * labels - 1)
    bce_loss = jax.nn.softplus(-logits)

    target_margin = margin_scale * jnp.log(gaps + 1)
    margin_violation = jnp.maximum(0, target_margin - jnp.abs(score_diff))

    return bce_loss + 2.0 * margin_violation


def create_train_state(config: SequenceRewardConfig, key, tokenizer_params=None, dynamics_params=None):
    """Create training state with pretrained weights."""
    model = SequenceRewardModel(config)

    dummy = jnp.zeros((1, config.seq_len, *config.frame_shape), dtype=jnp.uint8)
    params = model.init(key, dummy)

    # Load pretrained weights
    if tokenizer_params is not None:
        import flax
        params = flax.core.unfreeze(params)
        # Handle nested params structure
        tok_params = tokenizer_params.get('params', tokenizer_params)
        if 'params' in tok_params:
            tok_params = tok_params['params']
        params['params']['encoder'] = tok_params['encoder']
        params['params']['quantizer'] = tok_params['quantizer']
        params = flax.core.freeze(params)
        print("  Loaded pretrained encoder + quantizer")

    if dynamics_params is not None:
        # Note: Dynamics model uses different transformer structure
        # We'll train transformer head from scratch but benefit from encoder
        print("  (Dynamics model loaded but transformer structure differs - training head fresh)")

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {num_params:,}")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=0.01),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), model


def train(config: SequenceRewardConfig, data_dir: str, output_dir: str,
          tokenizer_path: str = None, dynamics_path: str = None):
    """Main training loop."""
    print(f"\n{'='*60}")
    print("Training Sequence Reward Model")
    print(f"{'='*60}")
    print(f"Sequence length: {config.seq_len} frames")
    print(f"Frame skip: {config.frame_skip}")
    print(f"Pairs per epoch: {config.pairs_per_epoch:,}")
    print(f"Hard negative ratio: {config.hard_negative_ratio}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained weights
    tokenizer_params = None
    dynamics_params = None

    if tokenizer_path:
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        tokenizer_params = tokenizer_data
        print(f"Loaded tokenizer: {tokenizer_path}")

    if dynamics_path:
        with open(dynamics_path, 'rb') as f:
            dynamics_data = pickle.load(f)
        dynamics_params = dynamics_data
        print(f"Loaded dynamics: {dynamics_path}")

    # Load data
    train_dataset = SequencePairDataset(data_dir, config, split='train')
    test_dataset = SequencePairDataset(data_dir, config, split='test')

    # Create model
    key = jax.random.PRNGKey(42)
    state, model = create_train_state(config, key, tokenizer_params, dynamics_params)

    # Training step
    @jax.jit
    def train_step(state, seqs_a, seqs_b, labels, gaps, key):
        def loss_fn(params):
            score_a = model.apply(params, seqs_a, train=True, rngs={'dropout': key})
            key2 = jax.random.fold_in(key, 1)
            score_b = model.apply(params, seqs_b, train=True, rngs={'dropout': key2})

            score_diff = score_b - score_a

            if config.use_margin:
                loss = margin_loss(score_diff, labels, gaps, config.margin_scale).mean()
            else:
                logits = score_diff
                loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

            preds = (score_diff > 0).astype(jnp.float32)
            accuracy = (preds == labels).mean()

            return loss, (accuracy, score_diff)

        (loss, (acc, diff)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, {'loss': loss, 'accuracy': acc, 'diff_mean': jnp.abs(diff).mean()}

    @jax.jit
    def eval_step(state, seqs_a, seqs_b, labels, gaps):
        score_a = model.apply(state.params, seqs_a, train=False)
        score_b = model.apply(state.params, seqs_b, train=False)

        score_diff = score_b - score_a
        loss = optax.sigmoid_binary_cross_entropy(score_diff, labels).mean()

        preds = (score_diff > 0).astype(jnp.float32)
        accuracy = (preds == labels).mean()

        return {'loss': loss, 'accuracy': accuracy}

    # Training loop
    steps_per_epoch = config.pairs_per_epoch // config.batch_size
    best_acc = 0

    for epoch in range(config.epochs):
        t0 = time.time()
        train_metrics = {'loss': [], 'accuracy': [], 'diff_mean': []}

        for step in range(steps_per_epoch):
            seqs_a, seqs_b, labels, gaps = train_dataset.sample_batch(config.batch_size)

            key, step_key = jax.random.split(key)
            state, metrics = train_step(
                state,
                jnp.array(seqs_a), jnp.array(seqs_b),
                jnp.array(labels), jnp.array(gaps),
                step_key
            )

            for k, v in metrics.items():
                train_metrics[k].append(float(v))

        # Evaluation
        test_metrics = {'loss': [], 'accuracy': []}
        for _ in range(50):
            seqs_a, seqs_b, labels, gaps = test_dataset.sample_batch(config.batch_size)
            metrics = eval_step(
                state,
                jnp.array(seqs_a), jnp.array(seqs_b),
                jnp.array(labels), jnp.array(gaps)
            )
            for k, v in metrics.items():
                test_metrics[k].append(float(v))

        train_loss = np.mean(train_metrics['loss'])
        train_acc = np.mean(train_metrics['accuracy'])
        test_loss = np.mean(test_metrics['loss'])
        test_acc = np.mean(test_metrics['accuracy'])
        diff_mean = np.mean(train_metrics['diff_mean'])

        epoch_time = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1%} | "
              f"Test: loss={test_loss:.4f} acc={test_acc:.1%} | "
              f"Δscore={diff_mean:.2f} | {epoch_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = output_path / 'reward_sequence_best.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'config': {
                        'seq_len': config.seq_len,
                        'frame_skip': config.frame_skip,
                        'vocab_size': config.vocab_size,
                        'embed_dim': config.embed_dim,
                        'encoder_channels': config.encoder_channels,
                        'num_layers': config.num_layers,
                        'num_heads': config.num_heads,
                    },
                    'test_accuracy': float(test_acc),
                }, f)
            print(f"  -> Saved best model (acc={test_acc:.1%})")

    print(f"\n{'='*60}")
    print(f"Training complete! Best accuracy: {best_acc:.1%}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/speedruns/frames")
    parser.add_argument("--output", type=str, default="reward_sequence")
    parser.add_argument("--tokenizer", type=str, default="pretrained_tokenizer.pkl")
    parser.add_argument("--dynamics", type=str, default="pretrained_dynamics.pkl")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pairs-per-epoch", type=int, default=100000)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.2)
    args = parser.parse_args()

    config = SequenceRewardConfig(
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        pairs_per_epoch=args.pairs_per_epoch,
        hard_negative_ratio=args.hard_negative_ratio,
    )

    train(config, args.data, args.output, args.tokenizer, args.dynamics)


if __name__ == "__main__":
    main()
