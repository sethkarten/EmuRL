#!/usr/bin/env python3
"""
Train a CNN reward model on Pokemon Red speedrun frames.

The model learns to predict "progress" from a single frame.
Training uses pairwise comparisons (Bradley-Terry style):
- Sample two frames from same speedrun
- Later frame should have higher progress score

Usage:
    python train_reward_cnn.py --data data/speedruns/frames --epochs 50
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


@dataclass
class Config:
    # Model
    channels: List[int] = None  # CNN channel progression
    embed_dim: int = 128

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 50
    pairs_per_epoch: int = 10000

    # Data
    frame_shape: Tuple[int, int, int] = (144, 160, 3)
    min_frame_gap: int = 10  # Minimum frames apart for comparison

    def __post_init__(self):
        if self.channels is None:
            self.channels = [32, 64, 128, 256]


class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""
    channels: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # First conv
        y = nn.Conv(self.channels, (3, 3), strides=(self.strides, self.strides), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        # Second conv
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        # Shortcut connection
        if self.strides != 1 or x.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, (1, 1), strides=(self.strides, self.strides), use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """
    Configurable ResNet architecture for reward prediction.

    Supports different sizes:
    - 'tiny':   [1,1,1,1] blocks, 32-64-128-256 channels  (~0.3M params)
    - 'small':  [1,1,1,1] blocks, 64-128-256-512 channels (~2.8M params)
    - 'medium': [2,2,2,2] blocks, 32-64-128-256 channels  (~1.2M params)
    - '18':     [2,2,2,2] blocks, 64-128-256-512 channels (~11M params)
    - '34':     [3,4,6,3] blocks, 64-128-256-512 channels (~21M params)
    """
    blocks_per_stage: tuple = (2, 2, 2, 2)
    channels: tuple = (64, 128, 256, 512)
    num_classes: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Normalize input
        x = x.astype(jnp.float32) / 255.0

        # Initial convolution
        x = nn.Conv(self.channels[0], (7, 7), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        # Residual stages
        for stage_idx, (num_blocks, ch) in enumerate(zip(self.blocks_per_stage, self.channels)):
            for block_idx in range(num_blocks):
                # First block of each stage (except first) has stride 2
                strides = 2 if block_idx == 0 and stage_idx > 0 else 1
                x = ResidualBlock(ch, strides=strides)(x, train)

        # Global average pooling
        x = x.mean(axis=(1, 2))

        # Final FC layer
        x = nn.Dense(self.num_classes)(x)

        return x.squeeze(-1)


# Preset configurations
def ResNetTiny():
    """~0.3M params"""
    return ResNet(blocks_per_stage=(1,1,1,1), channels=(32,64,128,256))

def ResNetSmall():
    """~2.8M params"""
    return ResNet(blocks_per_stage=(1,1,1,1), channels=(64,128,256,512))

def ResNetMedium():
    """~1.2M params"""
    return ResNet(blocks_per_stage=(2,2,2,2), channels=(32,64,128,256))

def ResNet18():
    """~11M params"""
    return ResNet(blocks_per_stage=(2,2,2,2), channels=(64,128,256,512))

def ResNet34():
    """~21M params"""
    return ResNet(blocks_per_stage=(3,4,6,3), channels=(64,128,256,512))


MODEL_CONFIGS = {
    'tiny': ResNetTiny,
    'small': ResNetSmall,
    'medium': ResNetMedium,
    '18': ResNet18,
    '34': ResNet34,
}

# Alias for backward compatibility
RewardCNN = ResNet18


class SpeedrunDataset:
    """Dataset of speedrun frames with progress labels."""

    def __init__(self, data_dir: Path, config: Config, split: str = 'train', test_ratio: float = 0.2):
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split

        # Load all index files
        all_videos = []
        for index_path in sorted(self.data_dir.glob("*_index.json")):
            with open(index_path) as f:
                index = json.load(f)
                # Load frame paths and progress values
                frames = []
                for entry in index['frames']:
                    frame_path = self.data_dir / entry['path']
                    if frame_path.exists():
                        frames.append({
                            'path': frame_path,
                            'progress': entry['progress'],
                            'frame_idx': entry['frame_idx'],
                        })
                if len(frames) > config.min_frame_gap * 2:
                    all_videos.append(frames)

        # Split videos into train/test (by video, not by frame)
        n_test = max(1, int(len(all_videos) * test_ratio))
        if split == 'train':
            self.videos = all_videos[n_test:]  # Use later videos for training
        else:
            self.videos = all_videos[:n_test]  # Use first video(s) for testing

        total_frames = sum(len(v) for v in self.videos)
        print(f"[{split}] Loaded {len(self.videos)} videos with {total_frames} frames")

    def sample_pair(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Sample a pair of frames from the same video.
        Returns (frame_a, frame_b, label) where label = 1 if b > a in progress.
        """
        # Pick random video
        video = random.choice(self.videos)

        # Pick two frames with minimum gap
        n = len(video)
        idx_a = random.randint(0, n - self.config.min_frame_gap - 1)
        idx_b = random.randint(idx_a + self.config.min_frame_gap, n - 1)

        frame_a = np.load(video[idx_a]['path'])
        frame_b = np.load(video[idx_b]['path'])

        # Label: 1 if b is more progressed than a (always true by construction)
        # But we randomly swap to create balanced pairs
        if random.random() < 0.5:
            return frame_a, frame_b, 1.0  # b > a
        else:
            return frame_b, frame_a, 0.0  # a > b (swapped)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of pairs."""
        frames_a = []
        frames_b = []
        labels = []

        for _ in range(batch_size):
            fa, fb, label = self.sample_pair()
            frames_a.append(fa)
            frames_b.append(fb)
            labels.append(label)

        return (
            np.stack(frames_a),
            np.stack(frames_b),
            np.array(labels, dtype=np.float32)
        )


def create_train_state(config: Config, key):
    """Initialize model and optimizer."""
    model = ResNet18(num_classes=1)

    dummy = jnp.zeros((1, *config.frame_shape), dtype=jnp.uint8)
    variables = model.init(key, dummy)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config.learning_rate),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    ), variables.get('batch_stats', {})


@jax.jit
def train_step(state, batch_stats, frames_a, frames_b, labels):
    """
    Bradley-Terry style pairwise loss.

    P(b > a) = sigmoid(score_b - score_a)
    Loss = BCE(P(b > a), label)
    """
    def loss_fn(params):
        # Get scores for both frames
        score_a, new_stats_a = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            frames_a, train=True,
            mutable=['batch_stats']
        )
        score_b, new_stats_b = state.apply_fn(
            {'params': params, 'batch_stats': new_stats_a['batch_stats']},
            frames_b, train=True,
            mutable=['batch_stats']
        )

        # Bradley-Terry: P(b > a) = sigmoid(score_b - score_a)
        logits = score_b - score_a

        # Binary cross-entropy
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        # Accuracy
        preds = (logits > 0).astype(jnp.float32)
        accuracy = (preds == labels).mean()

        return loss, (new_stats_b['batch_stats'], accuracy, score_a.mean(), score_b.mean())

    grads, (new_batch_stats, accuracy, mean_a, mean_b) = jax.grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, new_batch_stats, accuracy, mean_a, mean_b


@jax.jit
def eval_step(state, batch_stats, frames_a, frames_b, labels):
    """Evaluate without updating batch stats."""
    score_a = state.apply_fn(
        {'params': state.params, 'batch_stats': batch_stats},
        frames_a, train=False,
    )
    score_b = state.apply_fn(
        {'params': state.params, 'batch_stats': batch_stats},
        frames_b, train=False,
    )

    logits = score_b - score_a
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    preds = (logits > 0).astype(jnp.float32)
    accuracy = (preds == labels).mean()

    return loss, accuracy


def create_model(model_name: str):
    """Create model by name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/speedruns/frames",
                        help="Directory containing frame data")
    parser.add_argument("--model", type=str, default="18", choices=list(MODEL_CONFIGS.keys()),
                        help="Model size: tiny, small, medium, 18, 34")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"reward_resnet{args.model}.pkl"

    config = Config(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
    )

    # Load train and test datasets
    print("Loading datasets...")
    train_dataset = SpeedrunDataset(args.data, config, split='train', test_ratio=0.2)
    test_dataset = SpeedrunDataset(args.data, config, split='test', test_ratio=0.2)

    if len(train_dataset.videos) == 0:
        print("No training data found!")
        return

    # Initialize model
    print(f"\nInitializing ResNet-{args.model}...")
    model = create_model(args.model)

    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, *config.frame_shape), dtype=jnp.uint8)
    variables = model.init(key, dummy)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config.learning_rate),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    batch_stats = variables.get('batch_stats', {})

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}")

    # Training loop
    print(f"\nTraining for {config.epochs} epochs...")
    print("=" * 70)
    print(f"{'Epoch':>5} | {'Train Acc':>9} | {'Test Acc':>9} | {'Time':>6} | {'Status'}")
    print("-" * 70)

    steps_per_epoch = config.pairs_per_epoch // config.batch_size
    test_steps = 1000 // config.batch_size  # Fewer steps for test eval
    best_test_acc = 0.0

    for epoch in range(config.epochs):
        t0 = time.time()
        train_acc = 0.0

        # Training
        for step in range(steps_per_epoch):
            frames_a, frames_b, labels = train_dataset.sample_batch(config.batch_size)
            state, batch_stats, accuracy, _, _ = train_step(
                state, batch_stats,
                jnp.array(frames_a),
                jnp.array(frames_b),
                jnp.array(labels)
            )
            train_acc += float(accuracy)

        train_acc /= steps_per_epoch

        # Test evaluation
        test_acc = 0.0
        for step in range(test_steps):
            frames_a, frames_b, labels = test_dataset.sample_batch(config.batch_size)
            _, acc = eval_step(
                state, batch_stats,
                jnp.array(frames_a),
                jnp.array(frames_b),
                jnp.array(labels)
            )
            test_acc += float(acc)
        test_acc /= test_steps

        epoch_time = time.time() - t0

        # Save best model based on test accuracy
        status = ""
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model(state, batch_stats, config, args.output, args.model)
            status = f"★ Best ({best_test_acc:.3f})"

        print(f"{epoch+1:5d} | {train_acc:9.3f} | {test_acc:9.3f} | {epoch_time:5.1f}s | {status}")

    print("=" * 70)
    print(f"Training complete! Best test accuracy: {best_test_acc:.3f}")
    print(f"Model saved to: {args.output}")


def save_model(state, batch_stats, config, path, model_name='18'):
    """Save model weights and config."""
    import pickle

    with open(path, 'wb') as f:
        pickle.dump({
            'params': state.params,
            'batch_stats': batch_stats,
            'config': {
                'architecture': f'ResNet-{model_name}',
                'model_name': model_name,
                'frame_shape': config.frame_shape,
            }
        }, f)


def load_model(path):
    """Load model for inference."""
    import pickle

    with open(path, 'rb') as f:
        data = pickle.load(f)

    model_name = data['config'].get('model_name', '18')
    model = create_model(model_name)

    return model, data['params'], data['batch_stats']


if __name__ == "__main__":
    main()
