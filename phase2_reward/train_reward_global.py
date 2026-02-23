#!/usr/bin/env python3
"""
Train a CNN reward model with global progress understanding.

Key improvements over train_reward_cnn.py:
1. Variable temporal gaps (10 frames to 100K+ frames)
2. Cross-video comparisons at similar progress percentages
3. Margin-scaled loss based on temporal distance
4. Hard negative mining (intro vs gameplay)

This creates a reward model that understands GLOBAL progress ranking,
not just local frame ordering.

Usage:
    python train_reward_global.py --data data/speedruns/frames --epochs 100
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
from typing import List, Tuple, Optional
from functools import partial
import random
import time
import pickle


@dataclass
class Config:
    # Model
    model_name: str = '34'  # ResNet variant

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 100
    pairs_per_epoch: int = 20000

    # Data
    frame_shape: Tuple[int, int, int] = (144, 160, 3)

    # Gap sampling distribution (probabilities for each category)
    gap_probs: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.25, 0.15])
    # Categories: short (10-100), medium (100-1K), long (1K-10K), very_long (10K+)

    # Margin scaling
    use_margin_loss: bool = True
    margin_scale: float = 0.1  # score_diff should be >= margin_scale * log(gap)

    # Hard negative mining
    hard_negative_ratio: float = 0.2  # Fraction of pairs that are intro vs gameplay


# Model configs (same as train_reward_cnn.py)
MODEL_CONFIGS = {
    'tiny': {'blocks_per_stage': (1, 1, 1, 1), 'channels': (32, 64, 128, 256)},
    'small': {'blocks_per_stage': (1, 1, 1, 1), 'channels': (64, 128, 256, 512)},
    'medium': {'blocks_per_stage': (2, 2, 2, 2), 'channels': (32, 64, 128, 256)},
    '18': {'blocks_per_stage': (2, 2, 2, 2), 'channels': (64, 128, 256, 512)},
    '34': {'blocks_per_stage': (3, 4, 6, 3), 'channels': (64, 128, 256, 512)},
}


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
    """ResNet for reward prediction."""
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

        for stage_idx, (num_blocks, channels) in enumerate(zip(self.blocks_per_stage, self.channels)):
            for block_idx in range(num_blocks):
                strides = 2 if block_idx == 0 and stage_idx > 0 else 1
                x = ResidualBlock(channels, strides)(x, train)

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(self.num_classes)(x)

        return x.squeeze(-1)


def is_blank_frame(frame: np.ndarray, threshold: float = 20.0) -> bool:
    """Check if frame is blank (all black, all white, or very low variance)."""
    if frame.max() - frame.min() < threshold:
        return True
    if frame.var() < 100:
        return True
    return False


class GlobalPairDataset:
    """
    Dataset that samples pairs with variable temporal gaps for global progress understanding.
    """

    def __init__(self, data_dir: str, config: Config, split: str = 'train', test_ratio: float = 0.1):
        self.config = config
        self.data_dir = Path(data_dir)

        # Load all videos with frame metadata
        self.videos = []
        self._load_videos(split, test_ratio)

        # Precompute video statistics for stratified sampling
        self._compute_statistics()

    def _load_videos(self, split: str, test_ratio: float):
        """Load video frame paths from .npy files (parses filenames for frame index)."""
        all_videos = []

        # Find all video directories
        for video_dir in sorted(self.data_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            # Collect .npy frame files
            frame_files = sorted([f for f in video_dir.iterdir() if f.suffix == '.npy'])

            if len(frame_files) < 100:
                continue

            # Parse filenames: frame_000123_t45.67.npy -> frame_idx=123
            frames = []
            for f in frame_files:
                # Extract frame index from filename
                name = f.stem  # e.g., "frame_000123_t45.67"
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        frame_idx = int(parts[1])
                        frames.append({
                            'path': str(f),
                            'frame_idx': frame_idx,
                            'video_id': video_dir.name,
                        })
                    except ValueError:
                        continue

            if len(frames) > 100:
                all_videos.append({
                    'frames': sorted(frames, key=lambda x: x['frame_idx']),
                    'total_frames': len(frames),
                    'video_id': video_dir.name,
                })

        # Split by video
        n_test = max(1, int(len(all_videos) * test_ratio))
        if split == 'train':
            self.videos = all_videos[n_test:]
        else:
            self.videos = all_videos[:n_test]

        total_frames = sum(v['total_frames'] for v in self.videos)
        print(f"[{split}] Loaded {len(self.videos)} videos with {total_frames:,} frames")

    def _compute_statistics(self):
        """Compute statistics for stratified sampling."""
        # Group frames by progress percentile for hard negative mining
        self.early_frames = []  # First 10% of each video (intro)
        self.late_frames = []   # Last 70% of each video (gameplay)

        for video in self.videos:
            n = video['total_frames']
            early_cutoff = int(n * 0.1)
            late_start = int(n * 0.3)

            self.early_frames.extend(video['frames'][:early_cutoff])
            self.late_frames.extend(video['frames'][late_start:])

        print(f"  Early frames (intro): {len(self.early_frames):,}")
        print(f"  Late frames (gameplay): {len(self.late_frames):,}")

        # Load actual intro frames (title screen, copyright, Oak intro, etc.)
        # These should ALWAYS rank below any speedrun frame
        self.title_intro_frames = []

        # Base data directory (data/)
        base_data_dir = self.data_dir.parent.parent  # data/speedruns/frames -> data/

        # Original intro frames
        intro_dir = base_data_dir / 'intro_frames'
        if intro_dir.exists():
            for f in sorted(intro_dir.iterdir()):
                if f.suffix == '.npy':
                    self.title_intro_frames.append({'path': str(f)})

        # Expanded intro frames (from video starts)
        expanded_dir = base_data_dir / 'intro_frames_expanded'
        if expanded_dir.exists():
            for f in sorted(expanded_dir.iterdir()):
                if f.suffix == '.npy':
                    self.title_intro_frames.append({'path': str(f)})

        # Comprehensive intro frames (from emulator run)
        comprehensive_dir = base_data_dir / 'intro_frames_comprehensive'
        if comprehensive_dir.exists():
            for f in sorted(comprehensive_dir.iterdir()):
                if f.suffix == '.npy':
                    self.title_intro_frames.append({'path': str(f)})

        if self.title_intro_frames:
            print(f"  Title/intro frames (hard negative): {len(self.title_intro_frames):,}")

        # Load name entry screens (keyboard, preset names, etc.)
        # These should ALSO rank below any actual gameplay frame
        self.name_screen_frames = []
        name_dir = base_data_dir / 'name_screens'
        if name_dir.exists():
            for f in sorted(name_dir.iterdir()):
                if f.suffix == '.npy':
                    self.name_screen_frames.append({'path': str(f)})
            print(f"  Name entry screens (hard negative): {len(self.name_screen_frames):,}")

    def _sample_gap_category(self) -> Tuple[int, int]:
        """Sample a gap category and return (min_gap, max_gap)."""
        categories = [
            (10, 100),       # Short: 0.17s - 1.7s
            (100, 1000),     # Medium: 1.7s - 17s
            (1000, 10000),   # Long: 17s - 2.8min
            (10000, 100000), # Very long: 2.8min - 28min
        ]

        idx = np.random.choice(len(categories), p=self.config.gap_probs)
        return categories[idx]

    def sample_pair(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Sample a pair of frames.

        - 80% combinatorial: ANY two frames from video (learns global ordering)
        - 20% hard negative: intro screens vs late gameplay (ensures intro < gameplay)

        Returns (frame_a, frame_b, label, gap) where:
        - label = 1.0 if b > a in progress
        - gap = temporal distance in frames
        """
        r = random.random()

        # 80% combinatorial sampling
        if r < 0.80:
            return self._sample_combinatorial()

        # 20% hard negative: intro/name screens vs late gameplay
        if self.title_intro_frames or self.name_screen_frames:
            return self._sample_intro_hard_negative()

        return self._sample_combinatorial()

    def _sample_intro_hard_negative(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Sample intro/name screen vs late gameplay frame.

        Intro screens (Game Freak logo, title, name entry) should ALWAYS
        score lower than actual gameplay. This handles out-of-distribution
        screens that aren't in the speedrun training data.
        """
        for _ in range(max_retries):
            # Pick intro or name screen
            all_intro = self.title_intro_frames + self.name_screen_frames
            if not all_intro:
                return self._sample_combinatorial()

            intro_info = random.choice(all_intro)
            intro_frame = np.load(intro_info['path'])

            if is_blank_frame(intro_frame):
                continue

            # Pick late gameplay frame (>50% through video)
            video = random.choice(self.videos)
            frames = video['frames']
            late_start = len(frames) // 2
            late_idx = random.randint(late_start, len(frames) - 1)
            late_frame = np.load(frames[late_idx]['path'])

            if is_blank_frame(late_frame):
                continue

            # Late gameplay should ALWAYS be higher than intro
            # Large gap to enforce strong margin
            gap = 100000

            return intro_frame, late_frame, 1.0, gap  # late > intro

        return self._sample_combinatorial()

    def _sample_combinatorial(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        TRUE combinatorial sampling: randomly pick ANY two frames from a video.

        This ensures the model sees comparisons across the ENTIRE game:
        - Frame 0 vs Frame 12000 (intro vs endgame)
        - Frame 100 vs Frame 5000 (early vs mid)
        - Frame 3000 vs Frame 8000 (mid vs late)
        - etc.

        The label is simply: 1.0 if idx_b > idx_a, else 0.0
        """
        for _ in range(max_retries):
            # Pick a random video
            video = random.choice(self.videos)
            frames = video['frames']
            n = len(frames)

            # Pick ANY two different indices
            idx_a, idx_b = random.sample(range(n), 2)

            # Load frames
            frame_a = np.load(frames[idx_a]['path'])
            frame_b = np.load(frames[idx_b]['path'])

            # Skip blank frames
            if is_blank_frame(frame_a) or is_blank_frame(frame_b):
                continue

            # Gap is absolute difference
            gap = abs(idx_b - idx_a)

            # Label: 1.0 if b comes after a in the video
            if idx_b > idx_a:
                return frame_a, frame_b, 1.0, gap
            else:
                return frame_a, frame_b, 0.0, gap

        # Fallback
        return frame_a, frame_b, 1.0 if idx_b > idx_a else 0.0, abs(idx_b - idx_a)

    def _sample_variable_gap(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Sample pair with variable temporal gap from same video, skipping blank frames."""
        for _ in range(max_retries):
            video = random.choice(self.videos)
            frames = video['frames']
            n = len(frames)

            # Get gap range for this sample
            min_gap, max_gap = self._sample_gap_category()
            max_gap = min(max_gap, n - 1)  # Can't exceed video length

            if max_gap <= min_gap:
                min_gap = 10
                max_gap = min(100, n - 1)

            # Sample indices
            idx_a = random.randint(0, n - min_gap - 1)
            max_idx_b = min(idx_a + max_gap, n - 1)
            min_idx_b = idx_a + min_gap

            if min_idx_b >= max_idx_b:
                idx_b = max_idx_b
            else:
                idx_b = random.randint(min_idx_b, max_idx_b)

            gap = idx_b - idx_a

            frame_a = np.load(frames[idx_a]['path'])
            frame_b = np.load(frames[idx_b]['path'])

            # Skip blank frames
            if is_blank_frame(frame_a) or is_blank_frame(frame_b):
                continue

            # Randomly swap for balanced labels
            if random.random() < 0.5:
                return frame_a, frame_b, 1.0, gap
            else:
                return frame_b, frame_a, 0.0, gap

        # Fallback: return whatever we got (shouldn't happen often)
        return frame_a, frame_b, 1.0, gap

    def _sample_title_intro_negative(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Sample pair: title/intro screen vs LATE speedrun frame (actual gameplay).

        This is the hardest negative: title screens, copyright notices, and
        Game Freak logo should ALWAYS rank below any actual gameplay frame.

        We specifically use LATE frames (>50% into video) to ensure we're
        comparing against actual gameplay, not early game menus.
        """
        for _ in range(max_retries):
            # Get a random title intro frame
            intro_info = random.choice(self.title_intro_frames)
            intro_frame = np.load(intro_info['path'])

            # Skip if blank (shouldn't happen with curated intro frames)
            if is_blank_frame(intro_frame):
                continue

            # Get a LATE frame from speedrun data (guaranteed gameplay)
            video = random.choice(self.videos)
            frames = video['frames']

            # Sample from latter 50% of video ONLY - this is actual gameplay
            start_idx = int(len(frames) * 0.5)
            idx = random.randint(start_idx, len(frames) - 1)

            speedrun_frame = np.load(frames[idx]['path'])

            # Skip blank frames
            if is_blank_frame(speedrun_frame):
                continue

            # Use extremely large gap (intro is conceptually infinitely behind)
            # This creates a very large margin requirement
            gap = 10000000  # 10 million frames conceptually

            # Speedrun frame should ALWAYS rank higher than intro frame
            # We always put intro as frame_a (lower) for consistency
            return intro_frame, speedrun_frame, 1.0, gap  # speedrun > intro

        # Fallback
        return intro_frame, speedrun_frame, 1.0, 10000000

    def _sample_name_screen_negative(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Sample pair: name entry screen vs LATE speedrun frame (actual gameplay).

        Name entry screens (keyboard, preset names, Oak's "what is your name?" dialog)
        should ALWAYS rank below any actual gameplay frame. The agent was getting
        stuck on these screens because the reward model didn't know they weren't progress.
        """
        for _ in range(max_retries):
            # Get a random name screen frame
            name_info = random.choice(self.name_screen_frames)
            name_frame = np.load(name_info['path'])

            # Skip if blank
            if is_blank_frame(name_frame):
                continue

            # Get a LATE frame from speedrun data (guaranteed gameplay)
            video = random.choice(self.videos)
            frames = video['frames']

            # Sample from latter 50% of video ONLY - this is actual gameplay
            start_idx = int(len(frames) * 0.5)
            idx = random.randint(start_idx, len(frames) - 1)

            speedrun_frame = np.load(frames[idx]['path'])

            # Skip blank frames
            if is_blank_frame(speedrun_frame):
                continue

            # Use large gap (name screens are conceptually behind gameplay)
            gap = 10000000  # 10 million frames conceptually

            # Speedrun frame should ALWAYS rank higher than name screen
            return name_frame, speedrun_frame, 1.0, gap  # speedrun > name screen

        # Fallback
        return name_frame, speedrun_frame, 1.0, 10000000

    def _sample_hard_negative(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Sample pair with early frame vs late frame (intro vs gameplay), skipping blanks."""
        for _ in range(max_retries):
            early = random.choice(self.early_frames)
            late = random.choice(self.late_frames)

            frame_early = np.load(early['path'])
            frame_late = np.load(late['path'])

            # Skip blank frames
            if is_blank_frame(frame_early) or is_blank_frame(frame_late):
                continue

            # Estimate gap (could be from different videos, so use a large value)
            gap = 50000  # Assume large gap for cross-video comparisons

            # Late should always rank higher than early
            if random.random() < 0.5:
                return frame_early, frame_late, 1.0, gap  # late > early
            else:
                return frame_late, frame_early, 0.0, gap  # early < late

        # Fallback
        return frame_early, frame_late, 1.0, 50000

    def _sample_cross_video(self, max_retries: int = 10) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Sample pair from different videos at different progress percentages."""
        for _ in range(max_retries):
            # Pick two different videos
            if len(self.videos) < 2:
                return self._sample_variable_gap()

            vid_a, vid_b = random.sample(self.videos, 2)

            # Sample at different progress percentages
            pct_a = random.random()
            pct_b = random.random()

            idx_a = int(pct_a * (len(vid_a['frames']) - 1))
            idx_b = int(pct_b * (len(vid_b['frames']) - 1))

            frame_a = np.load(vid_a['frames'][idx_a]['path'])
            frame_b = np.load(vid_b['frames'][idx_b]['path'])

            if is_blank_frame(frame_a) or is_blank_frame(frame_b):
                continue

            # Label based on progress percentage (further = better)
            label = 1.0 if pct_b > pct_a else 0.0
            gap = int(abs(pct_b - pct_a) * 10000)  # Scale to frame-like units

            return frame_a, frame_b, label, max(gap, 100)

        return self._sample_variable_gap()

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of pairs."""
        frames_a, frames_b, labels, gaps = [], [], [], []

        for _ in range(batch_size):
            fa, fb, label, gap = self.sample_pair()
            frames_a.append(fa)
            frames_b.append(fb)
            labels.append(label)
            gaps.append(gap)

        return (
            np.stack(frames_a),
            np.stack(frames_b),
            np.array(labels, dtype=np.float32),
            np.array(gaps, dtype=np.float32),
        )


def create_train_state(model, config: Config, key):
    """Create training state."""
    dummy_input = jnp.zeros((1, *config.frame_shape), dtype=jnp.uint8)
    variables = model.init(key, dummy_input, train=True)

    params = variables['params']
    batch_stats = variables['batch_stats']

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")

    tx = optax.adamw(config.learning_rate, weight_decay=0.01)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), batch_stats


def margin_loss(score_diff: jnp.ndarray, labels: jnp.ndarray, gaps: jnp.ndarray,
                margin_scale: float = 0.3) -> jnp.ndarray:
    """
    Bradley-Terry loss with margin based on temporal gap.

    The score difference should be proportional to log(gap):
    - Small gaps (10 frames): small margin (~0.7)
    - Large gaps (10K frames): large margin (~2.8)
    - Intro vs gameplay (10M frames): very large margin (~4.8)
    """
    # Standard Bradley-Terry loss
    logits = score_diff * (2 * labels - 1)  # Flip sign based on label
    bce_loss = jax.nn.softplus(-logits)

    # Margin penalty: encourage score_diff >= margin_scale * log(gap)
    target_margin = margin_scale * jnp.log(gaps + 1)
    margin_violation = jnp.maximum(0, target_margin - jnp.abs(score_diff))

    # Very strong penalty for margin violations
    return bce_loss + 2.0 * margin_violation


@partial(jax.jit, static_argnums=(6,))
def train_step(state, batch_stats, frames_a, frames_b, labels, gaps, use_margin: bool = True):
    """Single training step."""

    def loss_fn(params):
        # Forward pass for both frames
        score_a, updates_a = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            frames_a, train=True, mutable=['batch_stats']
        )
        score_b, updates_b = state.apply_fn(
            {'params': params, 'batch_stats': updates_a['batch_stats']},
            frames_b, train=True, mutable=['batch_stats']
        )

        score_diff = score_b - score_a

        if use_margin:
            loss = margin_loss(score_diff, labels, gaps).mean()
        else:
            # Standard Bradley-Terry
            logits = score_diff
            loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        # Accuracy
        preds = (score_diff > 0).astype(jnp.float32)
        accuracy = (preds == labels).mean()

        return loss, (updates_b['batch_stats'], accuracy, score_diff)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_batch_stats, accuracy, score_diff)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'score_diff_mean': jnp.abs(score_diff).mean(),
        'score_diff_std': score_diff.std(),
    }

    return state, new_batch_stats, metrics


@jax.jit
def eval_step(state, batch_stats, frames_a, frames_b, labels, gaps):
    """Evaluation step."""
    score_a = state.apply_fn(
        {'params': state.params, 'batch_stats': batch_stats},
        frames_a, train=False
    )
    score_b = state.apply_fn(
        {'params': state.params, 'batch_stats': batch_stats},
        frames_b, train=False
    )

    score_diff = score_b - score_a
    logits = score_diff
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    preds = (score_diff > 0).astype(jnp.float32)
    accuracy = (preds == labels).mean()

    return {
        'loss': loss,
        'accuracy': accuracy,
        'score_diff_mean': jnp.abs(score_diff).mean(),
    }


def train(config: Config, data_dir: str, output_dir: str):
    """Main training loop."""
    print(f"\n{'='*60}")
    print("Training Global Reward Model")
    print(f"{'='*60}")
    print(f"Model: ResNet-{config.model_name}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gap distribution: {config.gap_probs}")
    print(f"Hard negative ratio: {config.hard_negative_ratio}")
    print(f"Margin loss: {config.use_margin_loss}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    train_dataset = GlobalPairDataset(data_dir, config, split='train')
    test_dataset = GlobalPairDataset(data_dir, config, split='test')

    # Create model
    model_cfg = MODEL_CONFIGS[config.model_name]
    model = ResNet(
        blocks_per_stage=model_cfg['blocks_per_stage'],
        channels=model_cfg['channels'],
    )

    # Initialize
    key = jax.random.PRNGKey(42)
    state, batch_stats = create_train_state(model, config, key)

    # Training loop
    best_test_acc = 0
    steps_per_epoch = config.pairs_per_epoch // config.batch_size

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Training
        train_metrics = {'loss': [], 'accuracy': [], 'score_diff_mean': []}

        for step in range(steps_per_epoch):
            frames_a, frames_b, labels, gaps = train_dataset.sample_batch(config.batch_size)

            state, batch_stats, metrics = train_step(
                state, batch_stats,
                jnp.array(frames_a), jnp.array(frames_b),
                jnp.array(labels), jnp.array(gaps),
                config.use_margin_loss
            )

            for k, v in metrics.items():
                if k in train_metrics:
                    train_metrics[k].append(float(v))

        # Evaluation
        test_metrics = {'loss': [], 'accuracy': [], 'score_diff_mean': []}
        for _ in range(50):
            frames_a, frames_b, labels, gaps = test_dataset.sample_batch(config.batch_size)
            metrics = eval_step(
                state, batch_stats,
                jnp.array(frames_a), jnp.array(frames_b),
                jnp.array(labels), jnp.array(gaps)
            )
            for k, v in metrics.items():
                if k in test_metrics:
                    test_metrics[k].append(float(v))

        # Aggregate metrics
        train_loss = np.mean(train_metrics['loss'])
        train_acc = np.mean(train_metrics['accuracy'])
        test_loss = np.mean(test_metrics['loss'])
        test_acc = np.mean(test_metrics['accuracy'])
        score_diff = np.mean(train_metrics['score_diff_mean'])

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1%} | "
              f"Test: loss={test_loss:.4f} acc={test_acc:.1%} | "
              f"Δscore={score_diff:.2f} | "
              f"{epoch_time:.1f}s")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = output_path / 'reward_global_best.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'batch_stats': batch_stats,
                    'config': {
                        'model_name': config.model_name,
                        'architecture': f'ResNet-{config.model_name}',
                        'frame_shape': config.frame_shape,
                        'gap_probs': config.gap_probs,
                        'hard_negative_ratio': config.hard_negative_ratio,
                    },
                    'test_accuracy': float(test_acc),
                }, f)
            print(f"  → Saved best model (acc={test_acc:.1%})")

    # Save final model
    final_path = output_path / 'reward_global_final.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump({
            'params': state.params,
            'batch_stats': batch_stats,
            'config': {
                'model_name': config.model_name,
                'architecture': f'ResNet-{config.model_name}',
                'frame_shape': config.frame_shape,
            },
            'test_accuracy': float(test_acc),
        }, f)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_test_acc:.1%}")
    print(f"Models saved to: {output_path}")
    print(f"{'='*60}")

    return state, batch_stats


def main():
    parser = argparse.ArgumentParser(description="Train global reward model")
    parser.add_argument("--data", type=str, required=True, help="Path to speedrun frames")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--model", type=str, default="34", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pairs-per-epoch", type=int, default=20000)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.2)
    parser.add_argument("--no-margin", action="store_true", help="Disable margin loss")

    args = parser.parse_args()

    config = Config(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pairs_per_epoch=args.pairs_per_epoch,
        hard_negative_ratio=args.hard_negative_ratio,
        use_margin_loss=not args.no_margin,
    )

    train(config, args.data, args.output)


if __name__ == "__main__":
    main()
