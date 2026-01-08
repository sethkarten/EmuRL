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

    # Progress checkpointing (for curriculum learning)
    checkpoint_progress_threshold: float = 0.5  # Save state when progress exceeds this delta
    max_checkpoints: int = 100  # Maximum number of checkpoints to keep
    reset_to_checkpoint_prob: float = 0.1  # Probability of resetting env to a checkpoint
    episode_length: int = 2048  # Max steps before soft reset (0 = no limit)

    # Logging
    wandb_project: str = "pokemon-red-vlm"
    log_interval: int = 10
    checkpoint_interval: int = 1000


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
                frame = frames[env_idx].copy()  # Store a copy of the frame
                self._insert_checkpoint(score, state_id, frame)
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


# ============================================================================
# POLICY MODEL (same as before)
# ============================================================================

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


class TransformerBlock(nn.Module):
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
            x = TransformerBlock(self.embed_dim, self.num_heads)(x, train)
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

def create_train_state(config: Config, key, pretrained_encoder=None):
    model = SmallVLMPolicy(
        embed_dim=config.embed_dim,
        vision_layers=config.vision_layers,
        decoder_layers=config.decoder_layers,
        num_heads=config.num_heads,
        num_actions=config.num_actions,
        patch_size=config.patch_size,
    )
    dummy = jnp.zeros((1, 144, 160, 3), dtype=jnp.uint8)
    params = model.init(key, dummy)

    # Replace encoder with pretrained weights if provided
    if pretrained_encoder is not None:
        params = flax.core.unfreeze(params)
        if 'params' in params:
            params['params']['VisionEncoder_0'] = pretrained_encoder
        else:
            params['VisionEncoder_0'] = pretrained_encoder
        params = flax.core.freeze(params)
        print("  Pretrained encoder weights loaded!")

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(config.learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-model", type=str, required=True,
                        help="Path to trained CNN reward model")
    parser.add_argument("--pretrained-encoder", type=str, default=None,
                        help="Path to pretrained visual encoder weights")
    parser.add_argument("--rom", default="roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--return-filter", type=float, default=0.0,
                        help="Only train on top X%% of returns (0 = no filtering, 0.8 = top 80%%)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    config = Config(num_envs=args.num_envs)

    # Load reward model
    print(f"Loading reward model: {args.reward_model}")
    reward_model, reward_params, reward_batch_stats = load_reward_model(args.reward_model)

    # JIT compile reward function
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

    # Initialize wandb
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"cnn-reward-{config.num_envs}env-{int(time.time())}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=asdict(config),
                tags=["pokemon-red", "cnn-reward", "ppo"],
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

    # Initialize policy
    key = jax.random.PRNGKey(int(time.time()))

    # Load pretrained encoder if provided (before creating train state)
    pretrained_encoder = None
    if args.pretrained_encoder:
        print(f"Loading pretrained encoder: {args.pretrained_encoder}")
        with open(args.pretrained_encoder, 'rb') as f:
            pretrained = pickle.load(f)
        pretrained_encoder = pretrained['encoder_params']

    state = create_train_state(config, key, pretrained_encoder)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Policy parameters: {num_params:,}")

    # JIT compile policy functions
    @jax.jit
    def select_action(state, frames, key):
        logits, value = state.apply_fn(state.params, frames, train=False)
        # Clip logits for numerical stability (consistent with ppo_update)
        logits = jnp.clip(logits, -20.0, 20.0)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(len(action)), action]
        return action, log_prob, value

    @jax.jit
    def ppo_update(state, frames, actions, returns, old_log_probs):
        def loss_fn(params):
            logits, values = state.apply_fn(params, frames, train=True)

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

    env.reset(obs)

    print(f"\nTraining config:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Envs: {config.num_envs}")
    print(f"  Trajectory: {config.trajectory_length} steps")
    print(f"  Reward computed every {config.reward_interval} steps")
    print("=" * 70)

    start_time = time.time()
    total_frames = 0

    for iteration in range(args.iterations):
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

            env.step(actions_np, obs)

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

        # Get progress scores
        progress_scores = compute_rewards(jnp.array(reward_frames))
        progress_scores = np.array(progress_scores).reshape(len(reward_indices), config.num_envs)

        # Update checkpointer with final progress scores (last frame of trajectory)
        final_progress = progress_scores[-1]  # (num_envs,)
        final_frames = all_frames[-1]  # (num_envs, 144, 160, 3)
        new_checkpoints = checkpointer.update(env, final_progress, final_frames)

        # Convert to per-step rewards via interpolation
        all_rewards = np.zeros((config.num_envs, config.trajectory_length), dtype=np.float32)
        for env_idx in range(config.num_envs):
            scores = progress_scores[:, env_idx]
            full_scores = np.interp(
                range(config.trajectory_length),
                reward_indices,
                scores
            )
            # Reward = progress delta - time penalty
            all_rewards[env_idx, 1:] = full_scores[1:] - full_scores[:-1] - config.time_penalty

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
            state, metrics = ppo_update(
                state,
                frames_flat[idx],
                actions_flat[idx],
                returns_flat[idx],
                log_probs_flat[idx]
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

            prog_range = ckpt_stats['progress_range']
            print(f"[{iteration:5d}] reward={avg_reward:+.4f} | "
                  f"ent={avg_entropy:.3f} | "
                  f"ckpts={ckpt_stats['num_checkpoints']} prog=[{prog_range[0]:.1f},{prog_range[1]:.1f}] | "
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

        # Checkpoints
        if iteration % config.checkpoint_interval == 0 and iteration > 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/policy_cnn_{iteration}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'params': state.params, 'config': asdict(config)}, f)
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
    with open("checkpoints/policy_final.pkl", 'wb') as f:
        pickle.dump({'params': state.params, 'config': asdict(config)}, f)
    print(f"  → Saved checkpoints/policy_final.pkl")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
