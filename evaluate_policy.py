#!/usr/bin/env python3
"""
Evaluate a trained policy by watching it play and measuring progress.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pickle
from pathlib import Path
from PIL import Image
import argparse


def load_policy(policy_path):
    """Load trained policy."""
    with open(policy_path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['config']


def load_reward_model(reward_path):
    """Load reward model for progress scoring."""
    from train_reward_cnn import load_model
    model, params, batch_stats = load_model(reward_path)

    @jax.jit
    def get_score(frame):
        return model.apply(
            {'params': params, 'batch_stats': batch_stats},
            frame, train=False
        )
    return get_score


def create_policy_fn(params, config):
    """Create policy inference function."""
    from train_with_cnn_reward import SmallVLMPolicy

    model = SmallVLMPolicy(
        embed_dim=config.get('embed_dim', 128),
        vision_layers=config.get('vision_layers', 3),
        decoder_layers=config.get('decoder_layers', 1),
        num_heads=config.get('num_heads', 4),
        num_actions=config.get('num_actions', 8),
        patch_size=config.get('patch_size', 16),
    )

    # Handle nested params structure
    p = params['params'] if 'params' in params else params

    @jax.jit
    def get_action(key, frame):
        logits, _ = model.apply({'params': p}, frame[None], train=False)
        action = jax.random.categorical(key, logits[0])
        probs = jax.nn.softmax(logits[0])
        return action, probs

    return get_action


def evaluate(policy_path, reward_path, rom_path, num_steps=1000, save_video=True):
    """Run evaluation."""
    from emurust import VecGameBoy

    print(f"Loading policy: {policy_path}")
    params, config = load_policy(policy_path)

    print(f"Loading reward model: {reward_path}")
    get_score = load_reward_model(reward_path)

    print("Creating policy function...")
    get_action = create_policy_fn(params, config)

    print(f"Loading ROM: {rom_path}")
    rom_data = np.fromfile(rom_path, dtype=np.uint8)

    # Single environment for evaluation
    env = VecGameBoy(rom_data, 1, frame_skip=30)
    obs = np.zeros((1, 144, 160, 3), dtype=np.uint8)
    env.reset(obs)

    key = jax.random.PRNGKey(42)

    # Track metrics
    progress_scores = []
    action_counts = np.zeros(8)
    frames = []

    print(f"\nRunning {num_steps} steps...")
    for step in range(num_steps):
        # Get action from policy
        key, action_key = jax.random.split(key)
        action, probs = get_action(action_key, jnp.array(obs[0]))
        action = int(action)
        action_counts[action] += 1

        # Step environment
        actions = np.array([action], dtype=np.uint8)
        env.step(actions, obs)

        # Compute progress score periodically
        if step % 64 == 0:
            score = float(get_score(jnp.array(obs))[0])
            progress_scores.append(score)

            if step % 256 == 0:
                print(f"  Step {step:5d}: progress={score:+.2f}, action={action}")

        # Save frames for video
        if save_video and step % 4 == 0:  # Every 4th frame
            frames.append(obs[0].copy())

    # Final stats
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nProgress scores:")
    print(f"  Start:  {progress_scores[0]:+.2f}")
    print(f"  End:    {progress_scores[-1]:+.2f}")
    print(f"  Min:    {min(progress_scores):+.2f}")
    print(f"  Max:    {max(progress_scores):+.2f}")
    print(f"  Delta:  {progress_scores[-1] - progress_scores[0]:+.2f}")

    print(f"\nAction distribution:")
    action_names = ['A', 'B', 'Select', 'Start', 'Right', 'Left', 'Up', 'Down']
    action_pcts = action_counts / action_counts.sum() * 100
    for i, (name, pct) in enumerate(zip(action_names, action_pcts)):
        bar = '█' * int(pct / 2)
        print(f"  {name:>6}: {pct:5.1f}% {bar}")

    # Check if near-uniform (random)
    entropy = -np.sum(action_pcts/100 * np.log(action_pcts/100 + 1e-10))
    max_entropy = np.log(8)
    print(f"\nAction entropy: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}% of max)")

    if entropy > 0.95 * max_entropy:
        print("  ⚠️  Policy is nearly random!")

    # Save video
    if save_video and frames:
        print(f"\nSaving video ({len(frames)} frames)...")
        output_dir = Path("eval_output")
        output_dir.mkdir(exist_ok=True)

        # Save as GIF
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            output_dir / "eval_rollout.gif",
            save_all=True,
            append_images=imgs[1:],
            duration=50,  # 20 fps
            loop=0
        )
        print(f"  Saved: {output_dir}/eval_rollout.gif")

        # Save key frames
        for i, idx in enumerate([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]):
            img = Image.fromarray(frames[idx])
            img = img.resize((160*3, 144*3), Image.NEAREST)
            img.save(output_dir / f"frame_{i}.png")
        print(f"  Saved key frames to {output_dir}/")

    # Compare to random baseline
    print("\n" + "="*60)
    print("RANDOM BASELINE COMPARISON")
    print("="*60)

    env.reset(obs)
    random_scores = []
    for step in range(num_steps):
        action = np.random.randint(0, 8)
        actions = np.array([action], dtype=np.uint8)
        env.step(actions, obs)

        if step % 64 == 0:
            score = float(get_score(jnp.array(obs))[0])
            random_scores.append(score)

    print(f"Random agent progress: {random_scores[0]:+.2f} → {random_scores[-1]:+.2f} (Δ={random_scores[-1]-random_scores[0]:+.2f})")
    print(f"Trained agent progress: {progress_scores[0]:+.2f} → {progress_scores[-1]:+.2f} (Δ={progress_scores[-1]-progress_scores[0]:+.2f})")

    trained_delta = progress_scores[-1] - progress_scores[0]
    random_delta = random_scores[-1] - random_scores[0]

    if trained_delta > random_delta + 0.5:
        print("✓ Trained agent outperforms random!")
    elif trained_delta > random_delta:
        print("~ Trained agent slightly better than random")
    else:
        print("✗ Trained agent not better than random")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/policy_final.pkl")
    parser.add_argument("--reward-model", default="reward_resnet34.pkl")
    parser.add_argument("--rom", default="roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    evaluate(
        args.policy,
        args.reward_model,
        args.rom,
        num_steps=args.steps,
        save_video=not args.no_video
    )


if __name__ == "__main__":
    main()
