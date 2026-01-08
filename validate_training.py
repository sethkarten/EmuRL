#!/usr/bin/env python3
"""
Validate the training framework before long runs.

Tests:
1. Reward model produces sensible progress scores
2. Environment runs correctly with frame skip
3. Short training doesn't crash and shows learning signal
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import json

def test_reward_model():
    """Test that reward model gives higher scores to later frames."""
    print("\n" + "="*60)
    print("TEST 1: Reward Model Sanity Check")
    print("="*60)

    from train_reward_cnn import load_model
    import jax

    model, params, batch_stats = load_model("reward_resnet34.pkl")

    @jax.jit
    def get_score(frame):
        return model.apply(
            {'params': params, 'batch_stats': batch_stats},
            frame[None], train=False
        )[0]

    # Load frames from speedrun data
    data_dir = Path("data/speedruns/frames")
    index_files = sorted(data_dir.glob("*_index.json"))

    if not index_files:
        print("  WARNING: No speedrun data found, skipping")
        return False

    # Test on first video
    with open(index_files[0]) as f:
        index = json.load(f)

    frames = index['frames']
    n = len(frames)

    # Sample frames at 0%, 25%, 50%, 75%, 100% through video
    test_indices = [0, n//4, n//2, 3*n//4, n-1]
    scores = []

    print(f"  Testing on {index_files[0].stem} ({n} frames)")
    print(f"  {'Position':<12} {'Frame':<8} {'Score':<10}")
    print(f"  {'-'*30}")

    for idx in test_indices:
        frame_path = data_dir / frames[idx]['path']
        frame = np.load(frame_path)
        score = float(get_score(jnp.array(frame)))
        scores.append(score)
        pct = idx * 100 // n
        print(f"  {pct:>3}%         {idx:<8} {score:>+.3f}")

    # Check if scores generally increase
    increases = sum(1 for i in range(len(scores)-1) if scores[i+1] > scores[i])
    print(f"\n  Score increases: {increases}/{len(scores)-1}")

    if increases >= 3:
        print("  ✓ PASS: Reward model shows progress over time")
        return True
    else:
        print("  ✗ FAIL: Reward model doesn't show clear progress")
        return False


def test_environment():
    """Test that environment runs correctly."""
    print("\n" + "="*60)
    print("TEST 2: Environment Sanity Check")
    print("="*60)

    from emurust import VecGameBoy

    rom_path = Path("roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb")
    if not rom_path.exists():
        print(f"  WARNING: ROM not found at {rom_path}, skipping")
        return False

    rom_data = np.fromfile(rom_path, dtype=np.uint8)

    # Test with frame_skip=30 (our training setting)
    num_envs = 4
    frame_skip = 30
    env = VecGameBoy(rom_data, num_envs, frame_skip=frame_skip)

    obs = np.zeros((num_envs, 144, 160, 3), dtype=np.uint8)
    env.reset(obs)

    print(f"  Created {num_envs} environments with frame_skip={frame_skip}")
    print(f"  Observation shape: {obs.shape}")

    # Run 100 steps with random actions
    for step in range(100):
        actions = np.random.randint(0, 8, size=num_envs, dtype=np.uint8)
        rewards, dones = env.step(actions, obs)

    # Check observations are not all zeros
    nonzero = (obs != 0).sum()
    total = obs.size
    pct = nonzero * 100 / total

    print(f"  After 100 steps: {pct:.1f}% non-zero pixels")
    print(f"  Frames rendered: {100 * frame_skip * num_envs:,}")

    if pct > 10:
        print("  ✓ PASS: Environment produces valid frames")
        return True
    else:
        print("  ✗ FAIL: Frames are mostly empty")
        return False


def test_save_load_state():
    """Test save/load state functionality."""
    print("\n" + "="*60)
    print("TEST 3: Save/Load State Check")
    print("="*60)

    from emurust import VecGameBoy

    rom_path = Path("roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb")
    if not rom_path.exists():
        print(f"  WARNING: ROM not found at {rom_path}, skipping")
        return False

    rom_data = np.fromfile(rom_path, dtype=np.uint8)
    env = VecGameBoy(rom_data, 2, frame_skip=30)

    obs = np.zeros((2, 144, 160, 3), dtype=np.uint8)
    env.reset(obs)

    # Run some steps to get past intro
    for _ in range(100):
        actions = np.array([0, 0], dtype=np.uint8)  # A button to advance
        env.step(actions, obs)

    # Save state of env 0
    state_id = env.save_state(0)
    saved_obs = obs[0].copy()

    # Run more steps with movement to change state
    for _ in range(100):
        actions = np.array([4, 5], dtype=np.uint8)  # Different directions
        env.step(actions, obs)

    changed_obs = obs[0].copy()

    # Load state back
    env.load_state(0, state_id)
    env.render(obs)
    restored_obs = obs[0].copy()

    # Check restoration
    diff_before = np.abs(saved_obs.astype(float) - changed_obs.astype(float)).mean()
    diff_after = np.abs(saved_obs.astype(float) - restored_obs.astype(float)).mean()

    print(f"  Saved state ID: {state_id}")
    print(f"  Diff before restore: {diff_before:.2f}")
    print(f"  Diff after restore: {diff_after:.2f}")

    if diff_after < 1.0 and diff_before > 5.0:
        print("  ✓ PASS: Save/load state works correctly")
        return True
    else:
        print("  ✗ FAIL: State restoration incorrect")
        return False


def test_short_training():
    """Run a very short training to check for crashes."""
    print("\n" + "="*60)
    print("TEST 4: Short Training Run")
    print("="*60)

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "train_with_cnn_reward.py",
         "--reward-model", "reward_resnet34.pkl",
         "--iterations", "20",
         "--num-envs", "16",
         "--no-wandb"],
        capture_output=True,
        text=True,
        timeout=300
    )

    print("  STDOUT (last 20 lines):")
    for line in result.stdout.strip().split('\n')[-20:]:
        print(f"    {line}")

    if result.returncode != 0:
        print(f"\n  STDERR:\n{result.stderr}")
        print("  ✗ FAIL: Training crashed")
        return False

    # Check for good signs
    output = result.stdout
    has_fps = "fps" in output.lower()
    has_reward = "reward=" in output
    has_checkpoints = "ckpts=" in output
    no_nan = "nan" not in output.lower()

    print(f"\n  FPS logging: {'✓' if has_fps else '✗'}")
    print(f"  Reward logging: {'✓' if has_reward else '✗'}")
    print(f"  Checkpoint logging: {'✓' if has_checkpoints else '✗'}")
    print(f"  No NaN values: {'✓' if no_nan else '✗'}")

    if has_fps and has_reward and no_nan:
        print("  ✓ PASS: Short training completed successfully")
        return True
    else:
        print("  ✗ FAIL: Training output looks wrong")
        return False


def main():
    print("\n" + "="*60)
    print("  TRAINING FRAMEWORK VALIDATION")
    print("="*60)

    results = {}

    # Run training first before other tests consume GPU memory
    results['training'] = test_short_training()
    results['reward_model'] = test_reward_model()
    results['environment'] = test_environment()
    results['save_load'] = test_save_load_state()

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20} {status}")
        if not passed:
            all_pass = False

    print("="*60)
    if all_pass:
        print("  All tests passed! Ready for training.")
    else:
        print("  Some tests failed. Please investigate before long runs.")

    return all_pass


if __name__ == "__main__":
    main()
