# EmuRL - Pokemon Red RL Training

## Project Notes

- Always use `uv run python` (not conda)
- UV cache: `UV_CACHE_DIR=/media/milkkarten/data/cache/uv`
- Emulator: `from emurust import VecGameBoy`

---

## Training Pipeline Plan

### Phase 1: Dreamer 4-Style Visual Pretraining

Train on speedrun videos (no actions needed).

#### Stage 1A: Tokenizer (Autoencoder)
```python
# Architecture: CNN encoder/decoder
z = encoder(frame)           # (144, 160, 3) -> (latent_dim,)
recon = decoder(z)           # (latent_dim,) -> (144, 160, 3)

# Loss: Reconstruction
loss = MSE(recon, frame) + lambda * LPIPS(recon, frame)
```

#### Stage 1B: Dynamics Model (Next Latent Prediction)
```python
# Architecture: Transformer or MLP
z_t = encoder(frame_t).detach()      # Freeze encoder
z_t1_pred = dynamics(z_t)            # Predict next latent
z_t1_true = encoder(frame_t+1).detach()

# Loss: Predict next latent
loss = MSE(z_t1_pred, z_t1_true)
```

**Data**: `data/speedruns/frames/` (64K frames from 5 speedrun videos)

---

### Phase 2: CNN Reward Model (DONE)

ResNet-34 trained on speedrun frame ordering.
- Input: Single frame (144, 160, 3)
- Output: Progress score (scalar)
- Accuracy: 88.7% on pairwise comparisons
- File: `reward_resnet34.pkl`

---

### Phase 3: RL Training (PPO)

#### Architecture
```
frame -> pretrained_encoder -> z -> policy_head -> action logits
                                 -> value_head  -> value estimate
```

#### Key Settings
```python
frame_skip = 30              # 32s game time per reward computation
reward_interval = 64         # Steps between reward computation
time_penalty = 0.001         # Cost per step (encourages speed)
return_filter = 0.8          # Train only on top 80% of trajectories
gamma = 0.99                 # Discount factor
```

#### Reward Function
```python
reward = progress_delta - time_penalty
# where progress_delta = reward_model(frame_t+1) - reward_model(frame_t)
```

#### Curriculum Learning
- Save game states at progress milestones
- Probabilistically reset envs to checkpoints
- Weighted sampling toward higher progress states

---

## Implementation Tasks

### TODO: Rewrite pretrain_visual.py
- [ ] Stage 1A: Train tokenizer (autoencoder with MSE + LPIPS)
- [ ] Stage 1B: Train dynamics model (next latent prediction)
- [ ] Save pretrained encoder for RL

### TODO: Update train_with_cnn_reward.py
- [ ] Load pretrained encoder (freeze or fine-tune)
- [ ] Verify return filtering works (--return-filter 0.8)
- [ ] Add better logging for debugging

### TODO: Fix intro problem
- Agent gets stuck in title screen (never presses Start)
- Options:
  1. Pretrained encoder should help (understands game visuals)
  2. Create save state after intro
  3. Add exploration bonus

### TODO: Evaluation
- [ ] Compare trained vs random agent
- [ ] Visualize trajectories (GIF output)
- [ ] Track checkpoints reached over training

---

## Commands

```bash
# Phase 1: Pretrain visual encoder (Dreamer 4 style)
uv run python pretrain_visual.py --epochs 100

# Phase 3: RL training
uv run python train_with_cnn_reward.py \
    --reward-model reward_resnet34.pkl \
    --pretrained-encoder pretrained_encoder.pkl \
    --return-filter 0.8 \
    --iterations 50000

# Evaluate
uv run python evaluate_policy.py --policy checkpoints/policy_final.pkl --steps 2000
```

---

## Known Issues

1. **Intro loop**: Agent never presses Start, stuck on title screens
2. **Random policy**: After 10K iterations, policy was 99.9% max entropy
3. **Disk space**: Root filesystem is full, use `UV_CACHE_DIR=/media/milkkarten/data/cache/uv`

---

## Hardware

- GPU: Available (32GB)
- Potential: 2x 5090 for data parallel training
- Emulator: ~250K FPS across 32 envs

---

## References

- [Dreamer 4 Paper](https://arxiv.org/abs/2509.24527) - Scalable world models with video pretraining
- [Pokemon Red RL](https://github.com/PWhiddy/PokemonRedExperiments) - Peter Whidden's project
