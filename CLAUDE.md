# EmuRL - Pokemon Red RL Training

## Project Notes

- Always use `uv run python` (not conda)
- UV cache: `UV_CACHE_DIR=/media/milkkarten/data/cache/uv`
- Emulator: `from emurust import VecGameBoy`

---

## Methodology

### Overview

We train an RL agent to play Pokemon Red using a VLM-inspired architecture. The key insight is that we can leverage unlabeled speedrun videos to pretrain visual representations before RL training, similar to how VLMs pretrain on images before instruction tuning.

### Why This Approach?

1. **Data efficiency**: Speedrun videos provide rich visual priors about game progression without needing action labels
2. **Discrete tokens**: VQ-VAE tokenization creates a discrete visual language, enabling transformer-based world models
3. **Transfer learning**: Pretrained encoder understands game visuals (menus, battles, overworld) before seeing any rewards

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Visual Pretraining (unlabeled speedrun videos)                │
│  ┌─────────────────┐    ┌─────────────────┐                             │
│  │ Stage 1A: VQ-VAE │───▶│ Stage 1B: Dynamics│                          │
│  │ (image tokenizer)│    │ (next token pred) │                          │
│  └─────────────────┘    └─────────────────┘                             │
│           │                                                              │
│           ▼                                                              │
│  pretrained_encoder.pkl (encoder + codebook)                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 2: Reward Model (pairwise frame comparisons from speedruns)      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ResNet-34: frame → progress score                                │    │
│  │ Bradley-Terry loss: P(frame_b > frame_a) = sigmoid(score_b - a) │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  reward_resnet34.pkl (88.7% pairwise accuracy)                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 3: RL Training (PPO with pretrained encoder + reward model)      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ VQPolicy: VQ Encoder → Token Embeddings → Transformer → Actions │    │
│  │ Reward: progress_delta - time_penalty                           │    │
│  │ Curriculum: checkpoint states at progress milestones            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline Details

### Compact Architecture (~7M params per model, bf16)

All models use bf16 training for speed. Key dimensions:
- `embed_dim = 192`
- `encoder_channels = (32, 64, 128, 192)`
- `num_layers = 4`, `num_heads = 4`, `mlp_ratio = 3`
- `vocab_size = 512`, `token_grid = 9x10`

### Phase 1: Dreamer 4-Style Visual Pretraining

Train on speedrun videos (no actions needed).

#### Stage 1A: VQ-VAE Tokenizer (~5M params)
```python
# Architecture: CNN encoder -> VQ codebook -> CNN decoder
# Converts frames to discrete token grids (like image tokenizer for VLMs)

z = encoder(frame)              # (144, 160, 3) -> (9, 10, 192)  spatial features
tokens, z_q = quantize(z)       # (9, 10) discrete tokens, (9, 10, 192) quantized
recon = decoder(z_q)            # (9, 10, 192) -> (144, 160, 3)

# Loss: Reconstruction + VQ losses
loss = MSE(recon, frame) + codebook_loss + 0.25 * commitment_loss
```

#### Stage 1B: Dynamics Model (Next Token Prediction)
```python
# Architecture: Transformer over token sequence
# Predicts next frame's tokens from current frame's tokens

embeds_t = codebook[tokens_t]          # (90, 256) token embeddings
logits = transformer(embeds_t)          # (90, vocab_size) next token logits
tokens_t1 = argmax(logits)              # Predicted next tokens

# Loss: Cross-entropy per token position
loss = CrossEntropy(logits, tokens_t1_true)
```

**Data**: `data/speedruns/frames/` (64K frames from 5 speedrun videos)
**Token grid**: 9x10 = 90 tokens per frame
**Vocab size**: 512 discrete tokens

---

### Phase 2: Reward Model (DONE)

Two options available:

#### Option A: ResNet-34 (legacy, standalone)
- ResNet-34 trained on speedrun frame ordering
- `reward_resnet34.pkl` (88.7% pairwise accuracy)

#### Option B: VQ-based (uses shared backbone)
- VQ Encoder (frozen from Stage 1A) -> MLP head -> progress score
- Trained with `train_reward.py`
- Shares encoder+codebook with policy for consistency

---

### Phase 3: RL Training (PPO)

#### Architecture (VLM-style)
```
frame -> VQ Encoder -> tokens (9x10) -> TokenTransformer -> pooled -> policy logits
                    -> codebook      -> embeddings (90x256)        -> value estimate

# Same transformer architecture as Stage 1B dynamics model
# Can initialize from pretrained dynamics for transfer learning
```

**Key insight**: The dynamics model learns temporal reasoning ("what comes next")
which transfers well to policy learning ("what action leads to good outcomes").

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

### DONE: Rewrite pretrain_visual.py (VQ-VAE)
- [x] Stage 1A: VQ-VAE tokenizer (encoder + codebook + decoder)
- [x] Stage 1B: Transformer dynamics (next token prediction)
- [x] Export encoder + codebook for RL (`pretrained_encoder.pkl`)

### DONE: Update train_with_cnn_reward.py (VLM-style)
- [x] Add VQPolicy: VQ encoder -> token embeddings -> transformer -> policy/value
- [x] Load pretrained encoder + codebook weights
- [x] Keep legacy ViT policy for backward compatibility
- [ ] Verify return filtering works (--return-filter 0.8)

### TODO: Fix intro problem
- Agent gets stuck in title screen (never presses Start)
- Solution: Pretrained encoder should help (understands game visuals from speedruns)

### TODO: Evaluation
- [ ] Compare trained vs random agent
- [ ] Visualize trajectories (GIF output)
- [ ] Track checkpoints reached over training

---

## Commands

```bash
# Phase 1: Pretrain visual encoder (Dreamer 4 style - two stages)
UV_CACHE_DIR=/media/milkkarten/data/cache/uv uv run python pretrain_visual.py \
    --data data/speedruns/frames --epochs 100

# Or train stages separately:
uv run python pretrain_visual.py --stage 1a --epochs 100  # Tokenizer only
uv run python pretrain_visual.py --stage 1b --tokenizer pretrained_tokenizer.pkl --epochs 50  # Dynamics only

# Phase 2 (optional): Train VQ-based reward model with shared backbone
uv run python train_reward.py \
    --data data/speedruns/frames \
    --tokenizer pretrained_tokenizer.pkl \
    --epochs 50

# Phase 3: RL training with pretrained backbone (finetune everything)
# ~600 iter/hour, 12K iterations ≈ 20 hours
PYTHONUNBUFFERED=1 uv run python train_with_cnn_reward.py \
    --reward-model reward_resnet34.pkl \
    --pretrained-tokenizer pretrained_tokenizer.pkl \
    --pretrained-dynamics pretrained_dynamics.pkl \
    --return-filter 0.8 \
    --iterations 12000

# Resume from checkpoint if interrupted
uv run python train_with_cnn_reward.py \
    --reward-model reward_resnet34.pkl \
    --pretrained-tokenizer pretrained_tokenizer.pkl \
    --pretrained-dynamics pretrained_dynamics.pkl \
    --resume checkpoints/policy_1000.pkl \
    --iterations 12000

# Evaluate
uv run python evaluate_policy.py --policy checkpoints/policy_final.pkl --steps 2000
```

---

## Known Issues

1. **Intro loop**: Agent never presses Start, stuck on title screens
2. **Random policy**: After 10K iterations, policy was 99.9% max entropy
3. **Disk space**: Root filesystem is full, use `UV_CACHE_DIR=/media/milkkarten/data/cache/uv`

### VQ-VAE Observations (100 epochs)

**Codebook usage**: Stabilized at ~45% (228 of 512 codes active)
- Much better than initial collapse to 17-20%

**Training metrics (100 epochs):**
- Tokenizer (3.9M params): Recon 0.061→0.007, VQ loss 0.52→0.012
- Dynamics (1.6M params): Accuracy 51%→72% next-token prediction
- Training time: ~15 min (tokenizer) + ~5 min (dynamics)

---

## Hardware

- GPU: Available (32GB)
- Potential: 2x 5090 for data parallel training
- Emulator: ~250K FPS across 32 envs

---

## References

- [Dreamer 4 Paper](https://arxiv.org/abs/2509.24527) - Scalable world models with video pretraining
- [Pokemon Red RL](https://github.com/PWhiddy/PokemonRedExperiments) - Peter Whidden's project
