# EmuRL — Pokemon Red RL Agent

Train a reinforcement learning agent to play Pokemon Red using a VLM-inspired architecture. Visual representations are pretrained on unlabeled speedrun videos before RL training, giving the agent game knowledge (menus, battles, overworld) before it sees any rewards.

## Pipeline Overview

```
Phase 1: Visual Pretraining   (speedrun videos, no action labels)
  ├── Stage 1A: VQ-VAE        frames → discrete 9×10 token grid
  └── Stage 1B: Dynamics      next-token prediction (temporal reasoning)
            │
            ▼
Phase 2: Reward Model         (pairwise frame ranking from speedruns)
  └── ResNet-34               Bradley-Terry loss → progress score
            │
            ▼
Phase 3: RL Training          (PPO with pretrained encoder + reward model)
  └── VQPolicy                VQ tokens → Transformer → actions
```

---

## Directory Structure

```
EmuRL/
├── phase1_pretrain/
│   └── pretrain_visual.py        VQ-VAE tokenizer + dynamics model
│
├── phase2_reward/
│   ├── train_reward_cnn.py       ResNet-34 reward model (main)
│   ├── train_reward.py           VQ-based reward (shared backbone)
│   ├── train_reward_global.py    Global progress variant
│   └── train_reward_sequence.py  Sequence-based variant
│
├── phase3_rl/
│   └── train_with_cnn_reward.py  PPO training loop
│
├── data_collection/
│   ├── scrape_speedruns.py       Fetch + download speedrun videos (main)
│   ├── preprocess_videos.py      Crop to Game Boy screen, extract frames
│   ├── collect_speedrun_data.py  Legacy collection pipeline
│   ├── scrape_youtube.py         YouTube gameplay search
│   ├── scrape_archive.py         Archive.org scraper
│   ├── download_youtube_videos.py
│   ├── download_twitch_videos.py
│   └── download_letsplay_playlist.py
│
├── tools/
│   ├── ood_detector.py           OOD detection (penalize off-distribution states)
│   ├── evaluate_policy.py        Run policy and measure progress
│   ├── validate_training.py      Sanity checks before long runs
│   └── analyze_video_crops.py    Visualize crop config results
│
├── configs/
│   └── video_crop_config.json    Per-video Game Boy screen crop coordinates
│
├── data/                         (gitignored — store locally or on shared storage)
│   └── speedruns/frames/         ~64K extracted frames from 5 speedruns
│
└── checkpoints/                  (gitignored — policy snapshots)
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (bf16 training; 8GB+ recommended, 32GB ideal)
- `uv` package manager

### Install

```bash
# Clone the repo
git clone <repo-url> && cd EmuRL

# Install dependencies (uv handles the venv)
uv sync
```

> **Note**: The `emurust` Game Boy emulator is a local Rust package referenced in `pyproject.toml`. If it's not on your machine, ask the project lead for the package path and update `pyproject.toml` accordingly.

### Environment

```bash
# Always use this prefix for commands (avoids root disk fill)
export UV_CACHE_DIR=/path/to/large/disk/cache/uv
```

---

## Running the Pipeline

### Phase 1: Visual Pretraining

Train a VQ-VAE image tokenizer + dynamics model on speedrun frames.

```bash
# Full two-stage pipeline (recommended)
uv run python phase1_pretrain/pretrain_visual.py \
    --data data/speedruns/frames --epochs 100

# Or train stages separately:
uv run python phase1_pretrain/pretrain_visual.py --stage 1a --epochs 100
uv run python phase1_pretrain/pretrain_visual.py \
    --stage 1b --tokenizer pretrained_tokenizer.pkl --epochs 50
```

**Outputs**: `pretrained_tokenizer.pkl`, `pretrained_encoder.pkl`, `pretrained_dynamics.pkl`

**Expected results** (~20 min on 32GB GPU):
- Tokenizer: reconstruction loss 0.061 → 0.007, ~45% codebook utilization
- Dynamics: 51% → 72% next-token accuracy

---

### Phase 2: Reward Model

Train a progress detector on speedrun frame ordering.

```bash
# Main option: ResNet-34 standalone (fast, 88.7% pairwise accuracy)
uv run python phase2_reward/train_reward_cnn.py \
    --data data/speedruns/frames --model 34 --epochs 50

# Alternative: VQ-based reward (shares backbone with policy, requires Phase 1)
uv run python phase2_reward/train_reward.py \
    --data data/speedruns/frames \
    --tokenizer pretrained_tokenizer.pkl --epochs 50
```

**Output**: `reward_resnet34.pkl`

**Training signal**: Frame pairs from the same speedrun — later frames should rank higher (Bradley-Terry loss).

---

### Phase 3: RL Training

PPO training with the pretrained encoder and reward model.

```bash
# Full run (~20 hours, ~600 iter/hr)
PYTHONUNBUFFERED=1 uv run python phase3_rl/train_with_cnn_reward.py \
    --reward-model reward_resnet34.pkl \
    --pretrained-tokenizer pretrained_tokenizer.pkl \
    --pretrained-dynamics pretrained_dynamics.pkl \
    --return-filter 0.8 \
    --iterations 12000

# Resume from checkpoint
uv run python phase3_rl/train_with_cnn_reward.py \
    --reward-model reward_resnet34.pkl \
    --pretrained-tokenizer pretrained_tokenizer.pkl \
    --pretrained-dynamics pretrained_dynamics.pkl \
    --resume checkpoints/policy_1000.pkl \
    --iterations 12000
```

**Key hyperparameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_skip` | 30 | Emulator frames per RL step (~0.5s game time) |
| `reward_interval` | 64 | Steps between reward computation |
| `time_penalty` | 0.001 | Cost per step (encourages speed) |
| `return_filter` | 0.8 | Train on top 80% of episode returns |
| `gamma` | 0.99 | Discount factor |

**Reward function**: `progress_delta - time_penalty`
where `progress_delta = reward_model(frame_t+1) - reward_model(frame_t)`

---

### Evaluation

```bash
uv run python tools/evaluate_policy.py \
    --policy checkpoints/policy_final.pkl --steps 2000
```

### Validate Before Long Runs

```bash
uv run python tools/validate_training.py \
    --reward-model reward_resnet34.pkl
```

---

## Data Collection

Speedrun video pipeline for expanding the training dataset.

```bash
# 1. Fetch metadata (1505 runs from speedrun.com)
uv run python data_collection/scrape_speedruns.py \
    --fetch-runs --output data/speedrun_metadata.json

# 2. Download videos (720p max)
uv run python data_collection/scrape_speedruns.py \
    --download \
    --metadata data/speedrun_metadata.json \
    --videos /path/to/storage/videos \
    --max-videos 50

# 3. Extract frames (2 FPS, cropped to Game Boy screen)
uv run python data_collection/scrape_speedruns.py \
    --extract-frames \
    --videos /path/to/storage/videos \
    --frames /path/to/storage/frames
```

**Crop config**: `configs/video_crop_config.json` stores per-video crop coordinates (Game Boy screen location varies by streamer setup). Use `tools/analyze_video_crops.py` to validate.

---

## Architecture

### VQPolicy (Phase 3)

```
frame (144×160×3)
    → VQ Encoder (CNN)           → spatial feature map (9×10×192)
    → VQ Codebook (512 tokens)   → discrete token grid (9×10) = 90 tokens/frame
    → Token Embeddings           → (90×192)
    → Transformer (4L, 4H)       → pooled representation
    → Policy head                → action logits (8 buttons)
    → Value head                 → scalar value estimate
```

### Model sizes (bf16)

| Model | Params | Notes |
|-------|--------|-------|
| VQ-VAE tokenizer | ~3.9M | encoder + codebook + decoder |
| Dynamics transformer | ~1.6M | next-token prediction |
| ResNet-34 reward | ~21M | pretrained backbone |
| VQPolicy | ~5.5M | encoder shared from Phase 1 |

---

## OOD Detection

Penalize the agent for visiting states that speedrunners never do (e.g., wandering off-route, soft-locks).

```python
from tools.ood_detector import OODDetector

detector = OODDetector.from_tokenizer(
    "pretrained_tokenizer.pkl",
    "pretrained_dynamics.pkl"
)
ood_scores = detector.compute_combined_ood_scores(frames_t, frames_t1)
reward_mult = detector.get_reward_multiplier(ood_scores)
adjusted_rewards = rewards * reward_mult
```

---

## Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| Intro loop | Open | Agent never presses Start; stuck on title screen |
| Policy entropy collapse | Open | After 10K iter, policy → 99.9% max entropy |
| Disk space | Ongoing | Use `UV_CACHE_DIR` on large disk |

The intro loop and entropy collapse are likely related — the agent isn't receiving a meaningful learning signal early. Expanding the training data and enabling OOD penalties are the planned fixes.

---

## Contributing

The main open tasks are:

1. **Expand training data** — download 50+ speedrun videos and retrain the reward model
2. **Fix intro problem** — agent needs to learn to press Start and get through the title/name screens
3. **Tune RL hyperparameters** — investigate the entropy collapse at 10K iterations
4. **Evaluation** — build comparison tooling (trained vs random agent, trajectory GIFs)

Ask the project lead for:
- The `emurust` emulator package path
- Access to existing model weights (`.pkl` files)
- The Pokemon Red ROM

---

## References

- [Dreamer 4](https://arxiv.org/abs/2509.24527) — Scalable world models with video pretraining (core inspiration)
- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) — Peter Whidden's original Pokemon Red RL work

---

## License

MIT
