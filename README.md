# EmuRL

RL training framework for emulated games, starting with Pokemon Red.

## Features

- **Visual pretraining**: DreamerV4-style contrastive learning on gameplay videos
- **CNN reward model**: ResNet-based progress detection from speedrun videos
- **PPO training**: With return-weighted filtering and curriculum learning
- **Progress checkpointing**: Automatic save states at progress milestones

## Installation

```bash
# Install EmuRust (Game Boy emulator)
pip install ../EmuRust

# Install EmuRL
pip install -e .
```

Or with uv:
```bash
uv sync
```

## Quick Start

### 1. Collect training data from speedrun videos

```bash
python collect_speedrun_data.py --urls speedrun_urls.txt --output data/speedruns
```

### 2. Train CNN reward model

```bash
python train_reward_cnn.py --data data/speedruns/frames --model 34 --epochs 50
```

### 3. Pretrain visual encoder (optional but recommended)

```bash
python pretrain_visual.py --data data/speedruns/frames --epochs 50
```

### 4. Train RL policy

```bash
python train_with_cnn_reward.py \
    --reward-model reward_resnet34.pkl \
    --pretrained-encoder pretrained_encoder.pkl \
    --return-filter 0.8 \
    --iterations 50000
```

### 5. Evaluate

```bash
python evaluate_policy.py --policy checkpoints/policy_final.pkl
```

## Architecture

```
Speedrun Videos → CNN Reward Model (ResNet-34)
                         ↓
                  Progress Scores
                         ↓
Game Boy Emulator → Policy (ViT) → Actions
     (EmuRust)           ↓
                  PPO Training
```

## Configuration

Key hyperparameters in `train_with_cnn_reward.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_skip` | 30 | Emulator frames per RL step |
| `reward_interval` | 64 | Steps between reward computation |
| `return_filter` | 0.8 | Train on top 80% of returns |
| `time_penalty` | 0.001 | Cost per step for faster progress |

## License

MIT
