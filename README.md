# GUI-Attention

**Saccade Foveation for Efficient GUI Grounding**

## Overview

GUI-Attention introduces a **saccade foveation** mechanism for GUI element grounding, inspired by human eye movement. Instead of processing screenshots at uniform high resolution (expensive, ~7,000+ visual tokens), we use:

- **Peripheral vision**: low-resolution full image (~1,300 patches) for coarse localization
- **Foveal vision**: one high-resolution crop (~2,000 patches) at the attention focus
- **Saccade**: the focus point moves across rounds if the target isn't found

This gives comparable accuracy to full-resolution methods while using **2-4x fewer visual tokens**.

### Key Differences from GUI-Actor

| | GUI-Actor | Ours |
|---|---|---|
| Visual features | Vision encoder embeddings (pre-LLM) | LLM last-layer hidden states (post-LLM, text-aware) |
| Action head | Self-Attn + MLP_V + MLP_T | MLP_V + MLP_T (no self-attn needed) |
| Backbone training | Full fine-tune (two-stage) | LoRA (single-stage) |
| Inference | Single image, single round | Multi-round saccade foveation |
| Resolution | Uniform high-res (~5.7M pixels) | Low-res full + high-res crop |

## Architecture

```
Round 0: [low-res full image ~1,300 patches] [instruction] [anchor]
         -> LLM -> last-layer hidden states -> ActionHead -> attention
         -> select focus point

Round 1: [low-res full (focus area masked)] + [high-res crop ~2,000 patches]
         -> LLM -> ActionHead (with mask)
         -> argmax in high-res patch -> CLICK (target found)
         -> argmax in low-res patch -> SACCADE (move focus, next round)

Round 2: [low-res + new high-res crop] -> repeat...

Stop: argmax in high-res, or max_rounds reached
```

Token budget stays constant: ~3,300-4,300 per round (vs GUI-Actor's ~7,000+).

## Results

### ScreenSpot-Pro (5K training subset)

| Method | hit@1 | overlap@1 | Avg time | Visual tokens |
|--------|-------|-----------|----------|---------------|
| GUI-Actor 5K SFT | 20.75% | 27.07% | 1.80s | ~7,000 |
| Ours single-round | 7.15% | 19.04% | 0.43s | ~1,300 |
| **Ours saccade 3 rounds** | **21.13%** | **36.50%** | **0.72s** | **~1,900 avg** |

### Full GUIAct Training (42K samples)

| Benchmark | All-text | All-icon | All-avg (hit@1) |
|-----------|---------|---------|-----------------|
| ScreenSpot v1 | 90.24 | 64.87 | **78.77** |
| ScreenSpot v2 | 97.29 | 77.49 | **80.97** |
| ScreenSpot-Pro | 39.30 | 6.62 | **26.82** |

## Quick Start

### One-Script Setup

For a fresh server, the setup script handles everything — environment, model, data, training, and evaluation:

```bash
git clone https://github.com/Fzkuji/GUI-Attention.git
cd GUI-Attention
bash jobs/setup_and_train.sh
```

Customize with environment variables:

```bash
# Use 4 GPUs, train on 5 datasets (skip UGround)
NUM_GPUS=4 DATASETS="guiact,guienv,amex,androidcontrol,waveui" bash jobs/setup_and_train.sh

# Custom workspace directory
WORK_DIR=/mnt/data/workspace bash jobs/setup_and_train.sh

# Only download data (skip training and eval)
SKIP_TRAIN=1 SKIP_EVAL=1 bash jobs/setup_and_train.sh

# Only train (data already downloaded)
SKIP_DOWNLOAD=1 bash jobs/setup_and_train.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `~/GUI-Attention-Workspace` | Root directory for data, models, results |
| `NUM_GPUS` | `2` | Number of GPUs for training |
| `DATASETS` | all 6 | Comma-separated: `guiact,guienv,amex,androidcontrol,waveui,uground` |
| `MAX_EPOCHS` | `1` | Training epochs |
| `CONDA_ENV` | `gui-attention` | Conda environment name |
| `SKIP_DOWNLOAD` | `0` | Set to `1` to skip data download |
| `SKIP_TRAIN` | `0` | Set to `1` to skip training |
| `SKIP_EVAL` | `0` | Set to `1` to skip evaluation |

### Manual Setup

```bash
conda create -n gui-attention python=3.10
conda activate gui-attention
pip install -e ".[train,eval]"
pip install flash-attn --no-build-isolation  # optional
```

No external dependencies on GUI-AIMA or GUI-Actor source code (fully self-contained).

## Training

### Single Dataset

```bash
export PYTHONPATH=src:$PYTHONPATH

python src/gui_attention/train.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /path/to/guiact_bbox.json \
    --image_folder /path/to/GUIAct/web_imgs \
    --output_dir /path/to/output \
    --crop_ratio 0.3 --max_saccade_rounds 3 \
    --lora_r 32 --lora_alpha 64 \
    --action_head_lr 1e-4 --lora_lr 5e-5 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --num_train_epochs 1 --bf16 true
```

### Multiple Datasets

Pass comma-separated paths for `--data_path` and `--image_folder`:

```bash
torchrun --nproc_per_node=2 src/gui_attention/train.py \
    --data_path "/data/guiact_bbox.json,/data/guienv_bbox.json,/data/amex_bbox.json" \
    --image_folder "/data/GUIAct/web_imgs,/data/GUIEnv/guienvs/images,/data/AMEX/screenshots" \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir /path/to/output \
    ...
```

### Training Data

We use the same 6 datasets as [GUI-Actor](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data):

| Dataset | Samples | Size | Content |
|---------|---------|------|---------|
| GUIAct | ~42K | 4 GB | Web screenshots |
| GUIEnv | ~80K | 6 GB | Web environments |
| Wave-UI | ~25K | 24 GB | Web UI grounding |
| AndroidControl | ~50K | 49 GB | Mobile app automation |
| AMEX | ~100K | 92 GB | Mobile apps (110 applications) |
| UGround | ~775K | 256 GB | General GUI grounding |

## Evaluation

### All ScreenSpot Benchmarks

```bash
bash jobs/eval_all_screenspot.sh /path/to/checkpoint /path/to/base_model 3 0.3 cuda:0
```

This runs ScreenSpot-Pro, v1, and v2 sequentially.

### Individual Benchmarks

```bash
export PYTHONPATH=src:$PYTHONPATH

# ScreenSpot-Pro (local data)
python eval/eval_screenspot_pro_aligned.py \
    --checkpoint /path/to/checkpoint \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /path/to/ScreenSpot-Pro \
    --rounds 3 --crop_ratio 0.3

# ScreenSpot v1 (auto-downloads from HuggingFace)
python eval/eval_screenspot.py \
    --checkpoint /path/to/checkpoint \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --rounds 3 --crop_ratio 0.3

# ScreenSpot v2 (auto-downloads from HuggingFace)
python eval/eval_screenspot_v2.py \
    --checkpoint /path/to/checkpoint \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --rounds 3 --crop_ratio 0.3
```

## Tests

```bash
python -m pytest tests/ -v
```

43 tests covering action head, labels, training logic, and integration.

## Project Structure

```
GUI-Attention/
├── src/gui_attention/
│   ├── train.py           # SaccadeTrainer: multi-round teacher-forcing training
│   ├── model.py           # Qwen25VLWithActionHead: backbone + LoRA + ActionHead
│   ├── action_head.py     # ActionHead: MLP_V + MLP_T + scaled dot product + KL loss
│   ├── inference.py       # BFS region prediction + multi-round saccade inference
│   ├── builder.py         # MultiRoundInputBuilder: tokenizes multi-image conversations
│   ├── foveation.py       # SaccadeLoop: round decisions (crop / saccade / stop)
│   ├── attention.py       # Hidden state extraction from LLM last layer
│   ├── labels.py          # Binary overlap labels + overlap masking
│   ├── constants.py       # Pointer tokens, resolution levels, chat template
│   ├── crop.py            # Image cropping utilities
│   └── sampling.py        # Attention sampling / argmax prediction
├── eval/
│   ├── eval_screenspot_pro_aligned.py  # ScreenSpot-Pro evaluation
│   ├── eval_screenspot.py              # ScreenSpot v1 evaluation
│   └── eval_screenspot_v2.py           # ScreenSpot v2 evaluation
├── jobs/
│   ├── setup_and_train.sh   # One-script setup for fresh servers
│   ├── train_all_2gpu.sh    # Train on all 6 datasets (2 GPU)
│   ├── train_guiact_2gpu.sh # Train on GUIAct only (2 GPU)
│   ├── eval_all_screenspot.sh # Run all ScreenSpot evaluations
│   └── ...
├── tests/                   # Unit tests (43 tests)
└── pyproject.toml
```

## Related Projects

- [GUI-Actor](https://github.com/microsoft/GUI-Actor) (NeurIPS 2025) — Action head architecture baseline
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language backbone

## License

Apache License 2.0
