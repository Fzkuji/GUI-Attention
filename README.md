# GUI-Attention

**Saccade Foveation for Efficient GUI Grounding**

## Overview

GUI-Attention introduces a **saccade foveation** mechanism for GUI element grounding, inspired by human eye movement. Instead of processing screenshots at uniform high resolution (expensive, ~7,000+ visual tokens), we use:

- **Peripheral vision**: low-resolution full image (~500 patches) for coarse localization
- **Foveal vision**: high-resolution crop (~1,300 patches) at the attention focus
- **Saccade**: the focus point moves across rounds if the target isn't found

This gives comparable accuracy to full-resolution methods while using **2-4× fewer visual tokens**.

## Architecture

```
Round 0: [low-res full image ~500 patches] [instruction] [anchor]
         → ActionHead → attention → select focus point

Round 1: [low-res full (focus area masked)] + [high-res crop ~1,300 patches]
         → ActionHead (with mask)
         → argmax in high-res → CLICK (target found)
         → argmax in low-res → SACCADE (move focus, next round)

Round 2+: [low-res + new high-res crop] → repeat...

Stop: argmax in high-res, or max_rounds reached
```

### ActionHead

- Visual encoder embeddings → Self-Attention → MLP_V → projected visual features
- LLM last-layer hidden states at `<pointer_pad>` → MLP_T → anchor query
- Scaled dot-product → softmax → attention weights over visual patches
- Training: KL divergence loss against binary overlap labels

## Quick Start

### Setup

```bash
git clone https://github.com/Fzkuji/GUI-Attention.git
cd GUI-Attention
pip install -e ".[train,eval]"
pip install flash-attn --no-build-isolation  # recommended
```

### Training

```bash
# Single-script setup (fresh server: env + data + training)
bash jobs/setup_and_train.sh

# Or manual training with torchrun
NUM_GPUS=8 bash jobs/train.sh
```

Key training arguments (configured in `jobs/train.sh`):

| Argument | Default | Description |
|----------|---------|-------------|
| `low_res_max_pixels` | 1,003,520 | Low-res resolution (~1,300 tokens) |
| `crop_target_pixels` | 1,003,520 | Crop resolution after upsampling |
| `crop_ratio` | 0.3 | Crop size as fraction of image |
| `max_saccade_rounds` | 4 | Max rounds (1 low-res + 3 crops) |
| `use_lora` | true | LoRA fine-tuning (false = full-param) |
| `action_head_lr` | 5e-5 | Learning rate for action head |
| `lora_lr` | 5e-5 | Learning rate for backbone (LoRA/full) |

### Evaluation

```bash
# All ScreenSpot benchmarks
bash jobs/eval_all_screenspot.sh /path/to/checkpoint /path/to/base_model 3 0.3 cuda:0

# ScreenSpot-Pro only
python eval/eval_screenspot_pro_aligned.py \
    --checkpoint /path/to/checkpoint \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /path/to/ScreenSpot-Pro \
    --rounds 3 --crop_ratio 0.3
```

### Visualization

Visualize the multi-round saccade process on any image:

```bash
# From ScreenSpot-Pro dataset
python eval/visualize_saccade.py \
    --checkpoint /path/to/checkpoint \
    --base_model /path/to/Qwen2.5-VL-3B-Instruct \
    --screenspot_dir /path/to/ScreenSpot-Pro \
    --sample_index 42 \
    --output viz_sample42.png

# Custom image
python eval/visualize_saccade.py \
    --checkpoint /path/to/checkpoint \
    --base_model /path/to/Qwen2.5-VL-3B-Instruct \
    --image screenshot.png \
    --instruction "Click the search button" \
    --gt_bbox 0.45,0.32,0.55,0.38 \
    --output viz_output.png
```

Output: one row per round showing attention heatmaps, crop regions, predicted clicks, and GT.

### Training Data

We use the same datasets as [GUI-Actor](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data):

| Dataset | Samples | Content |
|---------|---------|---------|
| GUIAct | ~42K | Web screenshots |
| AndroidControl | ~50K | Mobile app automation |
| Wave-UI | ~25K | Web UI grounding |
| UGround | ~775K | General GUI grounding |

## Project Structure

```
GUI-Attention/
├── src/gui_attention/
│   ├── train.py        # SaccadeTrainer: multi-round training loop
│   ├── model.py        # Qwen25VLWithActionHead: backbone + LoRA + ActionHead
│   ├── action_head.py  # Self-Attn + MLP_V + MLP_T + KL loss
│   ├── inference.py    # BFS region prediction + multi-round saccade
│   ├── builder.py      # MultiRoundInputBuilder: tokenizes multi-image conversations
│   ├── foveation.py    # SaccadeLoop: round decisions (crop/saccade/stop)
│   ├── attention.py    # Hidden state extraction + spatial utilities
│   ├── labels.py       # Binary overlap labels + crop masking
│   ├── constants.py    # Pointer tokens, resolution, chat template
│   └── crop.py         # Image cropping + coordinate helpers
├── eval/
│   ├── eval_screenspot_pro_aligned.py  # ScreenSpot-Pro evaluation
│   ├── eval_screenspot.py              # ScreenSpot v1 evaluation
│   ├── eval_screenspot_v2.py           # ScreenSpot v2 evaluation
│   └── visualize_saccade.py            # Multi-round visualization tool
├── jobs/
│   ├── train.sh              # Main training script
│   ├── setup_and_train.sh    # One-script server setup
│   ├── eval_all_screenspot.sh
│   └── download_gui_actor_data.sh
├── tests/                    # Unit tests
└── pyproject.toml
```

## Related Projects

- [GUI-Actor](https://github.com/microsoft/GUI-Actor) (NeurIPS 2025) — Action head architecture
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language backbone

## License

Apache License 2.0
