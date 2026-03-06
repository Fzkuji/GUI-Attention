# GUI-Attention

**Saccade Foveation for Efficient GUI Grounding**

Multi-round visual search: low-res overview → crop/zoom → refine, inspired by human saccadic eye movement.

## Architecture (v15)

- **Base model**: [GUI-Actor-3B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-3B-Qwen2.5-VL) (Qwen2.5-VL-3B + pointer head)
- **Dual Head**: LookHead (where to crop next) + ClickHead (precise click on crop)
- **Dual Tokens**: `<look_pad>` (explore) / `<pointer_pad>` (commit/click) — model decides autoregressively
- **LoRA** fine-tuning on backbone, full-param on heads

```
Round 0: [low-res full image ~1M pixels] [instruction]
         → LookHead → predict crop center

Round 1+: [low-res full] + [high-res crop ~924×924]
         → Model generates <look_pad> (saccade) or <pointer_pad> (click)
         → <look_pad>  → LookHead → next crop
         → <pointer_pad> → ClickHead → precise click coordinate

Stop: model generates <pointer_pad>, or max_rounds reached
```

## Setup

```bash
git clone https://github.com/Fzkuji/GUI-Attention.git
cd GUI-Attention
pip install -e .
```

## Training

**Important**: Base model must be GUI-Actor (not vanilla Qwen2.5-VL). The backbone hidden states and pointer head weights must match.

```bash
# On training server (8× GPU)
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention
NUM_GPUS=8 bash jobs/train.sh
```

Key config in `jobs/train.sh`:
- `--model_name_or_path`: GUI-Actor-3B-Qwen2.5-VL
- `--click_head_from`: Same GUI-Actor model (ClickHead init from pointer_head)
- `--use_dual_tokens true`: Enable `<look_pad>/<pointer_pad>` decision
- `--click_phase_step 0`: Both heads train from step 0
- `--low_res_max_pixels 1001600`: Match GUI-Actor's 1M resolution
- `--crop_size 308 --crop_upscale 3`: 308px crop → ×3 upscale → 924px (1089 tokens)
- `--max_saccade_rounds 6`: Up to 5 crop rounds + 1 initial

To resume from checkpoint:
```bash
# Add --resume_ckpt to train.sh, or HuggingFace Trainer auto-resumes from output_dir
NUM_GPUS=8 bash jobs/train.sh
```

## Evaluation

All eval scripts also use **GUI-Actor as base model**.

```bash
# ScreenSpot-Pro (8 GPU, DDP)
torchrun --nproc_per_node=8 eval/eval_screenspot.py \
    --dataset pro \
    --data_dir /path/to/ScreenSpot-Pro \
    --checkpoint /path/to/checkpoint-500 \
    --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
    --max_rounds 6 \
    --use_dual_tokens

# ScreenSpot v1 (downloads from HuggingFace)
torchrun --nproc_per_node=8 eval/eval_screenspot.py \
    --dataset v1 \
    --checkpoint /path/to/checkpoint-500 \
    --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
    --max_rounds 6 \
    --use_dual_tokens

# ScreenSpot v2
torchrun --nproc_per_node=8 eval/eval_screenspot.py \
    --dataset v2 \
    --data_dir /path/to/v2_data \
    --checkpoint /path/to/checkpoint-500 \
    --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
    --max_rounds 6 \
    --use_dual_tokens
```

### Evaluate GUI-Actor baseline (no saccade)

Use GUI-Actor's own eval code:
```bash
cd /path/to/GUI-Actor
# Delete old results first (script silently exits if results exist)
rm -f screenspot-Pro_all_preds_StandardResize.txt screenspot-Pro_all_preds_StandardResize.json
PYTHONPATH=src:$PYTHONPATH HF_HUB_OFFLINE=1 python eval/screenSpot_pro.py \
    --model_name_or_path /path/to/GUI-Actor-3B-Qwen2.5-VL \
    --data_path /path/to/ScreenSpot-Pro \
    --model_type qwen25vl
```

### Evaluate on training data

```bash
PYTHONPATH=src:$PYTHONPATH HF_HUB_OFFLINE=1 python eval/eval_train_hit.py \
    --model_path /path/to/GUI-Actor-3B-Qwen2.5-VL \
    --data_path /path/to/guiact_bbox.json,/path/to/androidcontrol_bbox.json \
    --image_folder /path/to/images,/path/to/images \
    --max_samples 2000
```

## Visualization

```bash
python eval/visualize_saccade.py \
    --checkpoint /path/to/checkpoint \
    --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
    --screenspot_dir /path/to/ScreenSpot-Pro \
    --sample_index 42 \
    --output viz_sample42.png
```

## GRPO Training (after SFT)

```bash
SFT_CKPT=/path/to/sft-checkpoint NUM_GPUS=8 bash jobs/train_grpo.sh
```

## Project Structure

```
src/gui_attention/
  train.py          # SaccadeTrainer: multi-round training with per-round backward
  model.py          # Qwen25VLWithDualHead: backbone + LookHead + ClickHead
  dual_head.py      # DualActionHead (LookHead + ClickHead)
  builder.py        # MultiRoundInputBuilder: tokenizes multi-image conversations
  inference.py      # Autoregressive saccade inference (generate → head)
  foveation.py      # SaccadeLoop: round state tracking
  grpo.py           # GRPO reinforcement learning trainer
  attention.py      # Hidden state extraction + spatial utilities
  crop.py           # Image cropping + coordinate helpers
  labels.py         # Binary/Gaussian labels + crop masking
  constants.py      # Tokens, chat template, resolution defaults
eval/
  eval_screenspot.py    # Unified ScreenSpot v1/v2/Pro evaluation
  eval_train_hit.py     # Single-round hit rate on training data
  visualize_saccade.py  # Multi-round visualization
jobs/
  train.sh              # SFT training script
  train_grpo.sh         # GRPO training script
  eval_all_screenspot.sh
```

## Results (v15 checkpoint-500)

| Benchmark | Hit@1 | Overlap@1 |
|-----------|-------|-----------|
| ScreenSpot-Pro | 18.34% | 29.41% |
| ScreenSpot v1 | 70.75% | 78.54% |
| ScreenSpot v2 | 74.82% | 81.35% |

GUI-Actor baseline: 43.20% on ScreenSpot-Pro (single-round, no saccade).

## Related Projects

- [GUI-Actor](https://github.com/microsoft/GUI-Actor) — Pointer head architecture (base model)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language backbone

## License

Apache License 2.0
