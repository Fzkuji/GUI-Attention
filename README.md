# GUI-Attention

**Saccade Foveation for Efficient GUI Grounding**

## Overview

GUI-Attention introduces a **saccade foveation** mechanism for GUI element grounding, inspired by human eye movement. Instead of processing screenshots at uniform high resolution (expensive, ~7,000+ visual tokens) or using multi-step zoom-in pipelines (slow), we use:

- **Peripheral vision**: low-resolution full image (~1,300 patches) for coarse localization
- **Foveal vision**: one high-resolution crop (~2,000 patches) at the attention focus
- **Saccade**: the focus point moves across rounds if needed

This gives comparable accuracy to full-resolution methods while using **2-4x fewer visual tokens** and being **2.5x faster**.

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

## Results (ScreenSpot-Pro, 5K training data)

| Method | hit@1 | overlap@1 | Avg time | Visual tokens |
|--------|-------|-----------|----------|---------------|
| GUI-Actor 5K SFT | 20.75% | 27.07% | 1.80s | ~7,000 |
| Ours single-round | 7.15% | 19.04% | 0.43s | ~1,300 |
| **Ours saccade 3 rounds** | **21.13%** | **36.50%** | **0.72s** | **~1,900 avg** |

- 97.3% of samples finish in 2 rounds, 2.7% need 3 rounds
- Both methods trained on the same 5K GUIAct subset for fair comparison

## Installation

```bash
conda create -n gui-attention python=3.11
conda activate gui-attention
pip install -e .
```

No external dependencies on GUI-AIMA or GUI-Actor source code (fully self-contained).

## Training

```bash
python -m gui_attention.train \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /path/to/guiact_5k_seed42.json \
    --image_folder /path/to/GUIAct/web_imgs \
    --output_dir /path/to/output \
    --crop_ratio 0.3 --max_saccade_rounds 3 \
    --lora_r 32 --lora_alpha 64 \
    --action_head_lr 1e-4 --lora_lr 5e-5 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --num_train_epochs 1 --bf16 true
```

See `jobs/train_ours_5k.sh` for a complete example.

## Evaluation

```bash
# Single-round (no saccade)
python eval/eval_screenspot_pro_aligned.py \
    --checkpoint /path/to/checkpoint \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /path/to/ScreenSpot-Pro \
    --rounds 1

# Multi-round saccade
python eval/eval_screenspot_pro_aligned.py \
    --checkpoint /path/to/checkpoint \
    --base_model Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /path/to/ScreenSpot-Pro \
    --rounds 3 --crop_ratio 0.3
```

See `jobs/eval_comparison.sh` for the full comparison pipeline.

## Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
GUI-Attention/
├── src/gui_attention/
│   ├── train.py           # SaccadeTrainer: multi-round training with model's own predictions
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
│   ├── eval_screenspot_pro_aligned.py   # ScreenSpot-Pro evaluation
│   └── time_gui_actor.py               # GUI-Actor inference timing
├── jobs/                  # SLURM job scripts
│   ├── train_ours_5k.sh   # Train our method on 5K subset
│   ├── train_guiactor_5k.sh  # Train GUI-Actor baseline
│   ├── eval_comparison.sh    # Run all evaluations
│   └── ...
└── tests/                 # Unit tests (43 tests)
```

## Related Projects

- [GUI-Actor](https://github.com/cckevinn/GUI-Actor) (NeurIPS 2025) — Action head architecture baseline
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language backbone

## License

Apache License 2.0
