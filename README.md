# GUI-Attention

**From Efficient Grounding to Efficient Agent: Foveated Visual Attention for GUI Understanding**

<p align="center">
  <img src="assets/images/overview.png" width="90%">
</p>

## 🔥 Highlights

- **Single-pass foveated grounding** — replaces multi-step zoom-in with human-vision-inspired multi-resolution processing
- **5x token reduction** — from ~18,800 to ~5,000 visual tokens without accuracy loss
- **Plug-and-play** — compatible with mainstream VLM backbones (Qwen2.5-VL, etc.)
- **State-of-the-art** on ScreenSpot-Pro, ScreenSpot-V2, and OSWorld benchmarks

## 📋 Overview

GUI-Attention introduces a foveated visual attention mechanism for GUI grounding that mimics human visual perception. Instead of processing the entire screen at uniform high resolution (expensive) or using multi-step zoom-in pipelines (slow), we apply a single-pass multi-resolution scheme: high resolution at the attention focus, progressively lower resolution in the periphery.

## 🚀 Quick Start

### Installation

```bash
conda create -n gui-attention python=3.11
conda activate gui-attention
pip install -e .
```

Requires [GUI-AIMA](https://github.com/HeimingX/GUI-AIMA) source on `PYTHONPATH` (the scripts handle this automatically).

### Training

Two-stage pipeline: SFT warm-up → GRPO reinforcement.

**Stage 1 — SFT** (teacher-forcing with GT coordinates):

```bash
bash scripts/train_sft.sh          # default: 2 rounds
bash scripts/train_sft.sh 3        # override max_rounds
```

**Stage 2 — GRPO** (policy gradient with sampled trajectories):

```bash
bash scripts/train_grpo.sh                           # from base model
bash scripts/train_grpo.sh /path/to/sft/checkpoint   # from SFT checkpoint
```

All scripts use `BASE_DIR` at the top for path configuration. Modify it for your environment.

### Evaluation

**Single configuration:**

```bash
# Standard eval (GUI-AIMA inference pipeline)
bash scripts/eval_standard.sh /path/to/model

# Aligned eval (matches training: attention extraction → crop → predict)
bash scripts/eval_aligned.sh /path/to/model

# Override parameters via environment variables
ROUNDS=3 PRED=argmax bash scripts/eval_aligned.sh /path/to/model
```

**Full evaluation suite** (6 configs: baselines + ablations):

```bash
bash scripts/eval_all.sh /path/to/model
MAX_SAMPLES=50 bash scripts/eval_all.sh /path/to/model   # quick test
```

### Tests

```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
GUI-Attention/
├── src/gui_attention/              # Core package (pip install -e .)
│   ├── train.py                    # Training entry point (SFT + GRPO)
│   ├── constants.py                # Precision levels, placeholder tokens
│   ├── attention.py                # Attention extraction (QK-recompute)
│   ├── sampling.py                 # Sample / argmax / region prediction
│   ├── crop.py                     # Image crop & coordinate helpers
│   └── builder.py                  # Multi-round conversation tokenizer
├── eval/                           # Evaluation scripts
│   ├── eval_screenspot_pro.py      # Standard eval (GUI-AIMA pipeline)
│   └── eval_screenspot_pro_aligned.py  # Aligned eval (our pipeline)
├── scripts/                        # Shell scripts
│   ├── train_sft.sh / train_grpo.sh
│   ├── eval_standard.sh / eval_aligned.sh / eval_all.sh
│   └── convert_screenspot_to_gta.py
└── tests/                          # Unit tests (28 tests)
```

## 📊 Results

| Method | Backbone | ScreenSpot-Pro | Tokens | Latency |
|--------|----------|----------------|--------|---------|
| GUI-AIMA (2-step) | Qwen2.5-VL-3B | 59.6% | ~18,800 | 2x |
| RegionFocus | Qwen2.5-VL-72B | 61.6% | ~75,000 | 3-5x |
| **GUI-Attention** | Qwen2.5-VL-3B | **TBD** | **~5,000** | **1x** |

## 🔗 Related Projects

- [GUI-AIMA](https://github.com/HeimingX/GUI-AIMA) — Attention-Informed Multi-grained Anchor for GUI grounding
- [RegionFocus](https://github.com/tiangeluo/RegionFocus) — Visual test-time scaling for GUI grounding
- [CoACT-1](https://github.com/SalesforceAIResearch/CoAct-1) — Collaborative Agent framework

## 📄 Citation

```bibtex
@inproceedings{fu2026guiattention,
  title={From Efficient Grounding to Efficient Agent: Foveated Visual Attention for GUI Understanding},
  author={Fu, Zichuan and others},
  booktitle={NeurIPS},
  year={2026}
}
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
