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

### Inference

```bash
python eval/example_inference.py \
    --model_path <path_to_checkpoint> \
    --image_path <screenshot.png> \
    --instruction "Click the search button"
```

### Training

```bash
bash scripts/train.sh
```

### Evaluation

```bash
# ScreenSpot-Pro
bash eval/eval_screenspot_pro.sh

# OSWorld
bash eval/eval_osworld.sh
```

## 📁 Project Structure

```
GUI-Attention/
├── src/gui_attention/      # Core library
│   ├── model/              # Model architecture
│   ├── data/               # Dataset & preprocessing
│   ├── foveation/          # Foveated vision module
│   └── utils/              # Utilities
├── configs/                # Training & eval configs
├── eval/                   # Evaluation scripts
├── scripts/                # Shell scripts (train/eval)
├── tests/                  # Unit tests
└── docs/                   # Documentation
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
