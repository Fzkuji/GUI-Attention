# CLAUDE.md - GUI-Attention Design Notes

## Architecture: Saccade Foveation

Multi-round visual grounding:
1. **Round 0**: Low-res full image → coarse localization → predict click point
2. **Round 1+**: Fixed-size crop around predicted point → upsampled → precise localization

Key difference from GUI-AIMA two-stage: we train multi-round end-to-end (model sees low-res + crop in same context), while GUI-AIMA runs two independent inferences at eval time only.

## Resolution Design (Token Efficiency Analysis)

### Qwen2.5-VL Token Mechanics
- ViT patch_size=14, spatial_merge_size=2 → **merge_patch_size=28**
- 1 token = 28×28 pixels on the resized image
- `smart_resize` rounds to factor=28 multiples, then clamps to [min_pixels, max_pixels]
- **smart_resize does NOT upscale**: if image < max_pixels, it only rounds to nearest 28-multiple
- To upscale, must explicitly resize (PIL) before passing to processor

### Current Config: 168×168 crop ×4 upscale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `low_res_max_pixels` | 400,000 | Full image: 480 tokens (1080p), 64×68 px/tok |
| `crop_size` | 168×168 px | Fixed absolute size, 28-divisible |
| `crop_upscale` | ×4 → 672×672 | 576 tokens, 7.0×7.0 px/tok |
| **Total (1 crop)** | **1,056 tokens** | vs GUI-AIMA 3,267, GUI-Actor 2,691 |

### Why Fixed Crop Size (not percentage)
- **Percentage crop problem**: On high-res images (4K/5K), 30% crop = 1152×648 = 746K px → gets **downsampled** to 400K, losing the zoom benefit
- **Fixed crop**: Always small enough to guarantee upsampling, consistent precision across all resolutions
- 168×168 on 1080p = 15.6% coverage; on 4K = 7.8% → higher res images get proportionally smaller crop (acceptable since they have more pixels per element)

### Precision Comparison

| Method | Tokens | px/tok (crop) | 16×16 icon | 24×24 icon | 32×32 icon |
|--------|--------|---------------|------------|------------|------------|
| GUI-Actor (full) | 2,691 | 27.8 | 0.33 tok | 0.73 tok | 1.31 tok |
| GUI-AIMA 2-stage | 3,267 | 14.0 (crop) | 1.31 tok | 2.94 tok | 5.22 tok |
| **Ours (168² ×4)** | **1,056** | **7.0 (crop)** | **5.22 tok** | **11.76 tok** | **20.90 tok** |

### Integer Upscale Sweet Spots (for ablation)

All crop sizes are 28-divisible; ×N produces 28-divisible resize target.

| Crop | ×N | Resize | Crop Tokens | Total | px/tok | 10×10 icon |
|------|-----|--------|-------------|-------|--------|------------|
| 140×140 | ×4 | 560×560 | 400 | 880 | 7.0 | 2.0 tok |
| **168×168** | **×4** | **672×672** | **576** | **1,056** | **7.0** | **2.0 tok** |
| 196×196 | ×3 | 588×588 | 441 | 921 | 9.3 | 1.1 tok |
| 224×224 | ×3 | 672×672 | 576 | 1,056 | 9.3 | 1.1 tok |
| 252×252 | ×3 | 756×756 | 729 | 1,209 | 9.3 | 1.1 tok |
| 168×168 | ×3 | 504×504 | 324 | 804 | 9.3 | 1.1 tok |
| 196×196 | ×4 | 784×784 | 784 | 1,264 | 7.0 | 2.0 tok |

16:9 variants:

| Crop | ×N | Resize | Crop Tokens | Total | 1080p Coverage |
|------|-----|--------|-------------|-------|----------------|
| 196×112 | ×4 | 784×448 | 448 | 928 | 10%×10% |
| 252×140 | ×3 | 756×420 | 405 | 885 | 13%×13% |
| 224×112 | ×3 | 672×336 | 288 | 768 | 12%×10% |

### ScreenSpot-Pro Target Size Distribution
- Average bbox area = 0.07% of image area
- Most targets: **16~32 px** (small icons dominate, 62% are icons)
- On 1080p: Small category (14×14 ~ 28×28 px area) is the majority
- Our ×4 crop: 16×16 icon spans 5.2 tokens → excellent coverage

### Low-res Full Image Options (for ablation)

| max_pixels | Tokens (1080p) | px/tok | Notes |
|------------|----------------|--------|-------|
| 400,000 | 480 | 64×68 | Current choice, good coarse localization |
| 200,000 | 231 | 91×98 | More aggressive, needs strong pointer head |
| 100,000 | 120 | 128×135 | Very coarse, may hurt crop_hit |

## Training Pipeline

### Two-stage training (following GUI-Actor)
1. **Warmup**: Single-round, pointer head only (`warmup_rounds_step=500`)
2. **Multi-round**: All rounds with crop, LM loss + pointer loss

### Loss
- `total_loss = lm_loss_weight × LM_loss + pointer_loss_weight × pointer_loss`
- LM loss: CrossEntropy on assistant tokens (per-round, no double counting)
- Pointer loss: KL divergence on attention head output

### Key Training Params
- `action_head_lr=1e-4`, `backbone_lr=5e-5` (LoRA) or `5e-6` (full-param)
- `warmup_ratio=0.1` for saccade training
- `gradient_accumulation_steps=2`
- `gradient_checkpointing=true` (93GB→19GB per GPU)

## Server Info
- **Tencent**: 8× H20 95GB, `/mnt/data/zichuanfu/GUI-Attention-Workspace/`
- **AML (CityU)**: 4× A100 80GB, `zichuanfu2@144.214.209.213`
- **AutoDL**: RTX A800 80GB, `connect.bjb1.seetacloud.com:12172`
- Local model: `models/Qwen2.5-VL-3B-Instruct`
- GitHub: `Fzkuji/GUI-Attention`

## Baselines
- GUI-Actor-3B: 43.20% ScreenSpot-Pro (our eval), 58.5% train hit
- GUI-AIMA-3B: 59.6% ScreenSpot-Pro (paper), 47.1% soft / 58.6% zoom-in
- GUI-ARP-7B: 60.8% ScreenSpot-Pro
