# Related Work: GUI Grounding

## 竞品对比

### GUI-Actor (Microsoft, 2024)
- **模型**: Qwen2.5-VL-3B + pointer head (attention-based)
- **方法**: 单轮，全分辨率输入 (~1M pixels, 646 tokens)
- **ScreenSpot-Pro**: 42.2% (3B), 我们复现 43.20%
- **特点**: pointer head 用 cross-attention (visual tokens × anchor token) 预测点击位置
- **代码**: https://github.com/microsoft/GUI-Actor

### GUI-AIMA (2024)
- **模型**: Qwen2.5-VL-3B + ANCHOR token + attention head
- **方法**: 两阶段（低分辨率定位 → 高分辨率裁剪），但两阶段独立推理（不是端到端）
- **ScreenSpot-Pro**: 59.6% (3B)
- **ScreenSpot v2**: 91.5% (3B)
- **特点**: KL divergence loss, Gaussian soft labels
- **代码**: https://github.com/sjz5202/GUI-AIMA
- **HF**: smz8599/GUI-AIMA-3B

### GUI-ARP (arXiv 2509.15532)
- **方法**: GRPO + ASC (Adaptive Spatial Calibration)
- **ScreenSpot-Pro**: 60.8% (7B)

### RegionFocus (2025)
- **方法**: crop selection + refocusing，应用于 UI-TARS / Qwen2.5-VL
- **ScreenSpot-Pro**: 61.6% — 当前 SOTA

### LASER + GTA1-7B (2025)
- **方法**: DPO-tuned grounding
- **ScreenSpot-Pro**: 55.7% (7B)

### Phi-Ground (Microsoft, 2025)
- **方法**: Agent-oriented GUI grounding
- **ScreenSpot-Pro**: 55.0% (agent setting)

### GUI-G 2 (2025)
- **方法**: Decomposed grounding pipeline
- **ScreenSpot-Pro**: 47.5-48.7%

---

## GroundCUA / GroundNext (ServiceNow, arXiv 2511.07332)

**"Grounding Computer Use Agents on Human Demonstrations"**

### 核心贡献
1. **GroundCUA 数据集**: ~56K screenshots, 3.56M UI elements, ~700K instruction-element pairs
   - 来自 87 个真实桌面应用的**人类操作录制**
   - 每个 UI 元素标注: bounding box, 文本标签 (UI text / OCR), 类别标签
   - 从人类 demo 自动生成多样化自然语言指令
2. **GroundNext 模型**: 3B 和 7B 参数的多模态 grounding 模型
   - 输入: screenshot + text instruction → 输出: 目标 UI 元素坐标
   - SFT 训练，不依赖 RL
3. **SFT vs RL 分析**: 证明高质量密集标注数据可以替代大规模 RL

### 关键发现
- **跨域泛化**: 在桌面 demo 上训练，仍在 web/mobile 上表现良好
- **数据质量 > RL**: 精细标注的人类 demo 数据比 RL 训练更有效
- Dense grounding supervision from real usage (非合成数据)

### 与我们的关系
- **数据方向互补**: 他们强调数据质量和规模，我们强调推理效率（token efficiency）
- **可能的数据源**: GroundCUA 数据集可作为额外训练数据（56K real screenshots）
- **竞品**: 如果他们在 ScreenSpot-Pro 上有结果，需要对比
- **不同路线**: 他们走"更好数据"路线，我们走"更高效推理"路线（saccade foveation）

### 论文信息
- **arXiv**: 2511.07332
- **机构**: ServiceNow
- **数据**: https://huggingface.co/datasets/ServiceNow/GroundCUA
- **代码**: https://github.com/ServiceNow/GroundCUA

---

## Benchmark 汇总 (ScreenSpot-Pro)

| Model | Size | Method | ScreenSpot-Pro |
|-------|------|--------|---------------|
| Qwen2-VL-7B | 7B | Generalist | <2% |
| GPT-4o | - | Generalist | <2% |
| AriaUI | - | Specialist | 11.3% |
| UGround-7B | 7B | Specialist | 16.5% |
| OS-Atlas-7B | 7B | Specialist | 18.9% |
| **Ours (v15 ckpt-500)** | **3B** | **Saccade** | **18.34%** |
| GUI-Actor-3B | 3B | Pointer head | 42.2% |
| GUI-G 2 | 7B | Decomposed | 47.5-48.7% |
| Phi-Ground | - | Agent | 55.0% |
| LASER+GTA1 | 7B | DPO | 55.7% |
| GUI-AIMA-3B | 3B | Two-stage | 59.6% |
| GUI-ARP | 7B | GRPO+ASC | 60.8% |
| RegionFocus | 7B | Crop+refocus | 61.6% |

**注**: 我们的 18.34% 是 500 步 checkpoint，远未收敛。
