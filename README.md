# GUI-Attention

基于 `GUI-Actor-3B-Qwen2.5-VL` 的多轮 GUI grounding 系统。

当前主线不是固定方框裁剪，而是：

- 第 0 轮：低分辨全图 overview
- 后续轮次：根据 `LookHead` attention 生成 `top-1 basin proposal bbox`
- 把该 bbox 裁出后统一 resize 到固定视觉预算
- 模型生成 `<look_pad>` / `<pointer_pad>` 决定继续看还是点击
- `ClickHead` 在高分辨证据上做最终点击

## 1. 当前状态

当前版本已经完成：

- 双头结构：`LookHead + ClickHead`
- `LookHead` / `ClickHead` 都从 GUI-Actor 原始 `pointer_head` 复制初始化
- proposal-based look crop
- free-reasoning SFT
- aligned GRPO（proposal-based look crop）
- 训练集可视化 / ScreenSpot-Pro 可视化
- GroundCUA 数据接入和转换

当前最重要的结论：

- `ScreenSpot-v2` 已经接近 GUI-Actor baseline
- `ScreenSpot-Pro` 仍明显偏低
- 当前主要问题更像是 **探索 / stop policy**，不是基础点击能力完全坏掉

### 当前主要版本 / 输出目录

- `GUI-Actor-3B-Qwen2.5-VL`：原始 baseline
- `results/ours_v15_dual`：当前最核心的 SFT 主线
- `results/ours_grpo`：当前 proposal-aligned GRPO 主线
- `results/ours_v15_dual_groundcua`：准备做的高分辨桌面二阶段 SFT 线

当前最常被引用的 SFT checkpoint：

- `checkpoint-500`
- `checkpoint-1000`
- `checkpoint-1500`

## 2. 目录结构

```text
src/gui_attention/
  inputs/
    builder.py
    crop.py
    labels.py
  modeling/
    attention.py
    dual_head.py
    model.py
  runtime/
    foveation.py
    inference.py
    proposals.py
    reasoning.py
  training/
    grpo.py
    sft.py

eval/
  eval_screenspot.py
  eval_train_hit.py
  visualize_saccade.py

jobs/
  train.sh
  train_grpo.sh
```

## 3. 当前默认训练配置（SFT）

当前 SFT 主设定：

- base model：`GUI-Actor-3B-Qwen2.5-VL`
- low-res 全图：`1001600` pixels
- `look` 裁图：`top1_basin_proposal -> resize_to 1001600 pixels`
- fallback square crop：`308x308 x3 -> 924x924`
- `max_saccade_rounds = 6`
- dual tokens：开启
- free reasoning SFT：开启
- assistant EOS：开启

当前 loss 权重：

- `lm_loss_weight = 0.5`
- `look_loss_weight = 1.0`
- `click_loss_weight = 4.0`

当前 LM token 权重：

- 普通 reasoning token：`0.1`
- 格式 / EOS：`0.5`
- `<look_pad>`：`1.0`
- `<pointer_pad>`：`2.0`

## 4. 安装

```bash
git clone https://github.com/Fzkuji/GUI-Attention.git
cd GUI-Attention
pip install -e .
```

## 5. 数据准备

### 5.1 常规数据

`jobs/train.sh` 会自动检测工作区里的这些 JSON：

- `guiact_bbox.json`
- `androidcontrol_bbox.json`
- `wave_ui_bbox.json`
- `groundcua_bbox.json`
- `uground_bbox.json`
- `gta/gta_data/gta_data_wo_web.json`

### 5.2 GroundCUA 转换

如果已经把原始 `GroundCUA` 下载到：

```bash
/mnt/data/zichuanfu/GUI-Attention-Workspace/data/GroundCUA
```

则转换命令：

```bash
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention

python scripts/convert_groundcua.py \
  --groundcua_dir /mnt/data/zichuanfu/GUI-Attention-Workspace/data/GroundCUA \
  --output_json /mnt/data/zichuanfu/GUI-Attention-Workspace/data/groundcua_bbox.json \
  --max_per_image 0 \
  --workers 16
```

当前完整转换结果约：

- `51174` 张图片
- `3197522` 条训练样本

建议：

- JSON 先全量转换保留
- 训练时先从 `60000` 或 `120000` 做 quick test

### 5.3 从 HF cache 复制 GroundCUA 的注意事项

如果从 HuggingFace cache 复制到项目数据目录，必须使用：

```bash
cp -aL <snapshot>/. /mnt/data/zichuanfu/GUI-Attention-Workspace/data/GroundCUA/
```

不要用 `cp -a`，否则会保留符号链接，导致 `images/*.png` 变成 broken symlink。

## 6. 训练

### 6.1 主线 SFT 训练

```bash
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention

bash jobs/train.sh --num_gpus 8
```

### 6.2 真正 resume（恢复训练状态）

```bash
bash jobs/train.sh \
  --num_gpus 8 \
  --resume_ckpt /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500
```

### 6.3 二阶段初始化继续训（不恢复旧训练状态）

这个场景要用 `--init_ckpt`，不要用 `--resume_ckpt`。

例如只用 `GroundCUA` 做二阶段 SFT：

```bash
bash jobs/train.sh \
  --num_gpus 7 \
  --init_ckpt /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500 \
  --output_dir /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual_groundcua \
  --data_paths /mnt/data/zichuanfu/GUI-Attention-Workspace/data/groundcua_bbox.json \
  --image_folders /mnt/data/zichuanfu/GUI-Attention-Workspace/data/GroundCUA \
  --max_samples_per_dataset 0
```

### 6.4 显式指定单个 / 多个数据集

```bash
bash jobs/train.sh \
  --num_gpus 8 \
  --data_paths /path/a.json,/path/b.json \
  --image_folders /path/images_a,/path/images_b \
  --max_samples_per_dataset 60000,60000
```

## 7. GRPO

当前 GRPO 已经和 proposal-based look crop 对齐。

第一版 GRPO 的设计原则是：

- `look` 时系统自动选择 `top-1 proposal bbox`
- RL 重点优化 `look / click`
- 不先让 RL 学 proposal 编号选择

命令：

```bash
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention

bash jobs/train_grpo.sh \
  --num_gpus 8 \
  --sft_ckpt /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500
```

可选参数：

```bash
bash jobs/train_grpo.sh \
  --num_gpus 8 \
  --sft_ckpt /path/to/checkpoint \
  --output_dir /path/to/output \
  --group_size 8 \
  --save_steps 50 \
  --reasoning_max_new_tokens 48 \
  --reward_round_penalty 0.02 \
  --reward_format 0.05 \
  --reward_malformed_penalty 0.05
```

## 8. 评估

### 8.1 ScreenSpot-Pro

```bash
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention

torchrun --nproc_per_node=8 eval/eval_screenspot.py \
  --dataset pro \
  --data_dir /mnt/data/zichuanfu/GUI-Attention-Workspace/data/ScreenSpot-Pro \
  --checkpoint /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500 \
  --base_model /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
  --rounds 6 \
  --use_dual_tokens \
  --no_adaptive_crop
```

### 8.2 ScreenSpot-v1

注意：`v1` 不需要 `--data_dir`。

```bash
torchrun --nproc_per_node=8 eval/eval_screenspot.py \
  --dataset v1 \
  --checkpoint /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500 \
  --base_model /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
  --rounds 6 \
  --use_dual_tokens \
  --no_adaptive_crop
```

### 8.3 ScreenSpot-v2

```bash
torchrun --nproc_per_node=8 eval/eval_screenspot.py \
  --dataset v2 \
  --data_dir /mnt/data/zichuanfu/GUI-Attention-Workspace/data/ScreenSpot-v2 \
  --checkpoint /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500 \
  --base_model /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
  --rounds 6 \
  --use_dual_tokens \
  --no_adaptive_crop
```

### 8.4 GUI-Actor baseline 对照

```bash
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Actor

rm -f /tmp/gui_actor_pro/screenspot-Pro_all_preds_StandardResize.txt
rm -f /tmp/gui_actor_pro/screenspot-Pro_all_preds_StandardResize.json

PYTHONPATH=src:$PYTHONPATH HF_HUB_OFFLINE=1 python eval/screenSpot_pro.py \
  --model_type qwen25vl \
  --model_name_or_path /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
  --data_path /mnt/data/zichuanfu/GUI-Attention-Workspace/data/ScreenSpot-Pro \
  --save_path /tmp/gui_actor_pro
```

## 9. 可视化

### 9.1 ScreenSpot-Pro

```bash
cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention

PYTHONPATH=src python eval/visualize_saccade.py \
  --checkpoint /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500 \
  --base_model /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
  --dataset pro \
  --data_dir /mnt/data/zichuanfu/GUI-Attention-Workspace/data/ScreenSpot-Pro \
  --sample_index 42 \
  --num_samples 10 \
  --output viz_sft_batch.png \
  --proposal_topk 4 \
  --device cuda:0
```

### 9.2 训练集样本

```bash
PYTHONPATH=src python eval/visualize_saccade.py \
  --checkpoint /mnt/data/zichuanfu/GUI-Attention-Workspace/results/ours_v15_dual/checkpoint-1500 \
  --base_model /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
  --dataset train \
  --train_json /mnt/data/zichuanfu/GUI-Attention-Workspace/data/groundcua_bbox.json \
  --image_root /mnt/data/zichuanfu/GUI-Attention-Workspace/data/GroundCUA \
  --sample_index 0 \
  --num_samples 10 \
  --output viz_train_groundcua.png \
  --device cuda:0
```

## 10. 当前结果判断

### 已确认的对照

- 原始 GUI-Actor 本地 `ScreenSpot-Pro`：约 `43.20`
- 当前 proposal-based 多轮模型：
  - `ScreenSpot-v2`：约 `88.43`
  - `ScreenSpot-Pro`：仍明显偏低

这说明：

- 当前基础单轮 / 简单场景能力已经基本保住
- 真正的短板在高分辨桌面图上的探索与停留策略

## 11. 当前最重要的问题

1. `ScreenSpot-Pro` 仍远低于 GUI-Actor baseline
2. 模型倾向于：
   - `round0 look`
   - `round1 过早 click`
3. `LookHead` 当前更像候选区域提议器，而不是稳定的精定位器
4. GRPO 的格式稳定性还不够强

## 12. 当前推荐实验顺序

1. 先做 `GroundCUA` 单独二阶段 SFT
2. 再重新评 `ScreenSpot-Pro`
3. 再继续做 proposal-aligned GRPO
4. RL 第一版只优化 `look/click`，不要先优化 proposal 选择

更细一点：

1. 从 `results/ours_v15_dual/checkpoint-1500` 用 `--init_ckpt` 启动 `GroundCUA` 单独二阶段 SFT
2. 训练期间同步看：
   - `GroundCUA` 训练集可视化
   - `ScreenSpot-Pro` 可视化
3. 重点判断是否仍存在大量 `round0 look -> round1 click -> miss`
4. 再重新评 `ScreenSpot-Pro / ScreenSpot-v2`
5. 如果 `v2` 基本维持、但 `Pro` 仍差，再继续 proposal-aligned GRPO
