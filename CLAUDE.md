# GUI-Attention 内部协作记录

> 最后更新：2026-03-12

这份文件记录当前代码库内部最重要的工程约定、已确认结论、常见坑和推荐实验路径。

## 1. 当前主线设计

当前系统已经不是旧版固定 crop。

当前 `look` 逻辑：

1. 从 `LookHead` attention map 生成 basin proposals
2. 取 `top-1 proposal bbox`
3. 按该 bbox 裁图
4. 把裁图 resize 到固定视觉预算（默认 `1001600` pixels）

当前 `click` 逻辑：

- 最终 `ClickHead` 只在高分辨 crop 证据上做精点击
- 低分辨全图更偏向用于探索 / stop 决策

## 2. 代码入口

### SFT

- 真实实现：`src/gui_attention/training/sft.py`
- 兼容入口：`src/gui_attention/train.py`
- 启动脚本：`jobs/train.sh`

### GRPO

- 真实实现：`src/gui_attention/training/grpo.py`
- 兼容入口：`src/gui_attention/grpo.py`
- 启动脚本：`jobs/train_grpo.sh`

### 推理

- `src/gui_attention/runtime/inference.py`

### Proposal

- `src/gui_attention/runtime/proposals.py`

### 可视化

- `eval/visualize_saccade.py`

## 3. 当前 SFT 关键约定

### 3.1 Head 初始化

当前不是：
- `ClickHead` 从 GUI-Actor pointer head 初始化
- `LookHead` 随机初始化

而是：
- `LookHead` 和 `ClickHead` 都从 GUI-Actor `pointer_head` 复制初始化

这是当前主线。

### 3.2 当前 loss 配置

当前默认：

- `lm_loss_weight = 0.5`
- `look_loss_weight = 1.0`
- `click_loss_weight = 4.0`

LM token 权重：

- reasoning token：`0.1`
- format / EOS：`0.5`
- `<look_pad>`：`1.0`
- `<pointer_pad>`：`2.0`

### 3.3 日志口径

当前 SFT 训练日志里的：

- `loss`
- `lm`
- `look`
- `click`

是严格同口径的样本级平均分解，满足：

```text
loss = lm + look + click
```

不要再回到早期那种不同分母、不同窗口的混合打印。

### 3.4 overlap penalty

`look_overlap_loss` 已经删除。

原因：

- 虽然直觉上“别重复看老区域”合理
- 但实际会把模型的搜索行为推偏
- 人类搜索本来也允许 revisit

所以当前不要再启用这项，除非重新做非常轻量的版本并有明确验证。

## 4. 当前 GRPO 关键约定

### 4.1 当前 GRPO 对齐策略

GRPO 已对齐 proposal-based look crop：

- 不再用旧版“点 + 固定方框”
- `look` 时自动取 `top-1 proposal bbox`
- GRPO 第一版不让模型学 proposal index

### 4.2 当前 RL 主要学什么

当前推荐 RL 重点：

- `look` / `click`
- 什么时候继续看
- 什么时候停

而不是：

- proposal `#1/#2/#3` 的选择
- 精点击坐标本身

### 4.3 当前 GRPO 的已知问题

当前最明显的问题不是 reward 完全没信号，而是：

- `fmt` 不够高
- `parse_fail` 偏高

所以早期 GRPO 的瓶颈更多是“生成协议没学稳”，不是完全没有策略学习。

## 5. 训练 / 继续训练的区别

### `--resume_ckpt`

用途：

- 恢复模型权重
- 恢复 optimizer / scheduler / global_step
- 恢复旧训练状态

适用于：

- 同一实验真正断点续训

### `--init_ckpt`

用途：

- 只加载模型权重
- 不恢复训练状态

适用于：

- 新数据集二阶段 SFT
- 新实验初始化

一个常见错误是：

- 用 `--resume_ckpt checkpoint-1500` 去跑 `GroundCUA` 二阶段 SFT
- 然后日志显示 `Epoch 1 already completed, skipping`

这是因为恢复了旧训练状态。  
正确做法是用 `--init_ckpt`。

## 6. 数据集接入规则

### 6.1 自动检测

当前 `jobs/train.sh` / `jobs/train_grpo.sh` 会自动检测：

- `guiact_bbox.json`
- `androidcontrol_bbox.json`
- `wave_ui_bbox.json`
- `groundcua_bbox.json`
- `uground_bbox.json`
- `gta/gta_data/gta_data_wo_web.json`

### 6.2 GroundCUA 特殊点

`GroundCUA` 的图片路径要特别小心：

- HF cache snapshot 里很多是符号链接
- 从 cache 复制到项目目录时必须用 `cp -aL`

错误做法：

```bash
cp -a snapshot/. GroundCUA/
```

这会把链接本体复制过来，导致 `broken symbolic link`。

正确做法：

```bash
cp -aL snapshot/. GroundCUA/
```

### 6.3 GroundCUA 转换

当前转换脚本：

- `scripts/convert_groundcua.py`

功能：

- 建立图片索引
- 多线程解析
- 跳过坏图
- 输出统一训练 JSON

当前完整转换结果：

- `51175` annotation files
- `51174` images converted
- `3197522` samples written

## 7. 可视化约定

### 7.1 当前可视化能看什么

`eval/visualize_saccade.py` 当前已经支持：

- ScreenSpot-Pro 样本
- 训练 JSON 样本
- basin proposal overlay
- 真实推理 trace

### 7.2 文本和动作不一致

当前可视化里可能会看到：

- reasoning 文本像是在说“继续看”
- 但最终动作是 `click`

这不是可视化乱画，而是当前协议里：

- 真正控制流程的是最后的 action token / action span
- 前面的 natural language reasoning 只是解释文本

所以调试时要优先看：

- `action=look/click`
- `fmt`
- `final click`

不要只看自然语言句子本身。

## 8. 当前已确认的实验结论

### 8.1 Baseline

原始 GUI-Actor 本地 `ScreenSpot-Pro` 约：

- `All-avg hit_top1 = 43.20`

这个数字非常重要，因为它说明：

- 不是 benchmark 有问题
- 也不是 base model 完全不会这个任务

### 8.2 当前多轮系统

当前 proposal-based SFT：

- `ScreenSpot-v2` 已接近 GUI-Actor 论文数字
- `ScreenSpot-Pro` 仍明显偏低

这意味着：

- 简单场景下能力基本还在
- 真正掉的是 high-res desktop / tiny-target / search-heavy 场景

### 8.3 当前最重要的问题

当前最像主因的是：

1. 过早 click
2. 探索不足
3. 高分辨桌面图分布不匹配

不是简单的：

- `ClickHead` 完全不会点

## 9. Proposal 这条线的认识

当前对 proposal 的认识已经比较明确：

- raw attention 不能直接当最终 crop
- proposal 更适合做候选区域提议
- 当前 `top-1 proposal` 已经足够做第一版系统

如果以后继续升级 proposal，更合理的方向是：

- 先做候选区域排序
- 再做更细搜索

而不是一上来并行截很多框。

## 10. 当前最推荐的实验顺序

1. `GroundCUA` 单独二阶段 SFT
2. 重新测 `ScreenSpot-Pro`
3. 观察 premature click 是否缓解
4. 再继续 proposal-aligned GRPO

GRPO 第一版继续坚持：

- 系统取 top-1 proposal
- 模型只学 `look / click`

## 11. 常见坑

### 坑 1：`v1` / `v2` / `pro` 混用

- `ScreenSpot-Pro` 必须用 `--dataset pro`
- `v1` 不需要 `--data_dir`
- `v2` 不能把 `ScreenSpot-Pro` 目录当成 `data_dir`

### 坑 2：在错误目录运行 `torchrun`

如果当前目录是 `data/`，再跑：

```bash
torchrun --nproc_per_node=8 eval/eval_screenspot.py ...
```

相对脚本路径会找不到。要么回到仓库目录，要么用绝对脚本路径。

### 坑 3：单独数据集二阶段训练用错 `resume`

再次强调：

- `resume` = 续跑老实验
- `init` = 拿旧权重开新实验

### 坑 4：HF cache 复制保留了符号链接

这个已经反复出现过，之后不要再踩。

## 12. 当前工作区路径

远程服务器当前主工作区：

```text
/mnt/data/zichuanfu/GUI-Attention-Workspace
```

其中：

- repo：`/mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention`
- models：`/mnt/data/zichuanfu/GUI-Attention-Workspace/models`
- data：`/mnt/data/zichuanfu/GUI-Attention-Workspace/data`
- results：`/mnt/data/zichuanfu/GUI-Attention-Workspace/results`

本地仓库：

```text
/Users/fzkuji/Documents/GUI Agent/GUI-Attention
```
