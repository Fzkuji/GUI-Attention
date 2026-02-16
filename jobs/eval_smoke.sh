#!/bin/bash
#SBATCH --job-name=eval_fov
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_smoke_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_smoke_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate adasparse

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1

DATA=/home/zichuanfu2/data/ScreenSpot-Pro
SAVE_BASE=/home/zichuanfu2/results/eval_smoke

# 1) Baseline: Qwen2.5-VL-3B (no SFT), single-round at L1 (original resolution)
echo "====== Baseline: Qwen2.5-VL-3B, single-round L1 ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/baseline_L1" \
    --rounds 1 \
    --initial_level 1 \
    --max_samples 20

# 2) SFT checkpoint-500, single-round at L1
echo "====== SFT-500: single-round L1 ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path /home/zichuanfu2/results/sft_foveated_smoke/checkpoint-500 \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/sft500_L1" \
    --rounds 1 \
    --initial_level 1 \
    --max_samples 20

# 3) SFT checkpoint-500, multi-round foveation from L0
echo "====== SFT-500: multi-round foveation L0->L2 ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path /home/zichuanfu2/results/sft_foveated_smoke/checkpoint-500 \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/sft500_fov" \
    --rounds 5 \
    --initial_level 0 \
    --crop_ratio 0.3 \
    --max_samples 20

echo "====== All evaluations done ======"
