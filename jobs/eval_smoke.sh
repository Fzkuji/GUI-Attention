#!/bin/bash
#SBATCH --job-name=eval_fov
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_smoke_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_smoke_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/zichuanfu2/GUI-AIMA/src:${PYTHONPATH}"

DATA=/home/zichuanfu2/data/ScreenSpot-Pro
SAVE_BASE=/home/zichuanfu2/results/eval_smoke
BASE_MODEL=/home/zichuanfu2/models/GUI-AIMA-3B
SFT_CKPT=/home/zichuanfu2/results/sft_foveated_smoke/checkpoint-500

# 1) Baseline: GUI-AIMA-3B, single-round at L1 (original resolution)
echo "====== Baseline: GUI-AIMA-3B, single-round L1 ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/baseline_L1" \
    --rounds 1 \
    --initial_level 1 \
    --query_weighting query_1 \
    --max_samples 20

# 2) SFT checkpoint-500, single-round at L1
echo "====== SFT-500: single-round L1 ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "$SFT_CKPT" \
    --base_model_path "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/sft500_L1" \
    --rounds 1 \
    --initial_level 1 \
    --query_weighting query_1 \
    --max_samples 20

# 3) SFT checkpoint-500, multi-round foveation from L0
echo "====== SFT-500: multi-round foveation L0->L2 ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "$SFT_CKPT" \
    --base_model_path "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/sft500_fov" \
    --rounds 5 \
    --initial_level 0 \
    --crop_ratio 0.3 \
    --query_weighting query_1 \
    --max_samples 20

echo "====== All evaluations done ======"
