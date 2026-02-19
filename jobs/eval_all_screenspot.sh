#!/bin/bash
#SBATCH --job-name=eval_ss_all
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_ss_all_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_ss_all_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00

# Evaluate on all 3 ScreenSpot benchmarks (Pro, v1, v2)
# Usage:
#   CHECKPOINT=/path/to/ckpt sbatch jobs/eval_all_screenspot.sh
#   CHECKPOINT=/path/to/ckpt DEVICE=cuda:1 sbatch jobs/eval_all_screenspot.sh

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

CHECKPOINT=${CHECKPOINT:-"/home/zichuanfu2/results/ours_v5_guiact_aligned/final"}
BASE_MODEL=${BASE_MODEL:-"/home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct"}
ROUNDS=${ROUNDS:-3}
CROP_RATIO=${CROP_RATIO:-0.3}
DEVICE=${DEVICE:-"cuda:0"}

echo "=== Evaluating checkpoint: $CHECKPOINT ==="
echo "  base_model=$BASE_MODEL"
echo "  rounds=$ROUNDS  crop_ratio=$CROP_RATIO  device=$DEVICE"
echo ""

# 1. ScreenSpot-Pro (local data, ~1.6K samples)
echo ">>> ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES="${DEVICE##*:}" python eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --data_path /home/zichuanfu2/data/ScreenSpot-Pro \
    --rounds "$ROUNDS" \
    --crop_ratio "$CROP_RATIO" \
    --device "cuda:0"

echo ""

# 2. ScreenSpot v1 (HuggingFace: rootsautomation/ScreenSpot, 1272 samples)
echo ">>> ScreenSpot v1"
CUDA_VISIBLE_DEVICES="${DEVICE##*:}" python eval/eval_screenspot.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --crop_ratio "$CROP_RATIO" \
    --device "cuda:0"

echo ""

# 3. ScreenSpot-v2 (HuggingFace: HongxinLi/ScreenSpot_v2, 1272 samples)
echo ">>> ScreenSpot-v2"
CUDA_VISIBLE_DEVICES="${DEVICE##*:}" python eval/eval_screenspot_v2.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --crop_ratio "$CROP_RATIO" \
    --device "cuda:0"

echo ""
echo "=== All evaluations done ==="
