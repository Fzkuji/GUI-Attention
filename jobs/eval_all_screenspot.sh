#!/bin/bash
#SBATCH --job-name=eval_ss_all
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_ss_all_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_ss_all_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

CHECKPOINT=${1:-"/home/zichuanfu2/results/ours_v4_5k/final"}
BASE_MODEL=${2:-"/home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct"}
ROUNDS=${3:-3}
CROP_RATIO=${4:-0.3}
DEVICE=${5:-"cuda:0"}

echo "=== Evaluating checkpoint: $CHECKPOINT ==="
echo "  rounds=$ROUNDS  crop_ratio=$CROP_RATIO"
echo ""

# 1. ScreenSpot-Pro (local data)
echo ">>> ScreenSpot-Pro"
python eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --data_path /home/zichuanfu2/data/ScreenSpot-Pro \
    --rounds "$ROUNDS" \
    --crop_ratio "$CROP_RATIO" \
    --device "$DEVICE"

echo ""

# 2. ScreenSpot v1 (HuggingFace: rootsautomation/ScreenSpot)
echo ">>> ScreenSpot v1"
python eval/eval_screenspot.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --crop_ratio "$CROP_RATIO" \
    --device "$DEVICE"

echo ""

# 3. ScreenSpot-v2 (HuggingFace: HongxinLi/ScreenSpot_v2)
echo ">>> ScreenSpot-v2"
python eval/eval_screenspot_v2.py \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --crop_ratio "$CROP_RATIO" \
    --device "$DEVICE"

echo ""
echo "=== All evaluations done ==="
