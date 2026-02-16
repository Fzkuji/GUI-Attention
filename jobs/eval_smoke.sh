#!/bin/bash
#SBATCH --job-name=eval_saccade
#SBATCH --output=/mnt/data/zichuanfu/GUI-Attention/logs/eval_smoke_%j.txt
#SBATCH --error=/mnt/data/zichuanfu/GUI-Attention/logs/eval_smoke_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /mnt/data/zichuanfu/GUI-Attention
export PYTHONUNBUFFERED=1
# No PYTHONPATH to GUI-AIMA needed (self-contained)

DATA=/mnt/data/zichuanfu/data/ScreenSpot-Pro
SAVE_BASE=/mnt/data/zichuanfu/results/eval_smoke
BASE_MODEL=/mnt/data/zichuanfu/models/Qwen2.5-VL-3B-Instruct
SFT_CKPT=/mnt/data/zichuanfu/results/sft_saccade_smoke/checkpoint-500

# 1) SFT checkpoint, single-round (no saccade, just action head on low-res)
echo "====== SFT-500: single-round ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$SFT_CKPT" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/sft500_single" \
    --rounds 1 \
    --max_samples 20

# 2) SFT checkpoint, multi-round saccade (3 rounds)
echo "====== SFT-500: saccade 3 rounds ======"
python3 eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$SFT_CKPT" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/sft500_saccade" \
    --rounds 3 \
    --crop_ratio 0.3 \
    --max_samples 20

echo "====== All evaluations done ======"
