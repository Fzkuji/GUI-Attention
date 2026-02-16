#!/bin/bash
#SBATCH --job-name=eval_ours
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_ours_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_ours_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1

DATA=/home/zichuanfu2/data/ScreenSpot-Pro
SAVE_BASE=/home/zichuanfu2/results/eval_ours_v4
BASE_MODEL=/home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct
SFT_CKPT=/home/zichuanfu2/results/sft_v4_guiact/checkpoint-5000

# 1) Single-round eval (no saccade, just action head on low-res)
echo "====== Ours v4: single-round (1581 examples) ======"
CUDA_VISIBLE_DEVICES="0" python3 eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$SFT_CKPT" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/single_round" \
    --rounds 1

# 2) Multi-round saccade (3 rounds)
echo "====== Ours v4: saccade 3 rounds (1581 examples) ======"
CUDA_VISIBLE_DEVICES="0" python3 eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$SFT_CKPT" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path "$SAVE_BASE/saccade_3rounds" \
    --rounds 3 \
    --crop_ratio 0.3

echo "====== All evaluations done ======"
