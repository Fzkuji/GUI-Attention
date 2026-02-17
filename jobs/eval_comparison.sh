#!/bin/bash
#SBATCH --job-name=eval_cmp
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_comparison_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_comparison_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn
export PYTHONUNBUFFERED=1

DATA=/home/zichuanfu2/data/ScreenSpot-Pro
BASE_MODEL=/home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct

# ===== 1. Eval GUI-Actor-5k (their method, their eval) =====
echo "====== Eval: GUI-Actor 5k SFT ======"
cd /home/zichuanfu2/GUI-Actor

# Find latest GUI-Actor SFT checkpoint
GUIACTOR_CKPT=$(ls -d /home/zichuanfu2/results/guiactor_5k_sft/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$GUIACTOR_CKPT" ]; then
    GUIACTOR_CKPT=/home/zichuanfu2/results/guiactor_5k_sft
fi
echo "GUI-Actor checkpoint: $GUIACTOR_CKPT"

CUDA_VISIBLE_DEVICES="0" python3 eval/screenSpot_pro.py \
    --model_type qwen25vl \
    --model_name_or_path "$GUIACTOR_CKPT" \
    --data_path "$DATA" \
    --save_path /home/zichuanfu2/results/eval_guiactor_5k \
    --resize_to_pixels 5760000 \
    --topk 3

# ===== 1b. Time GUI-Actor inference =====
echo "====== Timing: GUI-Actor ======"
CUDA_VISIBLE_DEVICES="0" python3 /home/zichuanfu2/GUI-Attention/eval/time_gui_actor.py \
    --model_name_or_path "$GUIACTOR_CKPT" \
    --data_path "$DATA" \
    --save_path /home/zichuanfu2/results/eval_guiactor_5k \
    --resize_to_pixels 5760000

# ===== 2. Eval Ours v4: single-round =====
echo "====== Eval: Ours v4 single-round ======"
cd /home/zichuanfu2/GUI-Attention

# Find latest our checkpoint
OURS_CKPT=$(ls -d /home/zichuanfu2/results/ours_v4_5k/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$OURS_CKPT" ]; then
    OURS_CKPT=/home/zichuanfu2/results/ours_v4_5k/final
fi
echo "Our checkpoint: $OURS_CKPT"

CUDA_VISIBLE_DEVICES="0" python3 eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$OURS_CKPT" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path /home/zichuanfu2/results/eval_ours_5k/single_round \
    --rounds 1

# ===== 3. Eval Ours v4: saccade 3 rounds =====
echo "====== Eval: Ours v4 saccade 3 rounds ======"
CUDA_VISIBLE_DEVICES="0" python3 eval/eval_screenspot_pro_aligned.py \
    --checkpoint "$OURS_CKPT" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA" \
    --save_path /home/zichuanfu2/results/eval_ours_5k/saccade_3rounds \
    --rounds 3 \
    --crop_ratio 0.3

echo "====== All evaluations done ======"
