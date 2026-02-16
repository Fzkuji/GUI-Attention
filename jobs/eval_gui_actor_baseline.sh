#!/bin/bash
#SBATCH --job-name=eval_guiactor
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/eval_guiactor_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/eval_guiactor_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Actor
export PYTHONUNBUFFERED=1

# Evaluate GUI-Actor-3B baseline on ScreenSpot-Pro (full 1581 examples)
CUDA_VISIBLE_DEVICES="0" python3 eval/screenSpot_pro.py \
    --model_type qwen25vl \
    --model_name_or_path /home/zichuanfu2/models/GUI-Actor-3B-Qwen2.5-VL \
    --data_path /home/zichuanfu2/data/ScreenSpot-Pro \
    --save_path /home/zichuanfu2/results/eval_gui_actor_3b \
    --resize_to_pixels 5760000 \
    --topk 3
