#!/bin/bash
#SBATCH --job-name=sft_fov
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/sft_smoke_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/sft_smoke_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate adasparse

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1

# Single GPU SFT smoke test
# Use Qwen2.5-VL-3B-Instruct as base model
# 500 steps should be enough to see if loss decreases
CUDA_VISIBLE_DEVICES="0" python3 -m gui_attention.train \
    --training_mode sft \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path /home/zichuanfu2/data/GUI-Actor/guiact_bbox.json \
    --image_folder /home/zichuanfu2/data/GUI-Actor/GUIAct/web_imgs \
    --output_dir /home/zichuanfu2/results/sft_foveated_smoke \
    --min_pixels 3136 \
    --max_pixels 250000 \
    --max_rounds 3 \
    --initial_level 0 \
    --crop_ratio 0.3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --max_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
