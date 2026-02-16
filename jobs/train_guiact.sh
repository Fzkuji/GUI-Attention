#!/bin/bash
#SBATCH --job-name=sft_v4
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_guiact_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_guiact_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1

# Full GUIAct training: LoRA + ActionHead, 2-round saccade teacher forcing
# 42K samples, bs=1, ga=4 → ~10.5K steps/epoch, train 5000 steps
CUDA_VISIBLE_DEVICES="0" python3 -m gui_attention.train \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path /home/zichuanfu2/data/GUI-Actor/guiact_bbox.json \
    --image_folder /home/zichuanfu2/data/GUI-Actor/GUIAct/web_imgs \
    --output_dir /home/zichuanfu2/results/sft_v4_guiact \
    --min_pixels 3136 \
    --low_res_max_pixels 1003520 \
    --high_res_max_pixels 5720064 \
    --crop_ratio 0.3 \
    --crop_jitter 0.05 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules "q_proj,v_proj" \
    --action_head_lr 1e-4 \
    --lora_lr 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --max_steps 5000 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
