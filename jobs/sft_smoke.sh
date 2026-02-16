#!/bin/bash
#SBATCH --job-name=sft_saccade
#SBATCH --output=/mnt/data/zichuanfu/GUI-Attention/logs/sft_smoke_%j.txt
#SBATCH --error=/mnt/data/zichuanfu/GUI-Attention/logs/sft_smoke_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /mnt/data/zichuanfu/GUI-Attention
export PYTHONUNBUFFERED=1
# No PYTHONPATH to GUI-AIMA needed (self-contained)

# Single GPU SFT smoke test: LoRA + ActionHead, 2-round saccade teacher forcing
# Uses Qwen2.5-VL-3B-Instruct as base model (no GUI-AIMA dependency)
# 500 steps should be enough to see if loss decreases
CUDA_VISIBLE_DEVICES="0" python3 -m gui_attention.train \
    --model_name_or_path /mnt/data/zichuanfu/models/Qwen2.5-VL-3B-Instruct \
    --data_path /mnt/data/zichuanfu/data/GUI-Actor/guiact_bbox.json \
    --image_folder /mnt/data/zichuanfu/data/GUI-Actor/GUIAct/web_imgs \
    --output_dir /mnt/data/zichuanfu/results/sft_saccade_smoke \
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
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --max_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
