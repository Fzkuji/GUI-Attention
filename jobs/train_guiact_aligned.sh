#!/bin/bash
#SBATCH --job-name=gui_attn_guiact
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_guiact_aligned_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_guiact_aligned_err_%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00

# GUIAct-only training with GUI-Actor-aligned settings (quick validation)
# 2 GPUs (0,1 NVLink pair) × bs=1 × ga=2 = effective batch 4
# ~42K / 4 = ~10.5K steps, estimated ~5.5h

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

DATA_ROOT=/home/zichuanfu2/data/GUI-Actor

CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 \
    src/gui_attention/train.py \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path "${DATA_ROOT}/guiact_bbox.json" \
    --image_folder "${DATA_ROOT}/GUIAct/web_imgs" \
    --output_dir /home/zichuanfu2/results/ours_v5_guiact_aligned \
    --min_pixels 3136 \
    --low_res_max_pixels 1003520 \
    --high_res_max_pixels 5720064 \
    --crop_ratio 0.3 \
    --crop_jitter 0.05 \
    --max_saccade_rounds 3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules "q_proj,v_proj" \
    --action_head_lr 1e-4 \
    --lora_lr 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
