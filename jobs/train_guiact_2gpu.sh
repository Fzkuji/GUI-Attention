#!/bin/bash
#SBATCH --job-name=guiact_2gpu
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_guiact_2gpu_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_guiact_2gpu_err_%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

# Full GUIAct (~42K samples), 2-GPU data parallel, 1 epoch
# 2 GPUs × bs=1 × ga=2 = effective batch 4
# ~42K / 4 = ~10.5K optimizer steps
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 \
    src/gui_attention/train.py \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path /home/zichuanfu2/data/GUI-Actor/guiact_bbox.json \
    --image_folder /home/zichuanfu2/data/GUI-Actor/GUIAct/web_imgs \
    --output_dir /home/zichuanfu2/results/ours_v4_guiact_full \
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
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
