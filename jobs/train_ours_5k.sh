#!/bin/bash
#SBATCH --job-name=ours_5k
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_ours_5k_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_ours_5k_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1

# Our v4: LoRA + ActionHead + Saccade, on 5000 GUIAct samples
# bs=1, ga=4 → 1250 optimizer steps per epoch, train 1 epoch
CUDA_VISIBLE_DEVICES="0" python3 -m gui_attention.train \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path /home/zichuanfu2/data/GUI-Actor/guiact_5k_seed42.json \
    --image_folder /home/zichuanfu2/data/GUI-Actor/GUIAct/web_imgs \
    --output_dir /home/zichuanfu2/results/ours_v4_5k \
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
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
