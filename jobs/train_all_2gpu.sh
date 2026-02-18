#!/bin/bash
#SBATCH --job-name=train_all
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_all_2gpu_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_all_2gpu_err_%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

DATA_ROOT=/home/zichuanfu2/data/GUI-Actor

# All 6 datasets (comma-separated)
DATA_PATHS="${DATA_ROOT}/guiact_bbox.json,${DATA_ROOT}/guienv_bbox.json,${DATA_ROOT}/amex_bbox.json,${DATA_ROOT}/androidcontrol_bbox.json,${DATA_ROOT}/wave_ui_bbox.json,${DATA_ROOT}/uground_bbox.json"
IMAGE_FOLDERS="${DATA_ROOT}/GUIAct/web_imgs,${DATA_ROOT}/GUIEnv/guienvs/images,${DATA_ROOT}/AMEX/screenshots,${DATA_ROOT}/AndroidControl/tfrecord/images,${DATA_ROOT}/Wave-UI/images_fixed,${DATA_ROOT}/Uground/images"

# 2 GPUs × bs=1 × ga=4 = effective batch 8
# ~1M samples / 8 = ~125K optimizer steps (1 epoch)
CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 \
    src/gui_attention/train.py \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path "$DATA_PATHS" \
    --image_folder "$IMAGE_FOLDERS" \
    --output_dir /home/zichuanfu2/results/ours_v4_all_6ds \
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
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
