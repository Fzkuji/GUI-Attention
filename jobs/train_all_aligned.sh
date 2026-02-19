#!/bin/bash
#SBATCH --job-name=gui_attn_5ds
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_aligned_5ds_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_aligned_5ds_err_%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00

# Training aligned with GUI-Actor settings
# Key alignments: weight_decay=0.0, warmup_ratio=0.03, effective_batch=4
# Server A: GPU 0,1 (NVLink pair for best inter-GPU bandwidth)
# 2 GPUs × bs=1 × ga=2 = effective batch 4
#
# Resume: set RESUME_CKPT env var to checkpoint path, e.g.:
#   RESUME_CKPT=/home/zichuanfu2/results/ours_v5_aligned_5ds/checkpoint-45000 sbatch jobs/train_all_aligned.sh

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Attention
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

DATA_ROOT=/home/zichuanfu2/data/GUI-Actor

# 5 datasets (no UGround — too large, 256GB, 75% of data)
# ~225K samples total, ~56K optimizer steps, estimated ~30h
# Will need 2 jobs: first ~45K steps (24h), then resume for remaining ~11K steps
DATA_PATHS="${DATA_ROOT}/guiact_bbox.json,${DATA_ROOT}/guienv_bbox.json,${DATA_ROOT}/amex_bbox.json,${DATA_ROOT}/androidcontrol_bbox.json,${DATA_ROOT}/wave_ui_bbox.json"
IMAGE_FOLDERS="${DATA_ROOT}/GUIAct/web_imgs,${DATA_ROOT}/GUIEnv/guienvs/images,${DATA_ROOT}/AMEX/screenshots,${DATA_ROOT}/AndroidControl/tfrecord/images,${DATA_ROOT}/Wave-UI/images_fixed"

# Build resume arg if RESUME_CKPT is set
RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from checkpoint: $RESUME_CKPT"
    RESUME_ARG="--resume_from_checkpoint $RESUME_CKPT"
fi

CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 \
    src/gui_attention/train.py \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --data_path "$DATA_PATHS" \
    --image_folder "$IMAGE_FOLDERS" \
    --output_dir /home/zichuanfu2/results/ours_v5_aligned_5ds \
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
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none \
    $RESUME_ARG
