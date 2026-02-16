#!/bin/bash
#SBATCH --job-name=guiactor_5k
#SBATCH --output=/home/zichuanfu2/GUI-Attention/logs/train_guiactor_5k_%j.txt
#SBATCH --error=/home/zichuanfu2/GUI-Attention/logs/train_guiactor_5k_err_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate fzc-guiattn

cd /home/zichuanfu2/GUI-Actor
export PYTHONUNBUFFERED=1

DATA_YAML=/home/zichuanfu2/GUI-Actor/data/guiact_5k_config.yaml

# ===== Stage 1: Warmup (pointer head only) =====
echo "====== Stage 1: Warmup ======"
CUDA_VISIBLE_DEVICES="0" python3 train.py \
    --data_path "$DATA_YAML" \
    --image_folder "" \
    --model_type qwen25vl \
    --model_name_or_path /home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/zichuanfu2/results/guiactor_5k_warmup \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 24576 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --max_pixels 5720064 \
    --unfreeze_all_parameters False \
    --unfreeze_pointer_head True \
    --unfreeze_lm_head False \
    --unfreeze_base_model False \
    --unfreeze_last_n_layers -1 \
    --unfreeze_new_tokens True \
    --unfreeze_visual False \
    --pointer_loss_weight 1.0 \
    --lm_loss_weight -1.0

echo "====== Stage 1 Done ======"

# ===== Stage 2: SFT (full fine-tune) =====
echo "====== Stage 2: SFT ======"

# Find the latest checkpoint from warmup
WARMUP_CKPT=$(ls -d /home/zichuanfu2/results/guiactor_5k_warmup/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$WARMUP_CKPT" ]; then
    WARMUP_CKPT=/home/zichuanfu2/results/guiactor_5k_warmup
fi
echo "Using warmup checkpoint: $WARMUP_CKPT"

CUDA_VISIBLE_DEVICES="0" python3 train.py \
    --data_path "$DATA_YAML" \
    --image_folder "" \
    --model_type qwen25vl \
    --model_name_or_path "$WARMUP_CKPT" \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/zichuanfu2/results/guiactor_5k_sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 24576 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --max_pixels 5720064 \
    --unfreeze_all_parameters True \
    --unfreeze_pointer_head False \
    --unfreeze_lm_head False \
    --unfreeze_base_model False \
    --unfreeze_last_n_layers -1 \
    --unfreeze_new_tokens False \
    --unfreeze_visual False \
    --pointer_loss_weight 1.0 \
    --lm_loss_weight 1.0

echo "====== Stage 2 Done ======"
