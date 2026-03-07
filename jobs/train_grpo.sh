#!/bin/bash
# ============================================================================
# GRPO Training: Saccade Strategy Optimization
#
# Prerequisite: SFT-trained checkpoint (e.g., from train.sh)
# GRPO fine-tunes saccade strategy using reward signal:
#   reward = hit (click in GT bbox) - α * rounds_used
#
# Usage:
#   SFT_CKPT=/path/to/sft/checkpoint NUM_GPUS=8 bash jobs/train_grpo.sh
# ============================================================================

set -e

NUM_GPUS="${NUM_GPUS:-8}"
GROUP_SIZE="${GROUP_SIZE:-8}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE_DIR=$(dirname "$SCRIPT_DIR")
WORK_DIR=$(dirname "$CODE_DIR")

DATA_DIR="$WORK_DIR/data"
MODEL_DIR="$WORK_DIR/models"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$CODE_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# SFT checkpoint to start from (edit this path directly)
SFT_CKPT="$RESULT_DIR/ours_v15_dual/checkpoint-500"

# Auto-detect datasets (same as train.sh)
declare -a ALL_DATASETS=(
    "guiact_bbox.json:GUIAct/GUIAct/web_imgs:0"
    "androidcontrol_bbox.json:AndroidControl/AndroidControl/tfrecord/images:0"
    "wave_ui_bbox.json:Wave-UI/Wave-UI/images_fixed:0"
    "uground_bbox.json:Uground/Uground/images:60000"
    "gta/gta_data/gta_data_wo_web.json:gta/gta_data/image:60000"
)

DATA_PATHS=""
IMAGE_FOLDERS=""
PER_DS_LIMITS=""
for entry in "${ALL_DATASETS[@]}"; do
    IFS=':' read -r json_file img_dir limit <<< "$entry"
    if [ -f "$DATA_DIR/$json_file" ]; then
        if [ -n "$DATA_PATHS" ]; then
            DATA_PATHS="$DATA_PATHS,$DATA_DIR/$json_file"
            IMAGE_FOLDERS="$IMAGE_FOLDERS,$DATA_DIR/$img_dir"
            PER_DS_LIMITS="$PER_DS_LIMITS,$limit"
        else
            DATA_PATHS="$DATA_DIR/$json_file"
            IMAGE_FOLDERS="$DATA_DIR/$img_dir"
            PER_DS_LIMITS="$limit"
        fi
        echo "  ✓ $json_file (limit=$limit)"
    else
        echo "  ✗ $json_file (not found, skipping)"
    fi
done

echo "============================================================"
echo "  GRPO Saccade Training"
echo "  SFT checkpoint: $SFT_CKPT"
echo "  GPUs: $NUM_GPUS"
echo "  Base model: ${BASE_MODEL:-$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL}"
echo "  Output: ${OUTPUT_DIR:-$RESULT_DIR/ours_grpo}"
echo "============================================================"

torchrun --nproc_per_node=$NUM_GPUS \
    src/gui_attention/grpo.py \
    --model_name_or_path "${BASE_MODEL:-$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL}" \
    --resume_ckpt "$SFT_CKPT" \
    --data_path "$DATA_PATHS" \
    --image_folder "$IMAGE_FOLDERS" \
    --max_samples_per_dataset "$PER_DS_LIMITS" \
    --output_dir "${OUTPUT_DIR:-$RESULT_DIR/ours_grpo}" \
    --min_pixels 3136 \
    --low_res_max_pixels 1001600 \
    --crop_size 308 \
    --crop_upscale 3 \
    --max_saccade_rounds 6 \
    --use_dual_tokens true \
    --group_size "$GROUP_SIZE" \
    --reward_hit 1.0 \
    --reward_proximity_weight 0.25 \
    --reward_round_penalty 0.05 \
    --kl_coeff 0.01 \
    --temperature 1.0 \
    --look_sft_weight 0.1 \
    --click_sft_weight 0.1 \
    --lm_loss_weight 0.0 \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules "q_proj,v_proj" \
    --action_head_lr 2e-5 \
    --lora_lr 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.0 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none \
    2>&1 | tee "$LOG_DIR/train_grpo.txt"
