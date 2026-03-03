#!/bin/bash
# ============================================================================
# v9 Training: v5 config (LoRA + 1M pixels) + M-RoPE position alignment
#
# vs v5: + align_crop_mrope (new)
# vs v8: LoRA instead of full-param, 1M instead of 200K pixels, higher LR
#
# Usage (Tencent server, 8 GPUs):
#   NUM_GPUS=8 bash jobs/train_v9.sh
#
# Resume:
#   RESUME_CKPT=/path/to/checkpoint NUM_GPUS=8 bash jobs/train_v9.sh
# ============================================================================

set -e

NUM_GPUS="${NUM_GPUS:-8}"

# Auto-detect workspace
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

# All candidate datasets
declare -a ALL_DATASETS=(
    "guiact_bbox.json:GUIAct/GUIAct/web_imgs:0"
    "androidcontrol_bbox.json:AndroidControl/AndroidControl/tfrecord/images:0"
    "wave_ui_bbox.json:Wave-UI/Wave-UI/images_fixed:0"
    "uground_bbox.json:Uground/Uground/images:60000"
    "gta/gta_data/gta_data_wo_web.json:gta/gta_data/image:60000"
)

# Auto-detect available datasets
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

if [ -z "$DATA_PATHS" ]; then
    echo "ERROR: No datasets found in $DATA_DIR"
    exit 1
fi

# Resume
RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    RESUME_ARG="--resume_from_checkpoint $RESUME_CKPT"
fi

echo "============================================================"
echo "  GUI-Attention v9 Training"
echo "  Config: LoRA + 1M pixels + M-RoPE alignment"
echo "  GPUs: $NUM_GPUS"
echo "  Base model: ${BASE_MODEL:-$MODEL_DIR/Qwen2.5-VL-3B-Instruct}"
echo "  Output: ${OUTPUT_DIR:-$RESULT_DIR/ours_v9_qwen}"
echo "============================================================"

torchrun --nproc_per_node=$NUM_GPUS \
    src/gui_attention/train.py \
    --model_name_or_path "${BASE_MODEL:-$MODEL_DIR/Qwen2.5-VL-3B-Instruct}" \
    --data_path "$DATA_PATHS" \
    --image_folder "$IMAGE_FOLDERS" \
    --max_samples_per_dataset "$PER_DS_LIMITS" \
    --output_dir "${OUTPUT_DIR:-$RESULT_DIR/ours_v9_qwen}" \
    --min_pixels 3136 \
    --low_res_max_pixels 1003520 \
    --crop_target_pixels 1003520 \
    --crop_ratio 0.3 \
    --crop_jitter 0.05 \
    --max_saccade_rounds 4 \
    --warmup_rounds_step 500 \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules "q_proj,v_proj" \
    --action_head_lr 1e-4 \
    --lora_lr 5e-5 \
    --lm_loss_weight 1.0 \
    --pointer_loss_weight 1.0 \
    --align_crop_mrope false \
    --teacher_forcing_crop false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --weight_decay 0.0 \
    --warmup_ratio 0.0 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --soft_labels false \
    --report_to none \
    $RESUME_ARG \
    2>&1 | tee "$LOG_DIR/train_v9_qwen.txt"
