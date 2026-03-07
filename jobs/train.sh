#!/bin/bash
# ============================================================================
# v15 Training: Dual Head (LookHead + ClickHead)
#
# ClickHead initialized from GUI-Actor pointer head (43% ScreenSpot-Pro).
# Both heads train from step 0 (no warmup — ClickHead already has ability).
#
# Config: LoRA + 1M low-res + 308px crop + mask old crops
#
# Usage (Tencent server, 8 GPUs):
#   NUM_GPUS=8 bash jobs/train.sh
#
# Resume:
#   RESUME_CKPT=/path/to/checkpoint NUM_GPUS=8 bash jobs/train.sh
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
    RESUME_ARG="--resume_ckpt $RESUME_CKPT"
fi

CLICK_HEAD_FROM="${CLICK_HEAD_FROM:-$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL}"
FREE_REASONING_SFT="${FREE_REASONING_SFT:-true}"
APPEND_ASSISTANT_EOS="${APPEND_ASSISTANT_EOS:-true}"

echo "============================================================"
echo "  GUI-Attention v15 Training (Dual Head: LookHead + ClickHead)"
echo "  ClickHead initialized from GUI-Actor pointer head"
echo "  Both heads from step 0 (no warmup phases)"
echo "  GPUs: $NUM_GPUS"
echo "  Base model: ${BASE_MODEL:-$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL}"
echo "  ClickHead from: $CLICK_HEAD_FROM"
echo "  Output: ${OUTPUT_DIR:-$RESULT_DIR/ours_v15_dual}"
echo "  Free-reasoning SFT: $FREE_REASONING_SFT"
echo "  Append assistant EOS: $APPEND_ASSISTANT_EOS"
echo "============================================================"

torchrun --nproc_per_node=$NUM_GPUS \
    src/gui_attention/train.py \
    --model_name_or_path "${BASE_MODEL:-$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL}" \
    --click_head_from "$CLICK_HEAD_FROM" \
    --data_path "$DATA_PATHS" \
    --image_folder "$IMAGE_FOLDERS" \
    --max_samples_per_dataset "$PER_DS_LIMITS" \
    --output_dir "${OUTPUT_DIR:-$RESULT_DIR/ours_v15_dual}" \
    --min_pixels 3136 \
    --low_res_max_pixels 1001600 \
    --crop_size 308 \
    --crop_upscale 3 \
    --crop_jitter 0.05 \
    --max_saccade_rounds 6 \
    --click_phase_step 0 \
    --use_dual_tokens true \
    --free_reasoning_sft "$FREE_REASONING_SFT" \
    --append_assistant_eos "$APPEND_ASSISTANT_EOS" \
    --use_lora true \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules "q_proj,v_proj" \
    --action_head_lr 1e-4 \
    --lora_lr 5e-5 \
    --lm_loss_weight 0.1 \
    --look_loss_weight 1.0 \
    --click_loss_weight 1.0 \
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
    $RESUME_ARG \
    2>&1 | tee "$LOG_DIR/train_v15_dual.txt"
