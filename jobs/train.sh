#!/bin/bash
# ============================================================================
# v15 Training: Dual Head (LookHead + ClickHead)
#
# Both LookHead and ClickHead are initialized from GUI-Actor pointer head.
# Both heads train from step 0 (no warmup).
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

usage() {
    cat <<'EOF'
Usage:
  bash jobs/train.sh [options]

Options:
  --num_gpus N
  --resume_ckpt PATH
  --base_model PATH
  --click_head_from PATH
  --output_dir PATH
  --free_reasoning_sft true|false
  --append_assistant_eos true|false
  --lm_loss_weight FLOAT
  --look_loss_weight FLOAT
  --click_loss_weight FLOAT
  --lm_reasoning_token_weight FLOAT
  --lm_format_token_weight FLOAT
  --lm_look_token_weight FLOAT
  --lm_click_token_weight FLOAT
  --help

Environment variables are still supported, but CLI flags take precedence.
EOF
}

NUM_GPUS="${NUM_GPUS:-8}"
BASE_MODEL="${BASE_MODEL:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
CLICK_HEAD_FROM="${CLICK_HEAD_FROM:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
FREE_REASONING_SFT="${FREE_REASONING_SFT:-true}"
APPEND_ASSISTANT_EOS="${APPEND_ASSISTANT_EOS:-true}"
LM_LOSS_WEIGHT="${LM_LOSS_WEIGHT:-0.5}"
LOOK_LOSS_WEIGHT="${LOOK_LOSS_WEIGHT:-1.0}"
CLICK_LOSS_WEIGHT="${CLICK_LOSS_WEIGHT:-4.0}"
LM_REASONING_TOKEN_WEIGHT="${LM_REASONING_TOKEN_WEIGHT:-0.1}"
LM_FORMAT_TOKEN_WEIGHT="${LM_FORMAT_TOKEN_WEIGHT:-0.5}"
LM_LOOK_TOKEN_WEIGHT="${LM_LOOK_TOKEN_WEIGHT:-1.0}"
LM_CLICK_TOKEN_WEIGHT="${LM_CLICK_TOKEN_WEIGHT:-2.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume_ckpt)
            RESUME_CKPT="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --click_head_from)
            CLICK_HEAD_FROM="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --free_reasoning_sft)
            FREE_REASONING_SFT="$2"
            shift 2
            ;;
        --append_assistant_eos)
            APPEND_ASSISTANT_EOS="$2"
            shift 2
            ;;
        --lm_loss_weight)
            LM_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --look_loss_weight)
            LOOK_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --click_loss_weight)
            CLICK_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --lm_reasoning_token_weight)
            LM_REASONING_TOKEN_WEIGHT="$2"
            shift 2
            ;;
        --lm_format_token_weight)
            LM_FORMAT_TOKEN_WEIGHT="$2"
            shift 2
            ;;
        --lm_look_token_weight)
            LM_LOOK_TOKEN_WEIGHT="$2"
            shift 2
            ;;
        --lm_click_token_weight)
            LM_CLICK_TOKEN_WEIGHT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

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
    "groundcua_bbox.json:GroundCUA/data:120000"
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

echo "============================================================"
echo "  GUI-Attention v15 Training (Dual Head: LookHead + ClickHead)"
echo "  LookHead and ClickHead initialized from GUI-Actor pointer head"
echo "  Both heads from step 0 (no warmup phases)"
echo "  GPUs: $NUM_GPUS"
echo "  Base model: ${BASE_MODEL:-$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL}"
echo "  Head init from: $CLICK_HEAD_FROM"
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
    --lm_loss_weight "$LM_LOSS_WEIGHT" \
    --look_loss_weight "$LOOK_LOSS_WEIGHT" \
    --click_loss_weight "$CLICK_LOSS_WEIGHT" \
    --lm_reasoning_token_weight "$LM_REASONING_TOKEN_WEIGHT" \
    --lm_format_token_weight "$LM_FORMAT_TOKEN_WEIGHT" \
    --lm_look_token_weight "$LM_LOOK_TOKEN_WEIGHT" \
    --lm_click_token_weight "$LM_CLICK_TOKEN_WEIGHT" \
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
