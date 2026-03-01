#!/bin/bash
# ============================================================================
# GUI-Attention v6: Full-param + DDP, GUI-AIMA aligned config
# ============================================================================
# Changes from v5:
#   - Full parameter fine-tuning (no LoRA), matching GUI-AIMA
#   - Manual DDP (torchrun + all_reduce), no DeepSpeed (95GB VRAM sufficient)
#   - lr=5e-6 for backbone (matching GUI-AIMA), action_head_lr=5e-5
#   - GA=8 → effective batch=64 (matching GUI-AIMA)
#   - Dataset: GUIAct + AndroidControl + Wave-UI + UGround(60K) + GTA(60K)
#     ≈ 259K instructions, ~85K images (matching GUI-AIMA)
#
# Usage:
#   conda activate gui-attention
#   NUM_GPUS=8 bash jobs/setup_and_train_v6.sh
# ============================================================================

set -e

# Auto-detect: script is at WORK_DIR/GUI-Attention/jobs/setup_and_train_v6.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname $(dirname "$SCRIPT_DIR"))}"
NUM_GPUS="${NUM_GPUS:-8}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

CODE_DIR="$WORK_DIR/GUI-Attention"
DATA_DIR="$WORK_DIR/data"
MODEL_DIR="$WORK_DIR/models"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"

mkdir -p "$WORK_DIR" "$DATA_DIR" "$MODEL_DIR" "$RESULT_DIR" "$LOG_DIR"

echo "============================================================"
echo "  GUI-Attention v6: Full-param + ZeRO-3 (GUI-AIMA aligned)"
echo "============================================================"
echo "  WORK_DIR:    $WORK_DIR"
echo "  NUM_GPUS:    $NUM_GPUS"
echo "  MAX_EPOCHS:  $MAX_EPOCHS"
echo "============================================================"

# ── Step 1: Code ────────────────────────────────────────────────────────────

echo ">>> [1/5] Setting up code"
if [ ! -d "$CODE_DIR" ]; then
    git clone https://github.com/Fzkuji/GUI-Attention.git "$CODE_DIR"
else
    cd "$CODE_DIR" && git pull
fi
cd "$CODE_DIR"
export PYTHONPATH=src:$PYTHONPATH

# ── Step 2: Model ───────────────────────────────────────────────────────────

echo ">>> [2/5] Checking base model"
if [ -d "$MODEL_DIR/Qwen2.5-VL-3B-Instruct" ]; then
    echo "  Base model exists"
else
    echo "  Downloading Qwen2.5-VL-3B-Instruct..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='$MODEL_DIR/Qwen2.5-VL-3B-Instruct')
"
fi

# ── Step 3: Data ────────────────────────────────────────────────────────────

if [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo ">>> [3/5] Data download (handle manually if needed)"
else
    echo ">>> [3/5] Skipping download"
fi

# ── Step 4: Train ───────────────────────────────────────────────────────────

if [ "$SKIP_TRAIN" = "1" ]; then
    echo ">>> [4/5] Skipping training"
else
    echo ">>> [4/5] Training (Full-param + ZeRO-3)"

    cd "$CODE_DIR"
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=src:$PYTHONPATH

    # Dataset: GUI-AIMA aligned (5 datasets)
    # GUIAct + AndroidControl + Wave-UI + UGround (first 60K) + GTA (60K)
    DATA_PATHS="$DATA_DIR/guiact_bbox.json,$DATA_DIR/androidcontrol_bbox.json,$DATA_DIR/wave_ui_bbox.json,$DATA_DIR/uground_bbox.json,$DATA_DIR/gta/gta_data/gta_data_wo_web.json"
    IMAGE_FOLDERS="$DATA_DIR/GUIAct/GUIAct/web_imgs,$DATA_DIR/AndroidControl/AndroidControl/tfrecord/images,$DATA_DIR/Wave-UI/Wave-UI/images_fixed,$DATA_DIR/Uground/Uground/images,$DATA_DIR/gta/gta_data/image"

    # Per-dataset limits: GUIAct=all, AndroidControl=all, Wave-UI=all, UGround=60K, GTA=60K
    MAX_PER_DS="0,0,0,60000,60000"

    OUTPUT_NAME="ours_v6_fullparam_aima_aligned"

    GA_STEPS=8  # 8 GPUs × 1 × GA8 = effective batch 64

    torchrun --nproc_per_node=$NUM_GPUS \
        src/gui_attention/train.py \
        --model_name_or_path "$MODEL_DIR/Qwen2.5-VL-3B-Instruct" \
        --data_path "$DATA_PATHS" \
        --image_folder "$IMAGE_FOLDERS" \
        --max_samples_per_dataset "$MAX_PER_DS" \
        --output_dir "$RESULT_DIR/$OUTPUT_NAME" \
        --min_pixels 3136 \
        --low_res_max_pixels 200704 \
        --high_res_max_pixels 5720064 \
        --crop_ratio 0.3 \
        --crop_target_pixels 200704 \
        --crop_jitter 0.05 \
        --max_saccade_rounds 3 \
        --no_use_lora \
        --action_head_lr 5e-5 \
        --lora_lr 5e-6 \
        --num_train_epochs "$MAX_EPOCHS" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $GA_STEPS \
        --learning_rate 5e-6 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --logging_steps 50 \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 2 \
        --bf16 true \
        --gradient_checkpointing true \
        --report_to none \
        2>&1 | tee "$LOG_DIR/train_${OUTPUT_NAME}.txt"
fi

# ── Step 5: Evaluate ────────────────────────────────────────────────────────

if [ "$SKIP_EVAL" = "1" ]; then
    echo ">>> [5/5] Skipping evaluation"
else
    echo ">>> [5/5] Evaluating"

    cd "$CODE_DIR"
    export PYTHONPATH=src:$PYTHONPATH

    OUTPUT_NAME="ours_v6_fullparam_aima_aligned"
    CHECKPOINT="$RESULT_DIR/$OUTPUT_NAME/final"
    BASE_MODEL="$MODEL_DIR/Qwen2.5-VL-3B-Instruct"

    if [ -d "$DATA_DIR/ScreenSpot-Pro" ]; then
        echo "  >>> ScreenSpot-Pro"
        python eval/eval_screenspot_pro_aligned.py \
            --checkpoint "$CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --data_path "$DATA_DIR/ScreenSpot-Pro" \
            --rounds 3 --crop_ratio 0.3 --crop_target_pixels 200704 --device cuda:0 \
            2>&1 | tee "$LOG_DIR/eval_screenspot_pro_v6.txt"
    fi

    echo "  >>> ScreenSpot v2"
    python eval/eval_screenspot_v2.py \
        --checkpoint "$CHECKPOINT" \
        --base_model "$BASE_MODEL" \
        --rounds 3 --crop_ratio 0.3 --crop_target_pixels 200704 --device cuda:0 \
        2>&1 | tee "$LOG_DIR/eval_screenspot_v2_v6.txt"
fi

echo "============================================================"
echo "  Done! Results: $RESULT_DIR"
echo "============================================================"
