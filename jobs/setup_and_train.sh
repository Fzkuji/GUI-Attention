#!/bin/bash
# ============================================================================
# GUI-Attention: Full Setup, Data Download, and Training
# ============================================================================
# One-script setup for a fresh server. Handles:
#   1. Clone repos & install dependencies
#   2. Download base model
#   3. Download all 6 GUI-Actor training datasets
#   4. Train (2-GPU data parallel, LoRA + ActionHead)
#   5. Evaluate on ScreenSpot v1/v2/Pro
#
# Usage:
#   bash jobs/setup_and_train.sh [OPTIONS]
#
# Options (env vars):
#   WORK_DIR        - working directory (default: ~/GUI-Attention-Workspace)
#   NUM_GPUS        - number of GPUs for training (default: 2)
#   SKIP_DOWNLOAD   - set to 1 to skip data download
#   SKIP_TRAIN      - set to 1 to skip training
#   SKIP_EVAL       - set to 1 to skip evaluation
#   DATASETS        - comma-separated dataset names to use
#                     (default: all 6: guiact,guienv,amex,androidcontrol,waveui,uground)
#   CONDA_ENV       - conda env name (default: gui-attention)
#   MAX_EPOCHS      - training epochs (default: 1)
# ============================================================================

set -e

# ── Configuration ───────────────────────────────────────────────────────────

WORK_DIR="${WORK_DIR:-$HOME/GUI-Attention-Workspace}"
NUM_GPUS="${NUM_GPUS:-2}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DATASETS="${DATASETS:-guiact,guienv,amex,androidcontrol,waveui,uground}"
CONDA_ENV="${CONDA_ENV:-gui-attention}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

CODE_DIR="$WORK_DIR/GUI-Attention"
DATA_DIR="$WORK_DIR/data"
MODEL_DIR="$WORK_DIR/models"
RESULT_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"

HF_DATA_REPO="https://huggingface.co/datasets/cckevinn/GUI-Actor-Data/resolve/main"
BASE_MODEL_HF="Qwen/Qwen2.5-VL-3B-Instruct"

mkdir -p "$WORK_DIR" "$DATA_DIR" "$MODEL_DIR" "$RESULT_DIR" "$LOG_DIR"

echo "============================================================"
echo "  GUI-Attention: Setup & Train"
echo "============================================================"
echo "  WORK_DIR:    $WORK_DIR"
echo "  NUM_GPUS:    $NUM_GPUS"
echo "  DATASETS:    $DATASETS"
echo "  MAX_EPOCHS:  $MAX_EPOCHS"
echo "============================================================"
echo ""

# ── Step 1: Clone code & install ────────────────────────────────────────────

echo ">>> [1/5] Setting up code and environment"

if [ ! -d "$CODE_DIR" ]; then
    echo "  Cloning GUI-Attention..."
    git clone https://github.com/Fzkuji/GUI-Attention.git "$CODE_DIR"
else
    echo "  GUI-Attention already cloned, pulling latest..."
    cd "$CODE_DIR" && git pull
fi

# Conda env setup
if ! conda info --envs 2>/dev/null | grep -q "$CONDA_ENV"; then
    echo "  Creating conda env: $CONDA_ENV"
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "  Installing dependencies..."
cd "$CODE_DIR"
pip install -e ".[train,eval]" -q
# flash-attn (optional, build from source)
pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "  (flash-attn install failed, continuing without it)"

echo ""

# ── Step 2: Download base model ────────────────────────────────────────────

echo ">>> [2/5] Downloading base model"

if [ -d "$MODEL_DIR/Qwen2.5-VL-3B-Instruct" ]; then
    echo "  Base model already exists, skipping"
else
    echo "  Downloading $BASE_MODEL_HF..."
    pip install huggingface_hub -q
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$BASE_MODEL_HF', local_dir='$MODEL_DIR/Qwen2.5-VL-3B-Instruct')
print('  Done.')
"
fi

echo ""

# ── Step 3: Download training data ─────────────────────────────────────────

if [ "$SKIP_DOWNLOAD" = "1" ]; then
    echo ">>> [3/5] Skipping data download (SKIP_DOWNLOAD=1)"
else
    echo ">>> [3/5] Downloading training data"

    cd "$DATA_DIR"

    # --- JSON annotations (always download, they're small) ---
    echo "  Downloading JSON annotations..."
    declare -A JSON_FILES=(
        ["guiact"]="guiact_bbox.json"
        ["guienv"]="guienv_bbox.json"
        ["amex"]="amex_bbox.json"
        ["androidcontrol"]="androidcontrol_bbox.json"
        ["waveui"]="wave_ui_bbox.json"
        ["uground"]="uground_bbox.json"
    )
    for ds in guiact guienv amex androidcontrol waveui uground; do
        json="${JSON_FILES[$ds]}"
        if echo "$DATASETS" | grep -q "$ds"; then
            if [ ! -f "$json" ]; then
                echo "    Downloading $json..."
                wget -q --show-progress "$HF_DATA_REPO/$json" -O "$json"
            else
                echo "    $json exists"
            fi
        fi
    done

    # --- Image archives ---
    # GUIAct (4 GB)
    if echo "$DATASETS" | grep -q "guiact"; then
        if [ -d "GUIAct/web_imgs" ] && [ -n "$(ls GUIAct/web_imgs/ 2>/dev/null | head -1)" ]; then
            echo "  GUIAct images: exists"
        else
            echo "  Downloading GUIAct images (4 GB)..."
            wget -q --show-progress "$HF_DATA_REPO/GUIAct_images.zip" -O GUIAct_images.zip
            unzip -q -o GUIAct_images.zip -d GUIAct/
            rm -f GUIAct_images.zip
        fi
    fi

    # GUIEnv (5.9 GB)
    if echo "$DATASETS" | grep -q "guienv"; then
        if [ -d "GUIEnv/guienvs/images" ] && [ -n "$(ls GUIEnv/guienvs/images/ 2>/dev/null | head -1)" ]; then
            echo "  GUIEnv images: exists"
        else
            echo "  Downloading GUIEnv images (5.9 GB)..."
            wget -q --show-progress "$HF_DATA_REPO/GUIEnv_images.zip" -O GUIEnv_images.zip
            mkdir -p GUIEnv
            unzip -q -o GUIEnv_images.zip -d GUIEnv/
            rm -f GUIEnv_images.zip
        fi
    fi

    # Wave-UI (24.4 GB)
    if echo "$DATASETS" | grep -q "waveui"; then
        if [ -d "Wave-UI/images_fixed" ] && [ -n "$(ls Wave-UI/images_fixed/ 2>/dev/null | head -1)" ]; then
            echo "  Wave-UI images: exists"
        else
            echo "  Downloading Wave-UI images (24.4 GB)..."
            wget -q --show-progress "$HF_DATA_REPO/Wave-UI_images.zip" -O Wave-UI_images.zip
            mkdir -p Wave-UI
            unzip -q -o Wave-UI_images.zip -d Wave-UI/
            rm -f Wave-UI_images.zip
        fi
    fi

    # AndroidControl (49.3 GB)
    if echo "$DATASETS" | grep -q "androidcontrol"; then
        if [ -d "AndroidControl/tfrecord/images" ] && [ -n "$(ls AndroidControl/tfrecord/images/ 2>/dev/null | head -1)" ]; then
            echo "  AndroidControl images: exists"
        else
            echo "  Downloading AndroidControl images (49.3 GB)..."
            wget -q --show-progress "$HF_DATA_REPO/AndroidControl_images.zip" -O AndroidControl_images.zip
            mkdir -p AndroidControl
            unzip -q -o AndroidControl_images.zip -d AndroidControl/
            rm -f AndroidControl_images.zip
        fi
    fi

    # AMEX (92 GB, 3 parts)
    if echo "$DATASETS" | grep -q "amex"; then
        if [ -d "AMEX/screenshots" ] && [ -n "$(ls AMEX/screenshots/ 2>/dev/null | head -1)" ]; then
            echo "  AMEX images: exists"
        else
            echo "  Downloading AMEX images (92 GB, 3 parts)..."
            for part in amex_images_part_aa amex_images_part_ab amex_images_part_ac; do
                if [ ! -f "$part" ]; then
                    wget -q --show-progress "$HF_DATA_REPO/$part" -O "$part"
                fi
            done
            echo "    Combining and extracting..."
            cat amex_images_part_* > amex_images.zip
            mkdir -p AMEX
            7z x amex_images.zip -aoa -oAMEX/ || unzip -q -o amex_images.zip -d AMEX/
            rm -f amex_images_part_* amex_images.zip
        fi
    fi

    # UGround (256 GB, 6 parts)
    if echo "$DATASETS" | grep -q "uground"; then
        if [ -d "Uground/images" ] && [ -n "$(ls Uground/images/ 2>/dev/null | head -1)" ]; then
            echo "  UGround images: exists"
        else
            echo "  Downloading UGround images (256 GB, 6 split files)..."
            for part in Uground_images_split.z01 Uground_images_split.z02 Uground_images_split.z03 Uground_images_split.z04 Uground_images_split.z05 Uground_images_split.zip; do
                if [ ! -f "$part" ]; then
                    wget -q --show-progress "$HF_DATA_REPO/$part" -O "$part"
                fi
            done
            echo "    Combining and extracting..."
            cat Uground_images_split.z* Uground_images_split.zip > Uground_images.zip
            mkdir -p Uground
            7z x Uground_images.zip -aoa -oUground/ || unzip -q -o Uground_images.zip -d Uground/
            rm -f Uground_images_split.z* Uground_images_split.zip Uground_images.zip
        fi
    fi

    # ScreenSpot-Pro eval data
    echo "  Downloading ScreenSpot-Pro eval data..."
    if [ ! -d "ScreenSpot-Pro" ]; then
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('liuzhch/ScreenSpot-Pro', local_dir='$DATA_DIR/ScreenSpot-Pro', repo_type='dataset')
print('  Done.')
" 2>/dev/null || echo "  (ScreenSpot-Pro download skipped, will use HF datasets at eval time)"
    else
        echo "  ScreenSpot-Pro exists"
    fi
fi

echo ""

# ── Step 4: Train ───────────────────────────────────────────────────────────

if [ "$SKIP_TRAIN" = "1" ]; then
    echo ">>> [4/5] Skipping training (SKIP_TRAIN=1)"
else
    echo ">>> [4/5] Training"

    cd "$CODE_DIR"
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=src:$PYTHONPATH

    # Build data_path and image_folder args based on DATASETS
    DATA_PATHS=""
    IMAGE_FOLDERS=""

    declare -A DS_JSON=(
        ["guiact"]="guiact_bbox.json"
        ["guienv"]="guienv_bbox.json"
        ["amex"]="amex_bbox.json"
        ["androidcontrol"]="androidcontrol_bbox.json"
        ["waveui"]="wave_ui_bbox.json"
        ["uground"]="uground_bbox.json"
    )
    declare -A DS_IMGDIR=(
        ["guiact"]="GUIAct/web_imgs"
        ["guienv"]="GUIEnv/guienvs/images"
        ["amex"]="AMEX/screenshots"
        ["androidcontrol"]="AndroidControl/tfrecord/images"
        ["waveui"]="Wave-UI/images_fixed"
        ["uground"]="Uground/images"
    )

    IFS=',' read -ra DS_LIST <<< "$DATASETS"
    for ds in "${DS_LIST[@]}"; do
        ds=$(echo "$ds" | xargs)  # trim
        json="${DS_JSON[$ds]}"
        imgdir="${DS_IMGDIR[$ds]}"
        if [ -z "$json" ]; then
            echo "  WARNING: Unknown dataset '$ds', skipping"
            continue
        fi
        if [ -n "$DATA_PATHS" ]; then
            DATA_PATHS="$DATA_PATHS,$DATA_DIR/$json"
            IMAGE_FOLDERS="$IMAGE_FOLDERS,$DATA_DIR/$imgdir"
        else
            DATA_PATHS="$DATA_DIR/$json"
            IMAGE_FOLDERS="$DATA_DIR/$imgdir"
        fi
    done

    echo "  Data paths: $DATA_PATHS"
    echo "  Image dirs: $IMAGE_FOLDERS"

    # Determine GPU list and gradient accumulation
    if [ "$NUM_GPUS" -le 2 ]; then
        GA_STEPS=4
    elif [ "$NUM_GPUS" -le 4 ]; then
        GA_STEPS=2
    else
        GA_STEPS=1
    fi
    EFFECTIVE_BATCH=$((NUM_GPUS * GA_STEPS))
    echo "  GPUs=$NUM_GPUS  ga=$GA_STEPS  effective_batch=$EFFECTIVE_BATCH"

    OUTPUT_NAME="ours_v4_$(echo $DATASETS | tr ',' '_')"

    torchrun --nproc_per_node=$NUM_GPUS \
        src/gui_attention/train.py \
        --model_name_or_path "$MODEL_DIR/Qwen2.5-VL-3B-Instruct" \
        --data_path "$DATA_PATHS" \
        --image_folder "$IMAGE_FOLDERS" \
        --output_dir "$RESULT_DIR/$OUTPUT_NAME" \
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
        --num_train_epochs "$MAX_EPOCHS" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $GA_STEPS \
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
        2>&1 | tee "$LOG_DIR/train_${OUTPUT_NAME}.txt"
fi

echo ""

# ── Step 5: Evaluate ────────────────────────────────────────────────────────

if [ "$SKIP_EVAL" = "1" ]; then
    echo ">>> [5/5] Skipping evaluation (SKIP_EVAL=1)"
else
    echo ">>> [5/5] Evaluating on ScreenSpot benchmarks"

    cd "$CODE_DIR"
    export PYTHONPATH=src:$PYTHONPATH

    OUTPUT_NAME="ours_v4_$(echo $DATASETS | tr ',' '_')"
    CHECKPOINT="$RESULT_DIR/$OUTPUT_NAME/final"
    BASE_MODEL="$MODEL_DIR/Qwen2.5-VL-3B-Instruct"

    echo "  Checkpoint: $CHECKPOINT"

    # ScreenSpot-Pro
    if [ -d "$DATA_DIR/ScreenSpot-Pro" ]; then
        echo "  >>> ScreenSpot-Pro"
        python eval/eval_screenspot_pro_aligned.py \
            --checkpoint "$CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --data_path "$DATA_DIR/ScreenSpot-Pro" \
            --rounds 3 --crop_ratio 0.3 --device cuda:0 \
            2>&1 | tee "$LOG_DIR/eval_screenspot_pro.txt"
    fi

    # ScreenSpot v1
    echo "  >>> ScreenSpot v1"
    python eval/eval_screenspot.py \
        --checkpoint "$CHECKPOINT" \
        --base_model "$BASE_MODEL" \
        --rounds 3 --crop_ratio 0.3 --device cuda:0 \
        2>&1 | tee "$LOG_DIR/eval_screenspot_v1.txt"

    # ScreenSpot v2
    echo "  >>> ScreenSpot v2"
    python eval/eval_screenspot_v2.py \
        --checkpoint "$CHECKPOINT" \
        --base_model "$BASE_MODEL" \
        --rounds 3 --crop_ratio 0.3 --device cuda:0 \
        2>&1 | tee "$LOG_DIR/eval_screenspot_v2.txt"
fi

echo ""
echo "============================================================"
echo "  All done!"
echo "  Results: $RESULT_DIR"
echo "  Logs:    $LOG_DIR"
echo "============================================================"
