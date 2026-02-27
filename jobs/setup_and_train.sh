#!/bin/bash
# ============================================================================
# GUI-Attention: Data Download, Training, and Evaluation
# ============================================================================
# Handles:
#   1. Clone/pull code
#   2. Download base model (Qwen2.5-VL-3B-Instruct)
#   3. Download all 6 GUI-Actor training datasets (~438GB)
#   4. Train (multi-GPU data parallel, LoRA + ActionHead)
#   5. Evaluate on ScreenSpot v1/v2/Pro
#
# Prerequisites — install environment manually before running:
#   conda create -n gui-attention python=3.10 -y
#   conda activate gui-attention
#   pip install -e ".[train,eval]"
#   pip install flash-attn --no-build-isolation
#
# Usage:
#   conda activate gui-attention
#   NUM_GPUS=8 bash jobs/setup_and_train.sh
#
# Options (env vars):
#   WORK_DIR        - working directory (default: ~/GUI-Attention-Workspace)
#   NUM_GPUS        - number of GPUs for training (default: 2)
#   SKIP_DOWNLOAD   - set to 1 to skip data download
#   SKIP_TRAIN      - set to 1 to skip training
#   SKIP_EVAL       - set to 1 to skip evaluation
#   DATASETS        - comma-separated dataset names to use
#                     (default: all 6: guiact,guienv,amex,androidcontrol,waveui,uground)
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

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_DATA_REPO="${HF_ENDPOINT}/datasets/cckevinn/GUI-Actor-Data/resolve/main"
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

# ── Step 1: Clone code ─────────────────────────────────────────────────────

echo ">>> [1/5] Setting up code"

if [ ! -d "$CODE_DIR" ]; then
    echo "  Cloning GUI-Attention..."
    git clone https://github.com/Fzkuji/GUI-Attention.git "$CODE_DIR"
else
    echo "  GUI-Attention already cloned, pulling latest..."
    cd "$CODE_DIR" && git pull
fi

cd "$CODE_DIR"
export PYTHONPATH=src:$PYTHONPATH

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
    echo "  Using HF endpoint: $HF_ENDPOINT"

    # Check if all datasets are already extracted
    ALL_EXTRACTED=1
    IFS=',' read -ra _DS_CHECK <<< "$DATASETS"
    declare -A _DS_CHECK_DIR=(
        ["guiact"]="GUIAct/web_imgs" ["guienv"]="GUIEnv/guienvs/images"
        ["amex"]="AMEX/screenshots" ["androidcontrol"]="AndroidControl/tfrecord/images"
        ["waveui"]="Wave-UI/images_fixed" ["uground"]="Uground/images"
    )
    for _ds in "${_DS_CHECK[@]}"; do
        _ds=$(echo "$_ds" | xargs)
        _dir="${_DS_CHECK_DIR[$_ds]}"
        if [ -n "$_dir" ] && [ ! -d "$DATA_DIR/$_dir" ]; then
            ALL_EXTRACTED=0
            break
        fi
    done

    if [ "$ALL_EXTRACTED" = "1" ]; then
        echo "  All datasets already extracted, skipping download"
    else
        # Download entire GUI-Actor dataset from HuggingFace (respects HF_ENDPOINT mirror)
        echo "  Downloading GUI-Actor dataset (~438GB)..."
        python -c "
import os
os.environ.setdefault('HF_ENDPOINT', '$HF_ENDPOINT')
from huggingface_hub import snapshot_download
snapshot_download('cckevinn/GUI-Actor-Data', local_dir='$DATA_DIR', repo_type='dataset')
print('  Download complete.')
"
    fi

    # Extract image archives
    cd "$DATA_DIR"

    # GUIAct (4 GB)
    if [ -f "GUIAct_images.zip" ] && [ ! -d "GUIAct/web_imgs" ]; then
        echo "  Extracting GUIAct..."
        unzip -q -o GUIAct_images.zip -d GUIAct/
        rm -f GUIAct_images.zip
    fi

    # GUIEnv (5.9 GB)
    if [ -f "GUIEnv_images.zip" ] && [ ! -d "GUIEnv/guienvs/images" ]; then
        echo "  Extracting GUIEnv..."
        mkdir -p GUIEnv
        unzip -q -o GUIEnv_images.zip -d GUIEnv/
        rm -f GUIEnv_images.zip
    fi

    # Wave-UI (24.4 GB)
    if [ -f "Wave-UI_images.zip" ] && [ ! -d "Wave-UI/images_fixed" ]; then
        echo "  Extracting Wave-UI..."
        mkdir -p Wave-UI
        unzip -q -o Wave-UI_images.zip -d Wave-UI/
        rm -f Wave-UI_images.zip
    fi

    # AndroidControl (49.3 GB)
    if [ -f "AndroidControl_images.zip" ] && [ ! -d "AndroidControl/tfrecord/images" ]; then
        echo "  Extracting AndroidControl..."
        mkdir -p AndroidControl
        unzip -q -o AndroidControl_images.zip -d AndroidControl/
        rm -f AndroidControl_images.zip
    fi

    # AMEX (92 GB, 3 split parts → cat → zip)
    if [ -f "amex_images_part_aa" ] && [ ! -d "AMEX/screenshots" ]; then
        echo "  Extracting AMEX (combining 3 parts)..."
        cat amex_images_part_* > amex_images_combined.zip
        mkdir -p AMEX
        7z x amex_images_combined.zip -aoa -oAMEX/ || unzip -q -o amex_images_combined.zip -d AMEX/
        rm -f amex_images_part_* amex_images_combined.zip
    fi

    # UGround (256 GB, 6 zip split volumes: .zip + .z01-.z05)
    # Use 7z directly on the main volume — do NOT cat/merge
    if [ -f "Uground_images_split.zip" ] && [ ! -d "Uground/images" ]; then
        echo "  Extracting UGround (6-part zip split volume)..."
        mkdir -p Uground
        7z x Uground_images_split.zip -aoa -oUground/
        # Inner zip: if 7z extracted a nested Uground_images.zip, unzip it
        if [ -f "Uground/Uground_images.zip" ]; then
            cd Uground && 7z x Uground_images.zip -aoa && rm -f Uground_images.zip && cd ..
        fi
        rm -f Uground_images_split.z* Uground_images_split.zip
    fi

    # ScreenSpot-Pro eval data
    echo "  Downloading ScreenSpot-Pro eval data..."
    if [ ! -d "ScreenSpot-Pro" ]; then
        python -c "
import os
os.environ.setdefault('HF_ENDPOINT', '$HF_ENDPOINT')
from huggingface_hub import snapshot_download
snapshot_download('liuzhch/ScreenSpot-Pro', local_dir='$DATA_DIR/ScreenSpot-Pro', repo_type='dataset')
print('  Done.')
" 2>/dev/null || echo "  (ScreenSpot-Pro download skipped)"
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
