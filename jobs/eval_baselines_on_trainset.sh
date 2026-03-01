#!/bin/bash
# ============================================================================
# Evaluate baseline models (GUI-AIMA, GUI-Actor, Qwen-vanilla) on our
# training datasets to measure how well pretrained models already handle
# the training data.
#
# Usage:
#   bash jobs/eval_baselines_on_trainset.sh [gui_aima|gui_actor|qwen_vanilla|all]
#   MAX_SAMPLES=2000 bash jobs/eval_baselines_on_trainset.sh gui_aima
# ============================================================================

set -e

MODEL_TYPE="${1:-gui_aima}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
DEVICE="${DEVICE:-cuda:0}"

# Auto-detect workspace
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE_DIR=$(dirname "$SCRIPT_DIR")
WORK_DIR=$(dirname "$CODE_DIR")

DATA_DIR="$WORK_DIR/data"
MODEL_DIR="$WORK_DIR/models"
RESULT_DIR="$WORK_DIR/results"

# GUI-AIMA code path (for gui_aima package imports)
GUI_AIMA_CODE="${GUI_AIMA_CODE:-$WORK_DIR/GUI-AIMA}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE_DIR/src:$PYTHONPATH"

cd "$CODE_DIR"

# Dataset paths (same as v7 training)
DATA_PATHS="$DATA_DIR/guiact_bbox.json,$DATA_DIR/androidcontrol_bbox.json,$DATA_DIR/wave_ui_bbox.json,$DATA_DIR/uground_bbox.json,$DATA_DIR/gta_bbox.json"
IMAGE_FOLDERS="$DATA_DIR/GUIAct/GUIAct/web_imgs,$DATA_DIR/AndroidControl/AndroidControl/tfrecord/images,$DATA_DIR/Wave-UI/Wave-UI/images_fixed,$DATA_DIR/Uground/Uground/images,$DATA_DIR/GTA/images"
# UGround limited to first 60K, GTA limited to first 60K (matching training)
PER_DS_LIMITS="0,0,0,60000,60000"

run_eval() {
    local mtype="$1"
    local mpath="$2"
    local save="$3"
    local extra_args="${4:-}"

    echo "============================================================"
    echo "  Evaluating: $mtype"
    echo "  Model:      $mpath"
    echo "  Samples:    $MAX_SAMPLES"
    echo "  Save:       $save"
    echo "============================================================"

    python eval/eval_baseline_on_trainset.py \
        --model_type "$mtype" \
        --model_path "$mpath" \
        --gui_aima_code "$GUI_AIMA_CODE" \
        --data_path "$DATA_PATHS" \
        --image_folder "$IMAGE_FOLDERS" \
        --max_samples "$MAX_SAMPLES" \
        --max_samples_per_dataset "$PER_DS_LIMITS" \
        --save_path "$save" \
        --device "$DEVICE" \
        $extra_args \
        2>&1 | tee "$save/eval.log"
}

case "$MODEL_TYPE" in
    gui_aima)
        run_eval "gui_aima" \
            "$MODEL_DIR/GUI-AIMA-3B" \
            "$RESULT_DIR/baseline_guiaima_on_trainset"
        ;;
    gui_actor)
        run_eval "gui_actor" \
            "$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL" \
            "$RESULT_DIR/baseline_guiactor_on_trainset"
        ;;
    qwen_vanilla)
        run_eval "qwen_vanilla" \
            "$MODEL_DIR/Qwen2.5-VL-3B-Instruct" \
            "$RESULT_DIR/baseline_qwen_vanilla_on_trainset"
        ;;
    all)
        run_eval "gui_aima" \
            "$MODEL_DIR/GUI-AIMA-3B" \
            "$RESULT_DIR/baseline_guiaima_on_trainset"
        run_eval "gui_actor" \
            "$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL" \
            "$RESULT_DIR/baseline_guiactor_on_trainset"
        run_eval "qwen_vanilla" \
            "$MODEL_DIR/Qwen2.5-VL-3B-Instruct" \
            "$RESULT_DIR/baseline_qwen_vanilla_on_trainset"
        ;;
    *)
        echo "Usage: $0 [gui_aima|gui_actor|qwen_vanilla|all]"
        exit 1
        ;;
esac

echo "Done!"
