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

# Dataset paths — auto-detect available datasets
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
        [ -n "$DATA_PATHS" ] && { DATA_PATHS="$DATA_PATHS,"; IMAGE_FOLDERS="$IMAGE_FOLDERS,"; PER_DS_LIMITS="$PER_DS_LIMITS,"; }
        DATA_PATHS="$DATA_PATHS$DATA_DIR/$json_file"
        IMAGE_FOLDERS="$IMAGE_FOLDERS$DATA_DIR/$img_dir"
        PER_DS_LIMITS="$PER_DS_LIMITS$limit"
        echo "  ✓ $json_file (limit=$limit)"
    else
        echo "  ✗ $json_file (skipping)"
    fi
done

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

    mkdir -p "$save"

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
        # Try local path first, fallback to HF id
        AIMA_PATH="$MODEL_DIR/GUI-AIMA-3B"
        [ ! -d "$AIMA_PATH" ] && AIMA_PATH="smz8599/GUI-AIMA-3B"
        run_eval "gui_aima" "$AIMA_PATH" "$RESULT_DIR/baseline_guiaima_on_trainset"
        ;;
    gui_actor)
        ACTOR_PATH="$MODEL_DIR/GUI-Actor-3B-Qwen2.5-VL"
        [ ! -d "$ACTOR_PATH" ] && ACTOR_PATH="Jingwen-01/GUI-Actor-3B-Qwen2.5-VL"
        run_eval "gui_actor" "$ACTOR_PATH" "$RESULT_DIR/baseline_guiactor_on_trainset"
        ;;
    qwen_vanilla)
        QWEN_PATH="$MODEL_DIR/Qwen2.5-VL-3B-Instruct"
        [ ! -d "$QWEN_PATH" ] && QWEN_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
        run_eval "qwen_vanilla" "$QWEN_PATH" "$RESULT_DIR/baseline_qwen_vanilla_on_trainset"
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
