#!/bin/bash
# Download GUI-Actor training data from HuggingFace
# Total: ~438 GB (need ~500 GB free for extraction)
# Usage: bash jobs/download_gui_actor_data.sh [DATA_ROOT]

set -e

DATA_ROOT=${1:-"/home/zichuanfu2/data/GUI-Actor"}
HF_REPO="https://huggingface.co/datasets/cckevinn/GUI-Actor-Data/resolve/main"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "=== Downloading GUI-Actor Data to $DATA_ROOT ==="
echo ""

# 1. Download all JSON annotation files first (small)
echo ">>> Downloading JSON annotations..."
for f in guiact_bbox.json guienv_bbox.json amex_bbox.json androidcontrol_bbox.json wave_ui_bbox.json uground_bbox.json; do
    if [ ! -f "$f" ]; then
        echo "  Downloading $f..."
        wget -q --show-progress "$HF_REPO/$f" -O "$f"
    else
        echo "  $f already exists, skipping"
    fi
done
echo ""

# 2. GUIAct (4 GB) - already have images
if [ -d "GUIAct/web_imgs" ] && [ "$(ls GUIAct/web_imgs/*.png 2>/dev/null | head -1)" ]; then
    echo ">>> GUIAct images already exist, skipping"
else
    echo ">>> Downloading GUIAct images (4 GB)..."
    wget -q --show-progress "$HF_REPO/GUIAct_images.zip" -O GUIAct_images.zip
    unzip -q -o GUIAct_images.zip -d GUIAct/
    rm GUIAct_images.zip
fi
echo ""

# 3. GUIEnv (5.9 GB)
if [ -d "GUIEnv" ] && [ "$(ls GUIEnv/guienvs/images/*.png 2>/dev/null | head -1)" ]; then
    echo ">>> GUIEnv images already exist, skipping"
else
    echo ">>> Downloading GUIEnv images (5.9 GB)..."
    wget -q --show-progress "$HF_REPO/GUIEnv_images.zip" -O GUIEnv_images.zip
    mkdir -p GUIEnv
    unzip -q -o GUIEnv_images.zip -d GUIEnv/
    rm GUIEnv_images.zip
fi
echo ""

# 4. Wave-UI (24.4 GB)
if [ -d "Wave-UI" ] && [ "$(ls Wave-UI/images_fixed/*.png 2>/dev/null | head -1)" ]; then
    echo ">>> Wave-UI images already exist, skipping"
else
    echo ">>> Downloading Wave-UI images (24.4 GB)..."
    wget -q --show-progress "$HF_REPO/Wave-UI_images.zip" -O Wave-UI_images.zip
    mkdir -p Wave-UI
    unzip -q -o Wave-UI_images.zip -d Wave-UI/
    rm Wave-UI_images.zip
fi
echo ""

# 5. AndroidControl (49.3 GB)
if [ -d "AndroidControl" ] && [ "$(ls AndroidControl/tfrecord/images/*.png 2>/dev/null | head -1)" ]; then
    echo ">>> AndroidControl images already exist, skipping"
else
    echo ">>> Downloading AndroidControl images (49.3 GB)..."
    wget -q --show-progress "$HF_REPO/AndroidControl_images.zip" -O AndroidControl_images.zip
    mkdir -p AndroidControl
    unzip -q -o AndroidControl_images.zip -d AndroidControl/
    rm AndroidControl_images.zip
fi
echo ""

# 6. AMEX (92 GB, split into 3 parts)
if [ -d "AMEX" ] && [ "$(ls AMEX/screenshots/*.jpg 2>/dev/null | head -1)" ]; then
    echo ">>> AMEX images already exist, skipping"
else
    echo ">>> Downloading AMEX images (92 GB, 3 parts)..."
    for part in amex_images_part_aa amex_images_part_ab amex_images_part_ac; do
        if [ ! -f "$part" ]; then
            wget -q --show-progress "$HF_REPO/$part" -O "$part"
        fi
    done
    echo "  Combining and extracting..."
    cat amex_images_part_* > amex_images.zip
    mkdir -p AMEX
    7z x amex_images.zip -aoa -oAMEX/
    rm amex_images_part_* amex_images.zip
fi
echo ""

# 7. UGround (256 GB, split into 6 parts) - download last since it's huge
if [ -d "Uground" ] && [ "$(ls Uground/images/*.png 2>/dev/null | head -1)" ]; then
    echo ">>> UGround images already exist, skipping"
else
    echo ">>> Downloading UGround images (256 GB, 6 parts)..."
    for part in Uground_images_split.z01 Uground_images_split.z02 Uground_images_split.z03 Uground_images_split.z04 Uground_images_split.z05 Uground_images_split.zip; do
        if [ ! -f "$part" ]; then
            wget -q --show-progress "$HF_REPO/$part" -O "$part"
        fi
    done
    echo "  Combining and extracting..."
    cat Uground_images_split.z* Uground_images_split.zip > Uground_images.zip
    mkdir -p Uground
    7z x Uground_images.zip -aoa -oUground/
    rm Uground_images_split.z* Uground_images_split.zip Uground_images.zip
fi
echo ""

echo "=== Download complete ==="
echo ""
echo "Dataset structure:"
echo "  $DATA_ROOT/"
echo "    guiact_bbox.json        + GUIAct/web_imgs/"
echo "    guienv_bbox.json        + GUIEnv/guienvs/images/"
echo "    amex_bbox.json          + AMEX/screenshots/"
echo "    androidcontrol_bbox.json + AndroidControl/tfrecord/images/"
echo "    wave_ui_bbox.json       + Wave-UI/images_fixed/"
echo "    uground_bbox.json       + Uground/images/"
