"""Extract a fixed 5000-sample subset from GUIAct for fair comparison."""
import json
import os
import random

random.seed(42)

DATA_PATH = "/home/zichuanfu2/data/GUI-Actor/guiact_bbox.json"
IMAGE_FOLDER = "/home/zichuanfu2/data/GUI-Actor/GUIAct/web_imgs"
OUTPUT_PATH = "/home/zichuanfu2/data/GUI-Actor/guiact_5k_seed42.json"
N = 5000

with open(DATA_PATH) as f:
    raw = json.load(f)

# Filter to samples that have valid image + bbox
valid = []
for item in raw:
    img_file = item["image"]
    if isinstance(img_file, list):
        img_file = img_file[0]
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    if not os.path.exists(img_path):
        continue
    has_bbox = any(c.get("bbox_gt") is not None for c in item.get("conversations", []))
    if has_bbox:
        valid.append(item)

print(f"Total valid samples: {len(valid)}")
random.shuffle(valid)
subset = valid[:N]

with open(OUTPUT_PATH, "w") as f:
    json.dump(subset, f, indent=2)
print(f"Saved {len(subset)} samples to {OUTPUT_PATH}")
