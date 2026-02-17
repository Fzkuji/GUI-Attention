"""
Convert ScreenSpot-Pro data to GTA training format for GUI-Attention.

Creates a GTA-format JSON where each sample has:
- image: relative path to screenshot
- conversations: [human instruction, gpt response with coordinates and bbox]

Usage:
    python scripts/convert_screenspot_to_gta.py \
        --data_path /root/autodl-tmp/data/ScreenSpot-Pro \
        --output_path /root/autodl-tmp/data/screenspot_pro_train.json \
        --split_ratio 0.8
"""

import argparse
import json
import glob
import os
import random


def convert_screenspot_to_gta(data_path, output_train, output_val, split_ratio=0.8):
    """Convert ScreenSpot-Pro annotations to GTA training format."""
    annotations_dir = os.path.join(data_path, "annotations")
    
    # Load all annotations
    all_data = []
    for json_file in sorted(glob.glob(os.path.join(annotations_dir, "*.json"))):
        if os.path.basename(json_file) == "all.json":
            continue
        with open(json_file) as f:
            items = json.load(f)
            all_data.extend(items)
    
    print(f"Loaded {len(all_data)} total samples")
    
    # Convert to GTA format
    gta_samples = []
    for item in all_data:
        img_w, img_h = item["img_size"]
        x1, y1, x2, y2 = item["bbox"]
        
        # Normalize bbox to [0,1]
        nx1 = x1 / img_w
        ny1 = y1 / img_h
        nx2 = x2 / img_w
        ny2 = y2 / img_h
        
        # Click center
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        
        gta_sample = {
            "image": item["img_filename"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image> {item['instruction']}"
                },
                {
                    "from": "gpt",
                    "value": f"pyautogui.click(x={cx:.4f}, y={cy:.4f})",
                    "bbox_gt": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)]
                }
            ]
        }
        gta_samples.append(gta_sample)
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(gta_samples)
    
    split_idx = int(len(gta_samples) * split_ratio)
    train_data = gta_samples[:split_idx]
    val_data = gta_samples[split_idx:]
    
    # Save
    os.makedirs(os.path.dirname(output_train) or ".", exist_ok=True)
    with open(output_train, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved {len(train_data)} training samples to {output_train}")
    
    if output_val:
        os.makedirs(os.path.dirname(output_val) or ".", exist_ok=True)
        with open(output_val, "w") as f:
            json.dump(val_data, f, indent=2)
        print(f"Saved {len(val_data)} validation samples to {output_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_train", type=str, default="/root/autodl-tmp/data/screenspot_pro_train.json")
    parser.add_argument("--output_val", type=str, default="/root/autodl-tmp/data/screenspot_pro_val.json")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    args = parser.parse_args()
    
    convert_screenspot_to_gta(args.data_path, args.output_train, args.output_val, args.split_ratio)
