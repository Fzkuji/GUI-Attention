"""
ScreenSpot-Pro Evaluation for GUI-Attention (FoveatedQwen25VL).

Evaluates grounding accuracy on ScreenSpot-Pro benchmark.
Predicted point inside GT bbox = correct.

Usage:
    python eval/eval_screenspot_pro.py \
        --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
        --data_path /root/autodl-tmp/data/ScreenSpot-Pro \
        --save_path /root/autodl-tmp/results/screenspot_pro \
        [--grounding_head_path /path/to/grounding_head.pt] \
        [--fixation center|random]
"""

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui_attention.model.foveated_qwen25vl import FoveatedQwen25VL


def normalize_bbox(bbox, img_width, img_height):
    """Normalize bbox to [0,1] if not already."""
    x1, y1, x2, y2 = bbox
    if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
        return x1, y1, x2, y2
    return x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height


def load_screenspot_pro_data(data_path):
    """Load ScreenSpot-Pro annotations from all JSON files."""
    annotations_dir = os.path.join(data_path, "annotations")
    
    # Try all.json first
    all_json = os.path.join(annotations_dir, "all.json")
    if os.path.exists(all_json):
        with open(all_json) as f:
            return json.load(f)
    
    # Otherwise merge all per-app JSONs
    data = []
    for json_file in sorted(glob.glob(os.path.join(annotations_dir, "*.json"))):
        with open(json_file) as f:
            items = json.load(f)
            data.extend(items)
    
    print(f"Loaded {len(data)} samples from {len(glob.glob(os.path.join(annotations_dir, '*.json')))} annotation files")
    return data


def evaluate(model, data, image_dir, fixation_mode="center"):
    """Run evaluation on ScreenSpot-Pro data."""
    results = []
    
    for i, example in tqdm(enumerate(data), total=len(data)):
        img_path = os.path.join(image_dir, example["img_filename"])
        if not os.path.exists(img_path):
            print(f"Warning: image not found: {img_path}, skipping")
            continue
        
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = example["img_size"]
        
        # Normalize GT bbox
        gt_bbox = normalize_bbox(example["bbox"], img_w, img_h)
        x1, y1, x2, y2 = gt_bbox
        
        # Set fixation point
        if fixation_mode == "center":
            fixation = (0.5, 0.5)
        elif fixation_mode == "random":
            import random
            fixation = (random.random(), random.random())
        elif fixation_mode == "gt":
            # Use GT center (oracle, for upper bound)
            fixation = ((x1 + x2) / 2, (y1 + y2) / 2)
        else:
            fixation = None  # model default (center)
        
        # Predict
        try:
            px, py = model.predict(image, example["instruction"], fixation=fixation)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            px, py = 0.5, 0.5
        
        # Check hit
        hit = 1 if (x1 <= px <= x2) and (y1 <= py <= y2) else 0
        
        result = {
            "id": example.get("id", f"sample_{i}"),
            "instruction": example["instruction"],
            "img_filename": example["img_filename"],
            "ui_type": example.get("ui_type", "unknown"),
            "group": example.get("group", "unknown"),
            "platform": example.get("platform", "unknown"),
            "application": example.get("application", "unknown"),
            "img_size": example["img_size"],
            "bbox_gt": list(gt_bbox),
            "pred_x": px,
            "pred_y": py,
            "hit": hit,
        }
        results.append(result)
    
    return results


def compute_metrics(results):
    """Compute per-group, per-ui_type, and overall metrics."""
    groups = ["Dev", "Creative", "CAD", "Scientific", "Office", "OS"]
    ui_types = ["text", "icon"]
    
    def acc(examples):
        if not examples:
            return None
        return sum(e["hit"] for e in examples) / len(examples)
    
    print("\n" + "=" * 80)
    print("ScreenSpot-Pro Results")
    print("=" * 80)
    
    # Per group x ui_type
    header = ["Group"] + ui_types + ["avg"]
    print(f"\n{'Group':<15} {'text':>8} {'icon':>8} {'avg':>8} {'count':>8}")
    print("-" * 50)
    
    all_results_by_ui = {ui: [] for ui in ui_types}
    
    for group in groups:
        g_examples = [r for r in results if r["group"] == group]
        vals = []
        for ui in ui_types:
            gu_examples = [r for r in g_examples if r["ui_type"] == ui]
            all_results_by_ui[ui].extend(gu_examples)
            a = acc(gu_examples)
            vals.append(a)
        g_acc = acc(g_examples)
        
        text_str = f"{vals[0]*100:.1f}" if vals[0] is not None else "N/A"
        icon_str = f"{vals[1]*100:.1f}" if vals[1] is not None else "N/A"
        avg_str = f"{g_acc*100:.1f}" if g_acc is not None else "N/A"
        print(f"{group:<15} {text_str:>8} {icon_str:>8} {avg_str:>8} {len(g_examples):>8}")
    
    # Overall
    print("-" * 50)
    overall = acc(results)
    text_acc = acc(all_results_by_ui["text"])
    icon_acc = acc(all_results_by_ui["icon"])
    text_str = f"{text_acc*100:.1f}" if text_acc is not None else "N/A"
    icon_str = f"{icon_acc*100:.1f}" if icon_acc is not None else "N/A"
    overall_str = f"{overall*100:.1f}" if overall is not None else "N/A"
    print(f"{'Overall':<15} {text_str:>8} {icon_str:>8} {overall_str:>8} {len(results):>8}")
    print("=" * 80)
    
    return {
        "overall": overall,
        "text": text_acc,
        "icon": icon_acc,
        "total_samples": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="ScreenSpot-Pro Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ScreenSpot-Pro dataset (contains annotations/ and images/)")
    parser.add_argument("--save_path", type=str, default="results/screenspot_pro")
    parser.add_argument("--grounding_head_path", type=str, default=None)
    parser.add_argument("--fixation", type=str, default="center",
                        choices=["center", "random", "gt", "none"])
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ScreenSpot-Pro data from {args.data_path}...")
    data = load_screenspot_pro_data(args.data_path)
    image_dir = os.path.join(args.data_path, "images")
    print(f"Total samples: {len(data)}")
    
    # Load model
    print(f"\nLoading FoveatedQwen25VL from {args.model_path}...")
    t0 = time.time()
    model = FoveatedQwen25VL(model_name_or_path=args.model_path)
    
    if args.grounding_head_path:
        state_dict = torch.load(args.grounding_head_path, map_location="cpu")
        model.grounding_head.load_state_dict(state_dict)
        print(f"Loaded grounding head from {args.grounding_head_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}")
    
    # Evaluate
    print(f"\nRunning evaluation (fixation={args.fixation})...")
    results = evaluate(model, data, image_dir, fixation_mode=args.fixation)
    
    # Compute and print metrics
    metrics = compute_metrics(results)
    
    # Save results
    os.makedirs(args.save_path, exist_ok=True)
    
    suffix = ""
    if args.grounding_head_path:
        suffix = "_trained"
    else:
        suffix = "_baseline"
    
    pred_path = os.path.join(args.save_path, f"predictions{suffix}.json")
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions to {pred_path}")
    
    metric_path = os.path.join(args.save_path, f"metrics{suffix}.json")
    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metric_path}")


if __name__ == "__main__":
    main()
