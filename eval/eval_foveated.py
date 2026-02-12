"""
Evaluate foveated inference vs standard inference on ScreenSpot-Pro.

Compares:
1. Standard GUI-AIMA: high-res single pass (~18K tokens)
2. Foveated 3-round: low-res → crop → refine (~700 tokens)
3. Foveated 1-round: low-res only (~200 tokens, lower bound)

Reports accuracy and token counts.
"""

import argparse
import json
import os
import sys
import glob
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/root/GUI-AIMA/src")

from gui_attention.foveated_inference import FoveatedInference, FoveationConfig


def load_screenspot_pro(data_path):
    """Load ScreenSpot-Pro annotations."""
    annotations_dir = os.path.join(data_path, "annotations")
    data = []
    for f in sorted(glob.glob(os.path.join(annotations_dir, "*.json"))):
        with open(f) as fh:
            data.extend(json.load(fh))
    return data


def normalize_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    if any(v > 1 for v in [x1, y1, x2, y2]):
        return x1/w, y1/h, x2/w, y2/h
    return x1, y1, x2, y2


def evaluate(model, data, image_dir, n_samples=None, verbose=False):
    results = []
    total_tokens = 0
    
    samples = data[:n_samples] if n_samples else data
    
    for i, ex in tqdm(enumerate(samples), total=len(samples)):
        img_path = os.path.join(image_dir, ex["img_filename"])
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).convert("RGB")
        iw, ih = ex["img_size"]
        bbox = normalize_bbox(ex["bbox"], iw, ih)
        x1, y1, x2, y2 = bbox
        gt_cx, gt_cy = (x1+x2)/2, (y1+y2)/2
        
        try:
            px, py = model.predict(image, ex["instruction"], verbose=verbose)
        except Exception as e:
            print(f"Error on {i}: {e}")
            px, py = 0.5, 0.5
        
        hit = 1 if (x1 <= px <= x2 and y1 <= py <= y2) else 0
        dist = ((px - gt_cx)**2 + (py - gt_cy)**2)**0.5
        
        results.append({
            "id": i,
            "instruction": ex["instruction"],
            "pred": (px, py),
            "gt_bbox": bbox,
            "hit": hit,
            "dist": dist,
        })
        
        if (i+1) % 50 == 0:
            acc = sum(r["hit"] for r in results) / len(results) * 100
            avg_dist = sum(r["dist"] for r in results) / len(results)
            print(f"  [{i+1}/{len(samples)}] acc={acc:.1f}% avg_dist={avg_dist:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to GUI-AIMA-3B checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ScreenSpot-Pro dataset")
    parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/results/foveated")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to evaluate (None = all)")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load data
    data = load_screenspot_pro(args.data_path)
    image_dir = os.path.join(args.data_path, "images")
    print(f"Loaded {len(data)} samples")
    
    # Foveated inference
    config = FoveationConfig(num_rounds=args.num_rounds)
    model = FoveatedInference(args.model_path, config)
    
    print(f"\n=== Foveated Inference ({args.num_rounds} rounds) ===")
    t0 = time.time()
    results = evaluate(model, data, image_dir, args.n_samples, args.verbose)
    elapsed = time.time() - t0
    
    # Report
    n = len(results)
    acc = sum(r["hit"] for r in results) / n * 100
    avg_dist = sum(r["dist"] for r in results) / n
    
    print(f"\n{'='*60}")
    print(f"Foveated ({args.num_rounds} rounds): {sum(r['hit'] for r in results)}/{n} ({acc:.1f}%)")
    print(f"Average distance: {avg_dist:.4f}")
    print(f"Time: {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")
    print(f"{'='*60}")
    
    # Save
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f"foveated_r{args.num_rounds}.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "avg_dist": avg_dist,
            "n_samples": n,
            "num_rounds": args.num_rounds,
            "time_seconds": elapsed,
            "results": [(r["hit"], r["dist"]) for r in results],
        }, f, indent=2)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
