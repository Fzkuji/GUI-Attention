"""
ScreenSpot-Pro evaluation using saccade foveation (v4).

Uses Qwen2.5-VL + LoRA + ActionHead with multi-round saccade inference.
No gui_aima dependency.

Usage:
  # Multi-round saccade eval
  python eval/eval_screenspot_pro_aligned.py \
      --checkpoint /path/to/checkpoint \
      --base_model Qwen/Qwen2.5-VL-3B-Instruct \
      --rounds 3 --crop_ratio 0.3

  # Single-round eval (no saccade)
  python eval/eval_screenspot_pro_aligned.py \
      --checkpoint /path/to/checkpoint \
      --base_model Qwen/Qwen2.5-VL-3B-Instruct \
      --rounds 1
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.constants import HIGH_RES_MAX_PIXELS, LOW_RES_MAX_PIXELS
from gui_attention.inference import run_saccade_inference
from gui_attention.model import Qwen25VLWithActionHead

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def do_boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    return (x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_metric(list_of_examples,
               groups=["Dev", "Creative", "CAD", "Scientific", "Office", "OS"],
               ui_types=["text", "icon"]):
    metrics = ["hit_top1", "overlap_top1"]

    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(example.get(key, 0) for example in examples) / len(examples)

    results = {metric: {} for metric in metrics}

    for group in groups:
        group_examples = [ex for ex in list_of_examples if ex.get("group") == group]
        for ui in ui_types:
            group_ui_examples = [ex for ex in group_examples if ex.get("ui_type") == ui]
            col_name = f"{group}-{ui}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(group_ui_examples, metric)
        col_name_avg = f"{group}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(group_examples, metric)

    for ui in ui_types:
        ui_examples = [ex for ex in list_of_examples if ex.get("ui_type") == ui]
        col_name = f"All-{ui}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(ui_examples, metric)

    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)

    columns_order = []
    for group in groups:
        for ui in ui_types:
            columns_order.append(f"{group}-{ui}")
        columns_order.append(f"{group}-avg")
    for ui in ui_types:
        columns_order.append(f"All-{ui}")
    columns_order.append("All-avg")

    header = [""] + columns_order
    col_widths = [max(len(col), 12) for col in header]

    def format_cell(cell):
        if isinstance(cell, float):
            return f"{cell * 100:.2f}"
        elif cell is None:
            return "N/A"
        return str(cell)

    header_line = " | ".join(word.ljust(width) for word, width in zip(header, col_widths))
    separator_line = "-+-".join("-" * width for width in col_widths)
    print(header_line)
    print(separator_line)

    for metric in metrics:
        row = [metric]
        for col in columns_order:
            val = results[metric].get(col)
            row.append(format_cell(val))
        row_line = " | ".join(word.ljust(width) for word, width in zip(row, col_widths))
        print(row_line)

    metric_info = "Tab-delimited Table for Excel:\n"
    header_tab = "\t".join([""] + columns_order)
    metric_info += header_tab + "\n"
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(model, tokenizer, data, image_dir, args, builder):
    device = next(model.parameters()).device
    results = []

    for i, example in tqdm(enumerate(data), total=len(data)):
        ele = {
            "file_name": example["img_filename"],
            "ui_type": example["ui_type"],
            "group": example["group"],
            "platform": example["platform"],
            "application": example["application"],
            "id": example["id"],
            "instruction": example["instruction"],
            "img_size": example["img_size"],
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["img_size"][0], example["img_size"][1]),
            "hit_top1": 0,
            "overlap_top1": 0,
        }

        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path).convert("RGB")
        gt_bbox = ele["bbox_x1y1x2y2"]

        t_start = time.time()
        pred = run_saccade_inference(
            image, image_path, example["instruction"],
            model, tokenizer, builder,
            max_rounds=args.rounds, crop_ratio=args.crop_ratio,
            device=str(device),
        )
        t_elapsed = time.time() - t_start

        topk_points = pred["topk_points"]
        if not topk_points:
            results.append(ele)
            continue

        n_w = pred.get("n_width", 1)
        n_h = pred.get("n_height", 1)
        IMAGE_PATCH_SIZE_x = 0.5 / max(n_w, 1)
        IMAGE_PATCH_SIZE_y = 0.5 / max(n_h, 1)

        x1, y1, x2, y2 = gt_bbox
        px, py = topk_points[0]

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1

        pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y,
                     px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1

        ele["pred_x"] = px
        ele["pred_y"] = py
        ele["num_rounds"] = pred.get("num_rounds", 1)
        ele["total_vis_tokens"] = pred.get("total_vis_tokens", 0)
        ele["inference_time"] = t_elapsed

        results.append(ele)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ScreenSpot-Pro eval (saccade foveation v4)")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (LoRA adapter + action head)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base model for loading backbone")

    # Data
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data/ScreenSpot-Pro")
    parser.add_argument("--save_path", type=str, default=None)

    # Saccade
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--crop_ratio", type=float, default=0.3)

    # Resolution
    parser.add_argument("--low_res_max_pixels", type=int, default=LOW_RES_MAX_PIXELS)
    parser.add_argument("--high_res_max_pixels", type=int, default=HIGH_RES_MAX_PIXELS)

    # Misc
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Auto save path
    if args.save_path is None:
        ckpt_name = Path(args.checkpoint).name
        tag = f"saccade_r{args.rounds}_crop{args.crop_ratio}"
        args.save_path = f"/root/autodl-tmp/results/screenspot_pro/{ckpt_name}/{tag}"

    print("=== Config ===")
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  base_model:  {args.base_model}")
    print(f"  rounds:      {args.rounds}")
    print(f"  crop_ratio:  {args.crop_ratio}")
    print(f"  device:      {args.device}")
    print()

    # Load data
    data_fn = os.path.join(args.data_path, "annotations/all.json")
    with open(data_fn, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_fn}")

    if args.max_samples is not None:
        data = data[:args.max_samples]
        print(f"Limiting to {len(data)} samples")

    image_dir = os.path.join(args.data_path, "images")

    # Load model
    print(f"Loading model: {args.checkpoint} (base: {args.base_model})")
    model, tokenizer = Qwen25VLWithActionHead.load_pretrained(
        args.checkpoint,
        base_model_name_or_path=args.base_model,
        device=args.device,
    )
    model.eval()

    # Builder
    builder = MultiRoundInputBuilder(
        args.base_model, tokenizer, min_pixels=3136,
        low_res_max_pixels=args.low_res_max_pixels,
        high_res_max_pixels=args.high_res_max_pixels,
    )

    # Check cache
    os.makedirs(args.save_path, exist_ok=True)
    ckpt_name = Path(args.checkpoint).name
    pred_path = os.path.join(args.save_path, f"{ckpt_name}_preds.json")
    metric_path = os.path.join(args.save_path, f"{ckpt_name}_metric.txt")

    if os.path.exists(pred_path):
        print(f"Loading cached predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        t0 = time.time()
        with torch.no_grad():
            results = evaluate_all(model, tokenizer, data, image_dir, args, builder)
        elapsed = time.time() - t0
        print(f"Evaluation took {elapsed:.1f}s ({elapsed / len(results):.2f}s/sample)")

        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {pred_path}")

        round_counts = {}
        for r in results:
            nr = r.get("num_rounds", 1)
            round_counts[nr] = round_counts.get(nr, 0) + 1
        print(f"\nRound distribution: {dict(sorted(round_counts.items()))}")

        # Token usage stats
        token_counts = [r.get("total_vis_tokens", 0) for r in results if r.get("total_vis_tokens", 0) > 0]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            print(f"Avg visual tokens (final round): {avg_tokens:.0f}")

        # Inference time stats
        times = [r.get("inference_time", 0) for r in results if r.get("inference_time", 0) > 0]
        if times:
            avg_time = sum(times) / len(times)
            print(f"Avg inference time: {avg_time:.3f}s/sample")
            print(f"Min/Max inference time: {min(times):.3f}s / {max(times):.3f}s")

    # Compute metrics
    if not os.path.exists(metric_path):
        print("\n=== Metrics ===")
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metrics to {metric_path}")
    else:
        print(f"Metrics already exist at {metric_path}")
        with open(metric_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    main()
