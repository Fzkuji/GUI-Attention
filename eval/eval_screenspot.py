"""
ScreenSpot (v1) evaluation using saccade foveation (v4).

Dataset: rootsautomation/ScreenSpot on HuggingFace
1,272 samples across mobile/desktop/web × text/icon.

Usage:
  python eval/eval_screenspot.py \
      --checkpoint /path/to/checkpoint \
      --base_model Qwen/Qwen2.5-VL-3B-Instruct \
      --rounds 3 --crop_ratio 0.3
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.constants import HIGH_RES_MAX_PIXELS, LOW_RES_MAX_PIXELS
from gui_attention.inference import run_saccade_inference
from gui_attention.model import Qwen25VLWithActionHead

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOMAIN_MAP = {
    "windows": "desktop", "macos": "desktop",
    "ios": "mobile", "android": "mobile",
    "web-gitlab": "web", "web-shop": "web", "web-forum": "web", "web-tool": "web",
}


def do_boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


# ---------------------------------------------------------------------------
# Metrics (mobile/desktop/web × text/icon)
# ---------------------------------------------------------------------------

def get_metric(list_of_examples,
               domains=["mobile", "desktop", "web"],
               ui_types=["text", "icon"]):
    metrics = ["hit_top1", "overlap_top1"]

    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(example.get(key, 0) for example in examples) / len(examples)

    results = {metric: {} for metric in metrics}

    for domain in domains:
        domain_examples = [ex for ex in list_of_examples if ex.get("domain") == domain]
        for ui in ui_types:
            domain_ui_examples = [ex for ex in domain_examples if ex.get("ui_type") == ui]
            col_name = f"{domain}-{ui}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(domain_ui_examples, metric)
        col_name_avg = f"{domain}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(domain_examples, metric)

    for ui in ui_types:
        ui_examples = [ex for ex in list_of_examples if ex.get("ui_type") == ui]
        col_name = f"All-{ui}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(ui_examples, metric)

    for metric in metrics:
        results[metric]["All-avg"] = compute_mean(list_of_examples, metric)

    columns_order = []
    for domain in domains:
        for ui in ui_types:
            columns_order.append(f"{domain}-{ui}")
        columns_order.append(f"{domain}-avg")
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

def evaluate_all(model, tokenizer, dataset, args, builder):
    device = next(model.parameters()).device
    results = []
    tmp_dir = os.path.join(args.save_path, "tmp_images")
    os.makedirs(tmp_dir, exist_ok=True)

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        data_source = example["data_source"]
        domain = DOMAIN_MAP.get(data_source, "web")

        ele = {
            "file_name": example["file_name"],
            "ui_type": example["data_type"],  # "text" or "icon"
            "domain": domain,
            "data_source": data_source,
            "instruction": example["instruction"],
            "bbox_x1y1x2y2": list(example["bbox"]),  # already normalized [0,1]
            "hit_top1": 0,
            "overlap_top1": 0,
        }

        image = example["image"].convert("RGB")
        gt_bbox = ele["bbox_x1y1x2y2"]

        # Save temp image for inference (run_saccade_inference needs a path)
        tmp_path = os.path.join(tmp_dir, f"tmp_{i}.png")
        image.save(tmp_path)

        t_start = time.time()
        pred = run_saccade_inference(
            image, tmp_path, example["instruction"],
            model, tokenizer, builder,
            max_rounds=args.rounds, crop_ratio=args.crop_ratio,
            crop_upsample_pixels=args.crop_upsample_pixels,
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
    parser = argparse.ArgumentParser(description="ScreenSpot v1 eval (saccade foveation v4)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")

    parser.add_argument("--dataset_name", type=str, default="rootsautomation/ScreenSpot")
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--crop_ratio", type=float, default=0.2)
    parser.add_argument("--crop_upsample_pixels", type=int, default=1003520)

    parser.add_argument("--low_res_max_pixels", type=int, default=LOW_RES_MAX_PIXELS)
    parser.add_argument("--high_res_max_pixels", type=int, default=HIGH_RES_MAX_PIXELS)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.save_path is None:
        ckpt_name = Path(args.checkpoint).name
        tag = f"saccade_r{args.rounds}_crop{args.crop_ratio}"
        args.save_path = f"results/screenspot_v1/{ckpt_name}/{tag}"

    print("=== Config ===")
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  base_model:  {args.base_model}")
    print(f"  rounds:      {args.rounds}")
    print(f"  crop_ratio:  {args.crop_ratio}")
    print(f"  device:      {args.device}")
    print()

    # Load data from HuggingFace
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="test")
    print(f"Loaded {len(dataset)} examples")

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limiting to {len(dataset)} samples")

    # Load model
    print(f"Loading model: {args.checkpoint} (base: {args.base_model})")
    model, tokenizer = Qwen25VLWithActionHead.load_pretrained(
        args.checkpoint,
        base_model_name_or_path=args.base_model,
        device=args.device,
    )
    model.eval()

    builder = MultiRoundInputBuilder(
        args.base_model, tokenizer, min_pixels=3136,
        low_res_max_pixels=args.low_res_max_pixels,
        high_res_max_pixels=args.high_res_max_pixels,
    )

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
            results = evaluate_all(model, tokenizer, dataset, args, builder)
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

        token_counts = [r.get("total_vis_tokens", 0) for r in results if r.get("total_vis_tokens", 0) > 0]
        if token_counts:
            print(f"Avg visual tokens: {sum(token_counts) / len(token_counts):.0f}")

        times = [r.get("inference_time", 0) for r in results if r.get("inference_time", 0) > 0]
        if times:
            print(f"Avg inference time: {sum(times) / len(times):.3f}s/sample")

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
