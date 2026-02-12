"""
ScreenSpot-Pro evaluation script for GUI-Attention project.
Fully aligned with GUI-AIMA original evaluation methodology.

Instead of fixed modes, evaluation is configured via composable parameters:
  --rounds          Number of inference rounds (1 = single-round)
  --multi_strategy  Multi-round strategy: "crop_resize" (GUI-AIMA) or "token_concat" (ours)
  --resolution      Per-round resolution: "high" or "low"
  --crop_ratio      Crop ratio for multi-round (default 0.3)

Examples:
  # GUI-AIMA standard single-round (high-res)
  python eval/eval_screenspot_pro.py --rounds 1 --resolution high

  # GUI-AIMA two-stage crop
  python eval/eval_screenspot_pro.py --rounds 2 --multi_strategy crop_resize

  # Our multi-round foveation (token concat), 5 rounds, low first round
  python eval/eval_screenspot_pro.py --rounds 5 --multi_strategy token_concat --resolution low

  # Single-round low-res
  python eval/eval_screenspot_pro.py --rounds 1 --resolution low
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference
from gui_aima.utils import do_boxes_overlap
from gui_aima.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)
from transformers import AutoProcessor, AutoTokenizer

# ---------------------------------------------------------------------------
# GUI-AIMA original system prompt (must match exactly)
# ---------------------------------------------------------------------------
GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human "
    "instruction, your task is to locate the screen element that corresponds "
    "to the instruction. You should output a PyAutoGUI action that performs a "
    "click on the correct position. To indicate the click location, we will "
    "use some special tokens, which is used to refer to a visual patch later. "
    "For example, you can output: pyautogui.click(<your_special_token_here>)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    """Normalize bbox to [0, 1] if not already."""
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    return (x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)


def crop_subimage(img_width, img_height, cx, cy, crop_size):
    """Crop a square region of crop_size around (cx, cy) in pixel coords.
    Returns (start_x, start_y, end_x, end_y)."""
    half = crop_size // 2
    start_x = max(0, cx - half)
    start_y = max(0, cy - half)
    end_x = min(img_width, start_x + crop_size)
    end_y = min(img_height, start_y + crop_size)
    if end_x - start_x < crop_size:
        start_x = max(0, end_x - crop_size)
    if end_y - start_y < crop_size:
        start_y = max(0, end_y - crop_size)
    return start_x, start_y, end_x, end_y


def crop_image_ratio(image, center_x, center_y, crop_ratio):
    """Crop a region around (center_x, center_y) in normalized [0,1] coords.
    Returns (cropped_image, crop_bbox_normalized)."""
    W, H = image.size
    crop_w = int(W * crop_ratio)
    crop_h = int(H * crop_ratio)
    cx_px = int(center_x * W)
    cy_px = int(center_y * H)
    x1 = max(0, cx_px - crop_w // 2)
    y1 = max(0, cy_px - crop_h // 2)
    x2 = min(W, x1 + crop_w)
    y2 = min(H, y1 + crop_h)
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)
    cropped = image.crop((x1, y1, x2, y2))
    bbox = (x1 / W, y1 / H, x2 / W, y2 / H)
    return cropped, bbox


# ---------------------------------------------------------------------------
# get_metric - copied from GUI-AIMA original
# ---------------------------------------------------------------------------

def get_metric(list_of_examples,
               groups=["Dev", "Creative", "CAD", "Scientific", "Office", "OS"],
               ui_types=["text", "icon"]):
    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk"]

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
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name_or_path, max_pixels, device="cuda:0"):
    """Load GUI-AIMA model and processor."""
    data_processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=max_pixels)
    tokenizer = data_processor.tokenizer

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    ).eval()

    # Ensure config attributes exist
    for attr, default in [
        ('query_topk', None), ('kl_query_weighting', False),
        ('part_query_weighting', False), ('layer_wise_query_weighting', False),
    ]:
        if not hasattr(model.config, attr):
            setattr(model.config, attr, default)

    if model.config.kl_query_weighting:
        print(f"Model: {model_name_or_path}, KL-weighting: True")
    elif model.config.query_topk is not None:
        print(f"Model: {model_name_or_path}, KL-weighting: False, Topk: {model.config.query_topk}")
    else:
        print(f"Model: {model_name_or_path}")

    return model, tokenizer, data_processor


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def make_conversation(image, instruction, system_message=GROUNDING_SYSTEM_MESSAGE):
    """Build single-image conversation for GUI-AIMA inference."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]},
    ]


def run_single_round(image, instruction, model, tokenizer, processor,
                     use_placeholder=True, topk=3, resize_to_pixels=None):
    """Single-round inference, optionally resizing the image first."""
    if resize_to_pixels is not None and resize_to_pixels > 0:
        w, h = image.size
        if w * h != resize_to_pixels:
            ratio = (resize_to_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * ratio), int(h * ratio)))

    conversation = make_conversation(image, instruction)
    pred = inference(conversation, model, tokenizer, processor,
                     logits_processor=None, use_placeholder=use_placeholder, topk=topk)
    return pred


def run_multi_crop_resize(image, instruction, model, tokenizer, processor,
                          use_placeholder=True, topk=3, rounds=2,
                          resize_to_pixels=None):
    """GUI-AIMA style multi-round: predict → crop around prediction → upscale → re-predict.
    Each subsequent round crops around the previous prediction."""
    img_w, img_h = image.size

    # Round 1
    pred = run_single_round(image, instruction, model, tokenizer, processor,
                            use_placeholder=use_placeholder, topk=topk,
                            resize_to_pixels=resize_to_pixels)
    if rounds <= 1:
        return pred

    cur_x, cur_y = pred["topk_points"][0]  # normalized [0,1]

    for r in range(1, rounds):
        # Compute crop size (GUI-AIMA formula)
        portion_size = 560 ** 2
        crop_size = int(((img_w * img_h * portion_size / 5760000) ** 0.5) / 28) * 28

        px_center = int(cur_x * img_w)
        py_center = int(cur_y * img_h)
        start_x, start_y, end_x, end_y = crop_subimage(img_w, img_h, px_center, py_center, crop_size)
        cropped = image.crop((start_x, start_y, end_x, end_y))
        cw, ch = cropped.size
        cropped_upscaled = cropped.resize((cw * 2, ch * 2), Image.BICUBIC)

        conversation = make_conversation(cropped_upscaled, instruction)
        pred_r = inference(conversation, model, tokenizer, processor,
                           logits_processor=None, use_placeholder=use_placeholder, topk=topk)

        # Map crop-local predictions back to original image coords
        cw2, ch2 = cropped_upscaled.size
        topk_mapped = []
        for (nx, ny) in pred_r["topk_points"]:
            px_orig = (nx * cw2) / 2 + start_x
            py_orig = (ny * ch2) / 2 + start_y
            topk_mapped.append((px_orig / img_w, py_orig / img_h))

        cur_x, cur_y = topk_mapped[0]
        pred = {
            "topk_points": topk_mapped,
            "n_width": pred_r["n_width"],
            "n_height": pred_r["n_height"],
        }

    return pred


def run_multi_token_concat(image, instruction, model, tokenizer, processor,
                           use_placeholder=True, topk=3, rounds=5,
                           crop_ratio=0.3, convergence_threshold=0.02,
                           resize_to_pixels=None):
    """Our multi-round foveation: predict → crop → feed (original + crop) → refine.
    Uses token concatenation instead of crop+resize replacement."""
    # Round 1
    pred = run_single_round(image, instruction, model, tokenizer, processor,
                            use_placeholder=use_placeholder, topk=topk,
                            resize_to_pixels=resize_to_pixels)
    if rounds <= 1:
        return pred

    prev_x, prev_y = pred["topk_points"][0]

    for r in range(1, rounds):
        cropped, crop_bbox = crop_image_ratio(image, prev_x, prev_y, crop_ratio)

        # Multi-image conversation: full image + cropped region
        content_items = [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
            {"type": "image", "image": cropped},
            {"type": "text", "text": (
                f"[Zoomed region around ({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})"
                f"-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"
            )},
        ]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_MESSAGE}]},
            {"role": "user", "content": content_items},
        ]

        try:
            pred_r = inference(conversation, model, tokenizer, processor,
                               logits_processor=None, use_placeholder=use_placeholder, topk=topk)
        except Exception as e:
            print(f"  Round {r + 1} error: {e}")
            break

        if not pred_r["topk_points"]:
            break

        # Map local crop coords back to global
        local_x, local_y = pred_r["topk_points"][0]
        bx1, by1, bx2, by2 = crop_bbox
        cur_x = bx1 + local_x * (bx2 - bx1)
        cur_y = by1 + local_y * (by2 - by1)

        dist = math.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
        prev_x, prev_y = cur_x, cur_y
        if dist < convergence_threshold:
            break

    return {
        "topk_points": [(prev_x, prev_y)],
        "n_width": pred["n_width"],
        "n_height": pred["n_height"],
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(model, tokenizer, processor, data, image_dir, args):
    """Run evaluation over all examples."""
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
            "hit_topk": 0,
            "overlap_topk": 0,
        }

        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path).convert("RGB")
        gt_bbox = ele["bbox_x1y1x2y2"]

        # --- Dispatch based on composable params ---
        if args.rounds <= 1 or args.multi_strategy == "none":
            # Single-round
            pred = run_single_round(
                image, example["instruction"], model, tokenizer, processor,
                use_placeholder=args.use_placeholder, topk=args.topk,
                resize_to_pixels=args.resize_to_pixels if args.resize_to_pixels > 0 else None,
            )
        elif args.multi_strategy == "crop_resize":
            pred = run_multi_crop_resize(
                image, example["instruction"], model, tokenizer, processor,
                use_placeholder=args.use_placeholder, topk=args.topk,
                rounds=args.rounds,
                resize_to_pixels=args.resize_to_pixels if args.resize_to_pixels > 0 else None,
            )
        elif args.multi_strategy == "token_concat":
            pred = run_multi_token_concat(
                image, example["instruction"], model, tokenizer, processor,
                use_placeholder=args.use_placeholder, topk=args.topk,
                rounds=args.rounds, crop_ratio=args.crop_ratio,
                convergence_threshold=args.convergence_threshold,
                resize_to_pixels=args.resize_to_pixels if args.resize_to_pixels > 0 else None,
            )
        else:
            raise ValueError(f"Unknown multi_strategy: {args.multi_strategy}")

        topk_points = pred["topk_points"]
        patch_w = pred["n_width"]
        patch_h = pred["n_height"]
        IMAGE_PATCH_SIZE_x = 0.5 / patch_w
        IMAGE_PATCH_SIZE_y = 0.5 / patch_h

        # Compute metrics
        x1, y1, x2, y2 = gt_bbox
        px, py = topk_points[0]

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

        pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y,
                     px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1

        for px_k, py_k in topk_points[1:]:
            if (x1 <= px_k <= x2) and (y1 <= py_k <= y2):
                ele["hit_topk"] = 1
            pred_bbox_k = [px_k - IMAGE_PATCH_SIZE_x, py_k - IMAGE_PATCH_SIZE_y,
                           px_k + IMAGE_PATCH_SIZE_x, py_k + IMAGE_PATCH_SIZE_y]
            if do_boxes_overlap(pred_bbox_k, gt_bbox):
                ele["overlap_topk"] = 1

        results.append(ele)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ScreenSpot-Pro evaluation (composable params)")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="smz8599/GUI-AIMA-3B")

    # Data
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data/ScreenSpot-Pro")
    parser.add_argument("--save_path", type=str, default=None)

    # Resolution control
    parser.add_argument("--resolution", type=str, default="high", choices=["high", "low"],
                        help="Resolution preset: 'high' (resize_to=5760000, max=14777616) "
                             "or 'low' (no resize, max=1003520)")
    parser.add_argument("--resize_to_pixels", type=int, default=None,
                        help="Override: resize images to this many pixels. -1 to disable.")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Override: max pixels for processor.")

    # Multi-round control
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of inference rounds (1 = single-round)")
    parser.add_argument("--multi_strategy", type=str, default="none",
                        choices=["none", "crop_resize", "token_concat"],
                        help="Multi-round strategy: crop_resize (GUI-AIMA) or token_concat (ours)")

    # Multi-round tuning
    parser.add_argument("--crop_ratio", type=float, default=0.3,
                        help="Crop ratio for token_concat multi-round")
    parser.add_argument("--convergence_threshold", type=float, default=0.02,
                        help="Early stop if prediction moves less than this between rounds")

    # Inference options
    parser.add_argument("--no-placeholder", dest="use_placeholder", action="store_false")
    parser.set_defaults(use_placeholder=True)
    parser.add_argument("--topk", type=int, default=3)

    # Quick test
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    # --- Resolve resolution presets (can be overridden) ---
    if args.resolution == "high":
        effective_resize = args.resize_to_pixels if args.resize_to_pixels is not None else 5760000
        effective_max = args.max_pixels if args.max_pixels is not None else 14777616
    else:  # low
        effective_resize = args.resize_to_pixels if args.resize_to_pixels is not None else -1
        effective_max = args.max_pixels if args.max_pixels is not None else 1003520

    args.resize_to_pixels = effective_resize
    args.max_pixels = effective_max

    # Auto-infer multi_strategy from rounds if not set
    if args.rounds > 1 and args.multi_strategy == "none":
        print(f"Warning: --rounds={args.rounds} but --multi_strategy=none. "
              f"Defaulting to crop_resize. Use --multi_strategy to specify.")
        args.multi_strategy = "crop_resize"

    # Auto save path
    if args.save_path is None:
        model_name = Path(args.model_name_or_path).name
        strategy_tag = f"r{args.rounds}_{args.multi_strategy}" if args.rounds > 1 else "single"
        res_tag = args.resolution
        args.save_path = f"/root/autodl-tmp/results/screenspot_pro/{model_name}/{strategy_tag}_{res_tag}"

    # Print config
    print(f"=== Config ===")
    print(f"  rounds:           {args.rounds}")
    print(f"  multi_strategy:   {args.multi_strategy}")
    print(f"  resolution:       {args.resolution}")
    print(f"  resize_to_pixels: {args.resize_to_pixels}")
    print(f"  max_pixels:       {args.max_pixels}")
    print(f"  use_placeholder:  {args.use_placeholder}")
    print(f"  topk:             {args.topk}")
    if args.rounds > 1:
        print(f"  crop_ratio:       {args.crop_ratio}")
        print(f"  convergence_thr:  {args.convergence_threshold}")
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
    print(f"Loading model with max_pixels={args.max_pixels}...")
    model, tokenizer, processor = load_model(
        args.model_name_or_path, max_pixels=args.max_pixels,
    )

    # Check for cached predictions
    os.makedirs(args.save_path, exist_ok=True)
    model_name = Path(args.model_name_or_path).name
    pred_path = os.path.join(args.save_path, f"{model_name}_preds.json")
    metric_path = os.path.join(args.save_path, f"{model_name}_metric.txt")

    if os.path.exists(pred_path):
        print(f"Loading cached predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        t0 = time.time()
        results = evaluate_all(model, tokenizer, processor, data, image_dir, args)
        elapsed = time.time() - t0
        print(f"Evaluation took {elapsed:.1f}s ({elapsed / len(results):.2f}s/sample)")

        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {pred_path}")

    # Compute and save metrics
    if not os.path.exists(metric_path):
        print(f"\n=== Metrics ===")
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
