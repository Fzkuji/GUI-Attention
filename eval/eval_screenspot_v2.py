"""
ScreenSpot v2 evaluation using saccade foveation.

Dataset: likaixin/ScreenSpot-v2 on HuggingFace
Extended from v1 with more platforms and instruction styles.

Usage:
  # Single GPU
  python eval/eval_screenspot_v2.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --rounds 3

  # Multi-GPU DDP
  torchrun --nproc_per_node=8 eval/eval_screenspot_v2.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --rounds 3
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.inference import run_saccade_inference
from gui_attention.model import Qwen25VLWithActionHead

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ScreenSpot-v2 platform → domain mapping
DOMAIN_MAP = {
    "windows": "desktop", "macos": "desktop", "linux": "desktop",
    "ios": "mobile", "android": "mobile",
    "web": "web",
}


def do_boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True, f"cuda:{local_rank}"
    return 0, 1, False, None


def gather_results(local_results, rank, world_size):
    """Gather list of dicts from all ranks to rank 0."""
    local_bytes = pickle.dumps(local_results)
    local_tensor = torch.ByteTensor(list(local_bytes)).cuda()
    size_tensor = torch.tensor([len(local_bytes)], dtype=torch.long, device="cuda")

    all_sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
    dist.all_gather(all_sizes, size_tensor)

    max_size = max(s.item() for s in all_sizes)
    padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
    padded[:len(local_bytes)] = local_tensor

    all_padded = [torch.zeros(max_size, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
    dist.all_gather(all_padded, padded)

    if rank == 0:
        all_results = []
        for i in range(world_size):
            sz = all_sizes[i].item()
            data = bytes(all_padded[i][:sz].cpu().tolist())
            all_results.extend(pickle.loads(data))
        return all_results
    return []


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

    for i, example in tqdm(enumerate(dataset), total=len(dataset), disable=(args._rank != 0)):
        # Detect field names (v2 may use different names)
        # Try v2 fields first, fall back to v1
        instruction = example.get("instruction", example.get("prompt", ""))
        data_type = example.get("data_type", example.get("ui_type", "text"))
        data_source = example.get("data_source", example.get("platform", "web"))
        file_name = example.get("file_name", example.get("img_filename", f"sample_{i}"))

        # Normalize domain
        domain = DOMAIN_MAP.get(data_source.lower().split("-")[0] if data_source else "web", "web")

        # Skip negative GT samples (v2 has gt_type field)
        gt_type = example.get("gt_type", "positive")
        if gt_type == "negative":
            continue

        ele = {
            "file_name": file_name,
            "ui_type": data_type,
            "domain": domain,
            "data_source": data_source,
            "instruction": instruction,
            "hit_top1": 0,
            "overlap_top1": 0,
        }

        # Get bbox — v2 uses [x1, y1, x2, y2] in pixels, normalized [0,1]
        bbox = example.get("bbox", None)
        if bbox is None:
            results.append(ele)
            continue

        bbox = list(bbox)
        ele["bbox_x1y1x2y2"] = bbox

        # Get image
        image = example["image"].convert("RGB")
        w, h = image.size

        # Normalize bbox if in pixel coords (> 1.0 means pixels)
        if any(v > 1.0 for v in bbox):
            gt_bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
        else:
            gt_bbox = bbox

        # Save temp image for inference
        tmp_path = os.path.join(tmp_dir, f"tmp_{args._rank}_{i}.png")
        image.save(tmp_path)

        t_start = time.time()
        pred = run_saccade_inference(
            image, tmp_path, instruction,
            model, tokenizer, builder,
            max_rounds=args.rounds,
            crop_ratio=args.crop_ratio,
            crop_size=args.crop_size,
            crop_upscale=args.crop_upscale,
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
    parser = argparse.ArgumentParser(description="ScreenSpot v2 eval (saccade foveation)")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")

    # Data
    parser.add_argument("--dataset_name", type=str, default="likaixin/ScreenSpot-v2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_path", type=str, default=None)

    # Saccade
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--crop_ratio", type=float, default=0.0,
                        help="(Legacy) Fraction crop. 0=use crop_size.")
    parser.add_argument("--crop_size", type=int, default=252,
                        help="Fixed crop side length in pixels")
    parser.add_argument("--crop_upscale", type=int, default=3,
                        help="Integer upscale factor for crop")

    # Resolution
    parser.add_argument("--low_res_max_pixels", type=int, default=400000)

    # Misc
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # DDP setup
    rank, world_size, is_distributed, ddp_device = setup_distributed()
    if ddp_device is not None:
        args.device = ddp_device
    args._rank = rank
    is_main = (rank == 0)

    # Auto save path
    if args.save_path is None:
        ckpt_name = Path(args.checkpoint).name
        if args.crop_size > 0:
            tag = f"saccade_r{args.rounds}_crop{args.crop_size}x{args.crop_upscale}"
        else:
            tag = f"saccade_r{args.rounds}_crop{args.crop_ratio}"
        args.save_path = f"results/screenspot_v2/{ckpt_name}/{tag}"

    if is_main:
        print("=== ScreenSpot v2 Eval ===")
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  base_model:  {args.base_model}")
        print(f"  rounds:      {args.rounds}")
        if args.crop_size > 0:
            up = args.crop_size * args.crop_upscale
            print(f"  crop:        {args.crop_size}x{args.crop_size} x{args.crop_upscale} -> {up}x{up}")
        else:
            print(f"  crop_ratio:  {args.crop_ratio}")
        print(f"  low_res:     {args.low_res_max_pixels}")
        print(f"  device:      {args.device}")
        if is_distributed:
            print(f"  DDP:         {world_size} GPUs")
        print()

    # Load data from HuggingFace
    if is_main:
        print(f"Loading dataset: {args.dataset_name} (split={args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    if is_main:
        print(f"Loaded {len(dataset)} examples")
        # Print column names for debugging
        print(f"Columns: {dataset.column_names}")

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        if is_main:
            print(f"Limiting to {len(dataset)} samples")

    # Shard data across ranks (pad to equal length to avoid DDP barrier hang)
    total_samples = len(dataset)
    if is_distributed:
        indices = list(range(rank, total_samples, world_size))
        # Pad shorter shards
        max_shard = (total_samples + world_size - 1) // world_size
        while len(indices) < max_shard:
            indices.append(indices[-1])
        dataset = dataset.select(indices)
        if is_main:
            print(f"Rank 0 processing {len(dataset)} samples (total {total_samples} split across {world_size} GPUs)")

    # Load model
    if is_main:
        print(f"Loading model: {args.checkpoint} (base: {args.base_model})")
    model, tokenizer = Qwen25VLWithActionHead.load_pretrained(
        args.checkpoint,
        base_model_name_or_path=args.base_model,
        device=args.device,
    )
    model.eval()

    crop_pixels = (args.crop_size * args.crop_upscale) ** 2 if args.crop_size > 0 else 200704
    builder = MultiRoundInputBuilder(
        args.base_model, tokenizer, min_pixels=3136,
        low_res_max_pixels=args.low_res_max_pixels,
        high_res_max_pixels=crop_pixels,
    )

    os.makedirs(args.save_path, exist_ok=True)
    ckpt_name = Path(args.checkpoint).name
    pred_path = os.path.join(args.save_path, f"{ckpt_name}_preds.json")
    metric_path = os.path.join(args.save_path, f"{ckpt_name}_metric.txt")

    if os.path.exists(pred_path) and is_main:
        print(f"Loading cached predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        t0 = time.time()
        with torch.no_grad():
            local_results = evaluate_all(model, tokenizer, dataset, args, builder)

        # Gather results from all ranks
        if is_distributed:
            dist.barrier()
            results = gather_results(local_results, rank, world_size)
        else:
            results = local_results

        if is_main:
            elapsed = time.time() - t0
            print(f"Evaluation took {elapsed:.1f}s ({elapsed / max(len(results), 1):.2f}s/sample)")

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

    if is_main:
        if not os.path.exists(metric_path):
            print("\n=== Metrics ===")
            metric_info = get_metric(results)
            with open(metric_path, "w") as f:
                f.write(metric_info)
            print(f"\nSaved metrics to {metric_path}")
        else:
            print(f"Metrics already exist at {metric_path}")
            with open(metric_path, "r") as f:
                print(f.read())

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
