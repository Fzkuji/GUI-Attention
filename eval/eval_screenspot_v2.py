"""
ScreenSpot v2 evaluation using saccade foveation.

Dataset: OS-Copilot/ScreenSpot-v2 on HuggingFace (manual JSON + zip format)
Fields: img_filename, bbox [x,y,w,h] in pixels, instruction, data_type (text/icon), data_source (windows/macos/ios/android/web)

Usage:
  # Auto-download from HF cache
  torchrun --nproc_per_node=8 eval/eval_screenspot_v2.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --rounds 3

  # Or specify local data dir
  python eval/eval_screenspot_v2.py \
      --data_dir /path/to/screenspot_v2_extracted/ \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --rounds 3
"""

import argparse
import glob
import json
import os
import pickle
import time
import zipfile
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.inference import run_saccade_inference
from gui_attention.model import Qwen25VLWithActionHead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN_MAP = {
    "windows": "desktop", "macos": "desktop", "linux": "desktop",
    "ios": "mobile", "android": "mobile",
    "web": "web",
}

JSON_FILES = {
    "mobile": "screenspot_mobile_v2.json",
    "desktop": "screenspot_desktop_v2.json",
    "web": "screenspot_web_v2.json",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_data_dir():
    """Find ScreenSpot-v2 data in HF cache."""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--OS-Copilot--ScreenSpot-v2")
    if not os.path.exists(cache_base):
        return None
    snapshots = glob.glob(os.path.join(cache_base, "snapshots", "*"))
    if snapshots:
        return snapshots[0]
    return None


def load_screenspot_v2(data_dir, image_dir=None):
    """Load all samples from ScreenSpot-v2 JSON files.
    
    Args:
        data_dir: Directory containing the 3 JSON files and optionally the zip
        image_dir: Directory with extracted images (if None, will extract from zip)
    
    Returns:
        list of dicts with keys: img_filename, bbox_x1y1x2y2, instruction, 
        data_type, data_source, domain, image_path
    """
    all_data = []
    
    for category, json_name in JSON_FILES.items():
        json_path = os.path.join(data_dir, json_name)
        if not os.path.exists(json_path):
            print(f"  Warning: {json_path} not found, skipping {category}")
            continue
        with open(json_path, "r") as f:
            items = json.load(f)
        print(f"  {category}: {len(items)} samples")
        for item in items:
            # Convert [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = item["bbox"]
            bbox_xyxy = [x, y, x + w, y + h]
            
            domain = DOMAIN_MAP.get(item.get("data_source", "").lower(), "web")
            
            all_data.append({
                "img_filename": item["img_filename"],
                "bbox_x1y1x2y2": bbox_xyxy,
                "instruction": item["instruction"],
                "data_type": item.get("data_type", "text"),
                "data_source": item.get("data_source", ""),
                "domain": domain,
                "ui_type": item.get("data_type", "text"),
            })
    
    # Find or extract images
    if image_dir is None:
        image_dir = os.path.join(data_dir, "screenspotv2_image")
    
    if not os.path.isdir(image_dir):
        zip_path = os.path.join(data_dir, "screenspotv2_image.zip")
        if os.path.exists(zip_path):
            print(f"  Extracting images from {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            print(f"  Extracted to {image_dir}")
        else:
            raise FileNotFoundError(f"No images found at {image_dir} and no zip at {zip_path}")
    
    # Set image paths
    for item in all_data:
        item["image_path"] = os.path.join(image_dir, item["img_filename"])
    
    return all_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Metrics
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

def evaluate_all(model, tokenizer, data, args, builder):
    device = next(model.parameters()).device
    results = []

    for i, example in tqdm(enumerate(data), total=len(data), disable=(args._rank != 0)):
        instruction = example["instruction"]
        
        ele = {
            "file_name": example["img_filename"],
            "ui_type": example["ui_type"],
            "domain": example["domain"],
            "data_source": example["data_source"],
            "instruction": instruction,
            "hit_top1": 0,
            "overlap_top1": 0,
        }

        bbox = example["bbox_x1y1x2y2"]
        image_path = example["image_path"]

        if not os.path.exists(image_path):
            results.append(ele)
            continue

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Normalize bbox to [0, 1]
        gt_bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

        t_start = time.time()
        pred = run_saccade_inference(
            image, image_path, instruction,
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
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with screenspot_*_v2.json + screenspotv2_image/. "
                             "If not set, auto-detect from HF cache.")
    parser.add_argument("--save_path", type=str, default=None)

    # Saccade
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--crop_ratio", type=float, default=0.0)
    parser.add_argument("--crop_size", type=int, default=252)
    parser.add_argument("--crop_upscale", type=int, default=3)

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
        tag = f"saccade_r{args.rounds}_crop{args.crop_size}x{args.crop_upscale}"
        args.save_path = f"results/screenspot_v2/{ckpt_name}/{tag}"

    if is_main:
        print("=== ScreenSpot v2 Eval ===")
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  base_model:  {args.base_model}")
        print(f"  rounds:      {args.rounds}")
        if args.crop_size > 0:
            up = args.crop_size * args.crop_upscale
            print(f"  crop:        {args.crop_size}x{args.crop_size} x{args.crop_upscale} -> {up}x{up}")
        print(f"  low_res:     {args.low_res_max_pixels}")
        print(f"  device:      {args.device}")
        if is_distributed:
            print(f"  DDP:         {world_size} GPUs")
        print()

    # Find data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = find_data_dir()
        if data_dir is None:
            # Download first with huggingface_hub
            if is_main:
                print("Downloading OS-Copilot/ScreenSpot-v2 from HuggingFace...")
                from huggingface_hub import snapshot_download
                data_dir = snapshot_download("OS-Copilot/ScreenSpot-v2", repo_type="dataset")
            if is_distributed:
                dist.barrier()
                if not is_main:
                    data_dir = find_data_dir()

    if is_main:
        print(f"Data dir: {data_dir}")

    # Load data
    if is_main:
        print("Loading ScreenSpot-v2 data...")
    data = load_screenspot_v2(data_dir)
    if is_main:
        print(f"Total: {len(data)} samples")

    if args.max_samples is not None:
        data = data[:args.max_samples]
        if is_main:
            print(f"Limiting to {len(data)} samples")

    # Shard data across ranks (pad to equal length)
    total_data = len(data)
    if is_distributed:
        data = data[rank::world_size]
        max_shard = (total_data + world_size - 1) // world_size
        while len(data) < max_shard:
            data.append(data[-1])
        if is_main:
            print(f"Rank 0 processing {len(data)} samples (total {total_data} split across {world_size} GPUs)")

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
            local_results = evaluate_all(model, tokenizer, data, args, builder)

        if is_distributed:
            if is_main:
                print(f"Rank 0 done, gathering results from {world_size} GPUs ...")
            dist.barrier()
            results = gather_results(local_results, rank, world_size)
        else:
            results = local_results

        if is_main:
            elapsed = time.time() - t0
            # Deduplicate padded samples
            seen = set()
            unique_results = []
            for r in results:
                key = r.get("file_name", "") + r.get("instruction", "")
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            results = unique_results

            print(f"Evaluation took {elapsed:.1f}s ({elapsed / max(len(results), 1):.2f}s/sample)")
            print(f"Total unique results: {len(results)}")

            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved predictions to {pred_path}")

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
