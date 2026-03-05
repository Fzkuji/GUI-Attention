"""
Unified ScreenSpot evaluation: v1, v2, and Pro.

Supports multi-round saccade foveation with DDP.

Usage:
  # ScreenSpot v1 (HuggingFace)
  torchrun --nproc_per_node=8 eval/eval_screenspot.py \
      --dataset v1 \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct

  # ScreenSpot v2 (local JSON + zip)
  torchrun --nproc_per_node=8 eval/eval_screenspot.py \
      --dataset v2 --data_dir /path/to/v2_data \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct

  # ScreenSpot-Pro (local directory)
  torchrun --nproc_per_node=8 eval/eval_screenspot.py \
      --dataset pro --data_dir /path/to/ScreenSpot-Pro \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct
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
from gui_attention.model import Qwen25VLWithDualHead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN_MAP = {
    "windows": "desktop", "macos": "desktop", "linux": "desktop",
    "ios": "mobile", "android": "mobile",
    "web": "web",
}

# ScreenSpot-Pro: application → category mapping
APP_TO_CATEGORY = {
    # Dev
    "android_studio": "Dev", "pycharm": "Dev", "vscode": "Dev",
    "quartus": "Dev", "vivado": "Dev", "unreal_engine": "Dev",
    # Creative
    "photoshop": "Creative", "illustrator": "Creative", "premiere": "Creative",
    "blender": "Creative", "fruitloops": "Creative", "davinci": "Creative",
    # CAD
    "autocad": "CAD", "solidworks": "CAD", "inventor": "CAD",
    # Scientific
    "matlab": "Scientific", "origin": "Scientific", "eviews": "Scientific",
    "stata": "Scientific",
    # Office
    "excel": "Office", "word": "Office", "powerpoint": "Office",
    # OS
    "linux_common": "OS", "macos_common": "OS", "windows_common": "OS",
    "vmware": "OS",
}
PRO_CATEGORIES = ["Dev", "Creative", "CAD", "Scientific", "Office", "OS"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_screenspot_v1(data_dir=None):
    """Load ScreenSpot v1 from HuggingFace."""
    from datasets import load_dataset
    dataset = load_dataset("rootsautomation/ScreenSpot", split="test",
                           trust_remote_code=True)
    samples = []
    for i, ex in enumerate(dataset):
        bbox = list(ex["bbox"])
        # v1 bbox is already normalised [0,1]
        if any(v > 1.0 for v in bbox):
            img = ex["image"]
            w, h = img.size
            bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

        platform = ex.get("data_type", ex.get("platform", "web"))
        domain = DOMAIN_MAP.get(platform.lower().split("-")[0] if platform else "web", "web")
        ui_type = ex.get("ui_type", "text")

        samples.append({
            "image": ex["image"],
            "image_path": None,  # will save temp
            "instruction": ex.get("instruction", ex.get("prompt", "")),
            "bbox_norm": bbox,  # [x1, y1, x2, y2] normalised
            "ui_type": ui_type,
            "domain": domain,
            "data_source": platform,
            "dataset_idx": i,
        })
    return samples


def _find_v2_data_dir():
    """Find ScreenSpot-v2 data in HF cache."""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--OS-Copilot--ScreenSpot-v2")
    if not os.path.exists(cache_base):
        return None
    snapshots = glob.glob(os.path.join(cache_base, "snapshots", "*"))
    return snapshots[0] if snapshots else None


def load_screenspot_v2(data_dir=None):
    """Load ScreenSpot v2 from local JSON files."""
    if data_dir is None:
        data_dir = _find_v2_data_dir()
    if data_dir is None:
        # Download
        from huggingface_hub import snapshot_download
        data_dir = snapshot_download("OS-Copilot/ScreenSpot-v2", repo_type="dataset")

    json_files = {
        "mobile": "screenspot_mobile_v2.json",
        "desktop": "screenspot_desktop_v2.json",
        "web": "screenspot_web_v2.json",
    }
    samples = []
    for category, json_name in json_files.items():
        json_path = os.path.join(data_dir, json_name)
        if not os.path.exists(json_path):
            print(f"  Warning: {json_path} not found, skipping {category}")
            continue
        with open(json_path, "r") as f:
            items = json.load(f)
        print(f"  {category}: {len(items)} samples")
        for item in items:
            # bbox is [x, y, w, h] in pixels
            x, y, w, h = item["bbox"]
            domain = DOMAIN_MAP.get(item.get("data_source", "").lower(), "web")
            samples.append({
                "image": None,
                "image_path": None,
                "img_filename": item["img_filename"],
                "instruction": item["instruction"],
                "bbox_xywh_px": [x, y, w, h],
                "ui_type": item.get("data_type", "text"),
                "domain": domain,
                "data_source": item.get("data_source", ""),
            })

    # Find or extract images
    image_dir = os.path.join(data_dir, "screenspotv2_image")
    if not os.path.isdir(image_dir):
        zip_path = os.path.join(data_dir, "screenspotv2_image.zip")
        if os.path.exists(zip_path):
            print(f"  Extracting images from {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)

    for item in samples:
        item["image_path"] = os.path.join(image_dir, item["img_filename"])

    return samples


def load_screenspot_pro(data_dir):
    """Load ScreenSpot-Pro from local directory."""
    if data_dir is None:
        raise ValueError("--data_dir required for ScreenSpot-Pro")

    # Find annotation files
    ann_dir = os.path.join(data_dir, "annotations")
    if os.path.isdir(ann_dir):
        json_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    else:
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    all_data = []
    for json_path in json_files:
        with open(json_path, "r") as f:
            items = json.load(f)
        if isinstance(items, list):
            all_data.extend(items)

    # Find image directory
    image_dir = data_dir
    if os.path.isdir(os.path.join(data_dir, "images")):
        image_dir = os.path.join(data_dir, "images")

    samples = []
    for item in all_data:
        bbox = item["bbox"]  # [x1, y1, x2, y2] in pixels
        img_path = os.path.join(image_dir, item["img_filename"])
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, item["img_filename"])

        # Determine category and ui_type
        ui_type = item.get("ui_type", "text")
        # Pro has application/platform fields for category grouping
        application = item.get("application", "")
        platform = item.get("platform", "")
        group = item.get("group", "")

        samples.append({
            "image": None,
            "image_path": img_path,
            "instruction": item["instruction"],
            "bbox_px": bbox,  # [x1, y1, x2, y2] in pixels
            "ui_type": ui_type,
            "application": application,
            "platform": platform,
            "group": group,
            "img_filename": item.get("img_filename", ""),
        })

    print(f"  Loaded {len(samples)} Pro samples")
    return samples


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
# Helpers
# ---------------------------------------------------------------------------

def do_boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics_v1v2(results):
    """Print metrics for v1/v2: mobile/desktop/web × text/icon."""
    domains = ["mobile", "desktop", "web"]
    ui_types = ["text", "icon"]
    return _print_metrics(results, domains, ui_types,
                          group_key="domain", ui_key="ui_type")


def print_metrics_pro(results):
    """Print metrics for Pro: Dev/Creative/CAD/Scientific/Office/OS × text/icon."""
    ui_types = ["text", "icon"]
    return _print_metrics(results, PRO_CATEGORIES, ui_types,
                          group_key="category", ui_key="ui_type")


def _print_metrics(results, groups, ui_types, group_key, ui_key):
    metrics = ["hit_top1", "overlap_top1"]

    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(ex.get(key, 0) for ex in examples) / len(examples)

    metric_results = {m: {} for m in metrics}

    for grp in groups:
        grp_examples = [r for r in results if r.get(group_key) == grp]
        for ui in ui_types:
            sub = [r for r in grp_examples if r.get(ui_key) == ui]
            col = f"{grp}-{ui}"
            for m in metrics:
                metric_results[m][col] = compute_mean(sub, m)
        col_avg = f"{grp}-avg"
        for m in metrics:
            metric_results[m][col_avg] = compute_mean(grp_examples, m)

    for ui in ui_types:
        sub = [r for r in results if r.get(ui_key) == ui]
        col = f"All-{ui}"
        for m in metrics:
            metric_results[m][col] = compute_mean(sub, m)

    for m in metrics:
        metric_results[m]["All-avg"] = compute_mean(results, m)

    # Build column order
    columns = []
    for grp in groups:
        for ui in ui_types:
            columns.append(f"{grp}-{ui}")
        columns.append(f"{grp}-avg")
    for ui in ui_types:
        columns.append(f"All-{ui}")
    columns.append("All-avg")

    # Print table
    header = [""] + columns
    col_widths = [max(len(c), 12) for c in header]

    def fmt(val):
        if isinstance(val, float):
            return f"{val * 100:.2f}"
        return "N/A" if val is None else str(val)

    print(" | ".join(w.ljust(cw) for w, cw in zip(header, col_widths)))
    print("-+-".join("-" * cw for cw in col_widths))
    for m in metrics:
        row = [m] + [fmt(metric_results[m].get(c)) for c in columns]
        print(" | ".join(w.ljust(cw) for w, cw in zip(row, col_widths)))

    # Tab-delimited for Excel
    info = "Tab-delimited Table for Excel:\n"
    info += "\t".join([""] + columns) + "\n"
    for m in metrics:
        row = [m] + [fmt(metric_results[m].get(c)) for c in columns]
        info += "\t".join(row) + "\n"
    print(info)
    return info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(sample, model, tokenizer, builder, args, device, tmp_dir, rank, idx):
    """Evaluate a single sample. Returns result dict."""
    instruction = sample["instruction"]

    result = {
        "instruction": instruction,
        "ui_type": sample.get("ui_type", "text"),
        "hit_top1": 0,
        "overlap_top1": 0,
    }

    # Copy dataset-specific fields
    for key in ["domain", "data_source", "category", "application",
                "platform", "group", "img_filename", "file_name"]:
        if key in sample:
            result[key] = sample[key]

    # Get image
    if sample.get("image") is not None:
        image = sample["image"].convert("RGB")
        tmp_path = os.path.join(tmp_dir, f"tmp_{rank}_{idx}.png")
        image.save(tmp_path)
        image_path = tmp_path
    elif sample.get("image_path") and os.path.exists(sample["image_path"]):
        image = Image.open(sample["image_path"]).convert("RGB")
        image_path = sample["image_path"]
    else:
        return result

    w, h = image.size

    # Get normalised GT bbox [x1, y1, x2, y2]
    if "bbox_norm" in sample:
        gt_bbox = sample["bbox_norm"]
    elif "bbox_xywh_px" in sample:
        x, y, bw, bh = sample["bbox_xywh_px"]
        gt_bbox = [x / w, y / h, (x + bw) / w, (y + bh) / h]
    elif "bbox_px" in sample:
        bbox = sample["bbox_px"]
        gt_bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
    else:
        return result

    t_start = time.time()
    pred = run_saccade_inference(
        image, image_path, instruction,
        model, tokenizer, builder,
        max_rounds=args.rounds,
        crop_size=args.crop_size,
        crop_upscale=args.crop_upscale,
        device=str(device),
    )
    t_elapsed = time.time() - t_start

    topk_points = pred.get("topk_points", [])
    if not topk_points:
        return result

    n_w = pred.get("n_width", 1)
    n_h = pred.get("n_height", 1)
    px, py = topk_points[0]

    x1, y1, x2, y2 = gt_bbox
    if x1 <= px <= x2 and y1 <= py <= y2:
        result["hit_top1"] = 1

    half_pw = 0.5 / max(n_w, 1)
    half_ph = 0.5 / max(n_h, 1)
    pred_bbox = [px - half_pw, py - half_ph, px + half_pw, py + half_ph]
    if do_boxes_overlap(pred_bbox, gt_bbox):
        result["overlap_top1"] = 1

    result["pred_x"] = px
    result["pred_y"] = py
    result["num_rounds"] = pred.get("num_rounds", 1)
    result["total_vis_tokens"] = pred.get("total_vis_tokens", 0)
    result["inference_time"] = t_elapsed

    return result


def evaluate_all(samples, model, tokenizer, builder, args, device, rank):
    """Evaluate all samples."""
    tmp_dir = os.path.join(args.save_path, "tmp_images")
    os.makedirs(tmp_dir, exist_ok=True)

    results = []
    for i, sample in tqdm(enumerate(samples), total=len(samples), disable=(rank != 0)):
        result = evaluate_sample(sample, model, tokenizer, builder, args, device,
                                 tmp_dir, rank, i)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Pro-specific: assign categories
# ---------------------------------------------------------------------------

def assign_pro_categories(results, data_dir):
    """Assign Pro categories based on annotation file names (application_platform.json)."""
    ann_dir = os.path.join(data_dir, "annotations")
    if not os.path.isdir(ann_dir):
        return

    # Map img_filename → category from annotation files
    filename_to_cat = {}
    for json_path in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        stem = Path(json_path).stem  # e.g., "photoshop_windows", "all"
        if stem == "all":
            continue
        # Extract application name (remove _platform suffix)
        parts = stem.rsplit("_", 1)  # ["photoshop", "windows"]
        app_name = parts[0] if len(parts) > 1 else stem
        category = APP_TO_CATEGORY.get(app_name, None)
        if category is None:
            print(f"  Warning: unknown app '{app_name}' from {Path(json_path).name}")
            continue

        with open(json_path) as f:
            items = json.load(f)
        if isinstance(items, list):
            for item in items:
                fn = item.get("img_filename", "")
                filename_to_cat[fn] = category

    assigned = 0
    for r in results:
        fn = r.get("img_filename", "")
        if fn in filename_to_cat:
            r["category"] = filename_to_cat[fn]
            assigned += 1
    print(f"  Assigned categories to {assigned}/{len(results)} samples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified ScreenSpot evaluation")

    # Dataset
    parser.add_argument("--dataset", type=str, default="v1",
                        choices=["v1", "v2", "pro"],
                        help="Dataset: v1 (ScreenSpot), v2 (ScreenSpot-v2), pro (ScreenSpot-Pro)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Local data directory (required for pro, optional for v2)")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")

    # Saccade
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--crop_ratio", type=float, default=0.0)
    parser.add_argument("--crop_size", type=int, default=252)
    parser.add_argument("--crop_upscale", type=int, default=3)
    parser.add_argument("--adaptive_crop", action="store_true", default=True,
                        help="Use 3x3 adaptive crop (default: True)")
    parser.add_argument("--no_adaptive_crop", action="store_true",
                        help="Disable adaptive crop, use fixed crop_size")

    # Resolution
    parser.add_argument("--low_res_max_pixels", type=int, default=400000)

    # Misc
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.no_adaptive_crop:
        args.adaptive_crop = False

    # DDP
    rank, world_size, is_distributed, ddp_device = setup_distributed()
    if ddp_device:
        args.device = ddp_device
    is_main = (rank == 0)

    # Auto save path
    if args.save_path is None:
        ckpt_name = Path(args.checkpoint).name
        crop_tag = "adaptive3x3" if args.adaptive_crop else f"crop{args.crop_size}x{args.crop_upscale}"
        tag = f"saccade_r{args.rounds}_{crop_tag}"
        args.save_path = f"results/screenspot_{args.dataset}/{ckpt_name}/{tag}"

    if is_main:
        print(f"=== ScreenSpot {args.dataset.upper()} Eval ===")
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  base_model:  {args.base_model}")
        print(f"  rounds:      {args.rounds}")
        print(f"  adaptive:    {args.adaptive_crop}")
        if not args.adaptive_crop:
            up = args.crop_size * args.crop_upscale
            print(f"  crop:        {args.crop_size}x{args.crop_size} x{args.crop_upscale} -> {up}x{up}")
        print(f"  low_res:     {args.low_res_max_pixels}")
        print(f"  device:      {args.device}")
        if is_distributed:
            print(f"  DDP:         {world_size} GPUs")
        print()

    # Load data
    if is_main:
        print(f"Loading {args.dataset} data...")
    if args.dataset == "v1":
        samples = load_screenspot_v1(args.data_dir)
    elif args.dataset == "v2":
        samples = load_screenspot_v2(args.data_dir)
    elif args.dataset == "pro":
        samples = load_screenspot_pro(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if is_main:
        print(f"Total: {len(samples)} samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        if is_main:
            print(f"Limiting to {len(samples)} samples")

    # Shard with padding
    total = len(samples)
    if is_distributed:
        samples = samples[rank::world_size]
        max_shard = (total + world_size - 1) // world_size
        while len(samples) < max_shard:
            samples.append(samples[-1])
        if is_main:
            print(f"Rank 0: {len(samples)} samples (total {total} across {world_size} GPUs)")

    # Load model
    if is_main:
        print(f"Loading model: {args.checkpoint}")
    model, tokenizer = Qwen25VLWithDualHead.load_pretrained(
        args.checkpoint,
        base_model_name_or_path=args.base_model,
        device=args.device,
    )
    model.eval()

    crop_pixels = (args.crop_size * args.crop_upscale) ** 2 if not args.adaptive_crop else 756 ** 2
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
            local_results = evaluate_all(samples, model, tokenizer, builder,
                                         args, args.device, rank)

        if is_distributed:
            if is_main:
                print(f"Gathering results from {world_size} GPUs...")
            dist.barrier()
            results = gather_results(local_results, rank, world_size)
        else:
            results = local_results

        if is_main:
            # Deduplicate padded samples
            seen = set()
            unique = []
            for r in results:
                key = r.get("img_filename", r.get("file_name", "")) + r.get("instruction", "")
                if key and key not in seen:
                    seen.add(key)
                    unique.append(r)
                elif not key:
                    unique.append(r)
            results = unique

            elapsed = time.time() - t0
            print(f"Eval took {elapsed:.1f}s ({elapsed / max(len(results), 1):.2f}s/sample)")
            print(f"Total unique results: {len(results)}")

            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved predictions to {pred_path}")

            # Stats
            round_counts = {}
            for r in results:
                nr = r.get("num_rounds", 1)
                round_counts[nr] = round_counts.get(nr, 0) + 1
            print(f"Round distribution: {dict(sorted(round_counts.items()))}")

            tokens = [r.get("total_vis_tokens", 0) for r in results if r.get("total_vis_tokens", 0) > 0]
            if tokens:
                print(f"Avg visual tokens: {sum(tokens) / len(tokens):.0f}")

    # Metrics
    if is_main:
        if os.path.exists(metric_path):
            print(f"Metrics already exist at {metric_path}")
            with open(metric_path, "r") as f:
                print(f.read())
        else:
            print("\n=== Metrics ===")

            # Assign Pro categories if needed
            if args.dataset == "pro" and args.data_dir:
                assign_pro_categories(results, args.data_dir)

            if args.dataset in ("v1", "v2"):
                metric_info = print_metrics_v1v2(results)
            else:
                metric_info = print_metrics_pro(results)

            with open(metric_path, "w") as f:
                f.write(metric_info)
            print(f"\nSaved metrics to {metric_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
