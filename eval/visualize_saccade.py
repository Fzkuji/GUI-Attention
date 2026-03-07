"""Visualize current multi-round GUI inference on a single image.

Produces a figure showing:
  - Each round's attention heatmap overlaid on the image
  - Crop bounding boxes and predicted click points
  - Final prediction vs GT (if provided)

Usage:
  # With a trained checkpoint
  python eval/visualize_saccade.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
      --image /path/to/screenshot.png \
      --instruction "Click the search button" \
      --output viz_output.png

  # With GT bbox (normalised x1,y1,x2,y2)
  python eval/visualize_saccade.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
      --image /path/to/screenshot.png \
      --instruction "Click the search button" \
      --gt_bbox 0.45,0.32,0.55,0.38 \
      --output viz_output.png

  # From ScreenSpot-Pro dataset (auto-loads image + instruction + GT)
  python eval/visualize_saccade.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/GUI-Actor-3B-Qwen2.5-VL \
      --data_dir /path/to/ScreenSpot-Pro \
      --sample_index 42 \
      --output viz_output.png
"""

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if TYPE_CHECKING:
    from gui_attention.inputs.builder import MultiRoundInputBuilder

from gui_attention.runtime.proposals import compute_basin_proposals, rect_box


# ---------------------------------------------------------------------------
# Saccade inference with per-round recording
# ---------------------------------------------------------------------------

def run_saccade_with_recording(
    image: Image.Image,
    image_path: str,
    instruction: str,
    model,
    tokenizer,
    builder: "MultiRoundInputBuilder",
    max_rounds: int = 6,
    crop_ratio: float = 0.0,
    crop_size: int = 308,
    crop_upscale: int = 3,
    crop_target_pixels: int = 0,
    device: str = "cuda:0",
    reasoning_max_new_tokens: int = 48,
) -> dict:
    """Run the current inference pipeline and return its trace for plotting."""
    from gui_attention.runtime.inference import run_saccade_inference

    del crop_ratio, crop_target_pixels  # legacy args kept for CLI compatibility
    return run_saccade_inference(
        image=image,
        image_path=image_path,
        instruction=instruction,
        model=model,
        tokenizer=tokenizer,
        builder=builder,
        max_rounds=max_rounds,
        crop_size=crop_size,
        crop_upscale=crop_upscale,
        device=device,
        use_click_head=True,
        use_dual_tokens=True,
        reasoning_max_new_tokens=reasoning_max_new_tokens,
        return_trace=True,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_grid(ax, n_width, n_height, img_w, img_h, color="white", alpha=0.3, linewidth=0.5):
    """Draw patch grid lines on an axis."""
    patch_w = img_w / n_width
    patch_h = img_h / n_height
    for i in range(1, n_width):
        ax.axvline(x=i * patch_w, color=color, alpha=alpha, linewidth=linewidth)
    for j in range(1, n_height):
        ax.axhline(y=j * patch_h, color=color, alpha=alpha, linewidth=linewidth)


def _shorten_reasoning(text: str, limit: int = 90) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _draw_basin_proposals(
    ax,
    proposals,
    *,
    width_px: int,
    height_px: int,
    label_prefix: str = "#",
):
    """Overlay ranked basin proposals on an existing image axis."""
    colors = ["lime", "cyan", "orange", "magenta", "yellow"]
    for idx, proposal in enumerate(proposals, start=1):
        x1, y1, x2, y2 = rect_box(
            proposal.rect.center_x,
            proposal.rect.center_y,
            proposal.rect.width,
            proposal.rect.height,
        )
        color = colors[(idx - 1) % len(colors)]
        rect = patches.Rectangle(
            (x1 * width_px, y1 * height_px),
            (x2 - x1) * width_px,
            (y2 - y1) * height_px,
            linewidth=2.0,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1 * width_px + 4,
            y1 * height_px + 16,
            f"{label_prefix}{idx}",
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )


def plot_saccade_rounds(image: Image.Image, result: dict, gt_bbox=None,
                        instruction: str = "", output_path: str = "viz_saccade.png",
                        show_basin_proposals: bool = True, proposal_topk: int = 3):
    """Plot multi-round saccade results.

    Layout: one row per round.
      - Left: full image with optional low-res heatmap, crop box, and predicted point
      - Right: selected crop with crop heatmap when available
    """
    rounds_info = result.get("trace", [])
    n_rounds = len(rounds_info)
    if n_rounds == 0:
        print("No rounds to visualize.")
        return

    img_w, img_h = image.size
    img_np = np.array(image)

    # Count columns: round 0 = 1 col, subsequent = 2 cols (full + crop)
    n_cols = 2
    fig, axes = plt.subplots(n_rounds, n_cols, figsize=(n_cols * 6, n_rounds * 5))
    if n_rounds == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Instruction: "{instruction}"', fontsize=12, y=0.98)

    for i, rinfo in enumerate(rounds_info):
        ri = rinfo["round"]

        ax_full = axes[i, 0]
        ax_crop = axes[i, 1]

        ax_full.imshow(img_np)
        low_attn = rinfo.get("low_attn_2d")
        if low_attn is not None:
            heatmap_low = _upsample_heatmap(low_attn, img_w, img_h)
            ax_full.imshow(heatmap_low, alpha=0.4, cmap="jet", extent=[0, img_w, img_h, 0])
            if show_basin_proposals:
                low_props, _ = compute_basin_proposals(low_attn, top_k=proposal_topk)
                _draw_basin_proposals(
                    ax_full,
                    low_props,
                    width_px=img_w,
                    height_px=img_h,
                    label_prefix="#",
                )
        if rinfo.get("grid_dims_low") is not None:
            nh_low, nw_low = rinfo["grid_dims_low"]
            _draw_grid(ax_full, nw_low, nh_low, img_w, img_h, color="white", alpha=0.3, linewidth=0.5)

        crop_bbox = rinfo.get("crop_bbox")
        decision_crop_bbox = rinfo.get("decision_crop_bbox")
        if decision_crop_bbox is not None:
            cx1, cy1, cx2, cy2 = decision_crop_bbox
            rect = patches.Rectangle(
                (cx1 * img_w, cy1 * img_h),
                (cx2 - cx1) * img_w, (cy2 - cy1) * img_h,
                linewidth=2.0, edgecolor="cyan", facecolor="none", linestyle="--"
            )
            ax_full.add_patch(rect)
        if crop_bbox is not None and crop_bbox != decision_crop_bbox:
            cx1, cy1, cx2, cy2 = crop_bbox
            rect = patches.Rectangle(
                (cx1 * img_w, cy1 * img_h),
                (cx2 - cx1) * img_w, (cy2 - cy1) * img_h,
                linewidth=2.0, edgecolor="orange", facecolor="none", linestyle="-."
            )
            ax_full.add_patch(rect)

        if rinfo.get("pred_x") is not None and rinfo.get("pred_y") is not None:
            px, py = rinfo["pred_x"] * img_w, rinfo["pred_y"] * img_h
            marker = "o" if rinfo.get("point_kind") == "click" else "*"
            ax_full.plot(px, py, marker, color="cyan", markersize=16,
                         markeredgecolor="black", markeredgewidth=1.5)
        if gt_bbox is not None:
            _draw_gt(ax_full, gt_bbox, img_w, img_h)

        title = (
            f"Round {ri}: action={rinfo.get('action')} "
            f"(attended={rinfo.get('attended_image', '?')}, fmt={'ok' if rinfo.get('format_ok') else 'bad'})"
        )
        ax_full.set_title(title, fontsize=10)
        ax_full.axis("off")

        if crop_bbox is not None:
            cx1, cy1, cx2, cy2 = crop_bbox
            crop_pil = image.crop((
                int(cx1 * img_w), int(cy1 * img_h),
                int(cx2 * img_w), int(cy2 * img_h),
            ))
            crop_np = np.array(crop_pil)
            cw, ch = crop_pil.size
            ax_crop.imshow(crop_np)
            if rinfo.get("crop_attn_2d") is not None:
                crop_heatmap = _upsample_heatmap(rinfo["crop_attn_2d"], cw, ch)
                ax_crop.imshow(crop_heatmap, alpha=0.5, cmap="jet", extent=[0, cw, ch, 0])
                if show_basin_proposals:
                    crop_props, _ = compute_basin_proposals(rinfo["crop_attn_2d"], top_k=proposal_topk)
                    _draw_basin_proposals(
                        ax_crop,
                        crop_props,
                        width_px=cw,
                        height_px=ch,
                        label_prefix="p",
                    )
            if rinfo.get("grid_dims_crop") is not None:
                crop_nh, crop_nw = rinfo["grid_dims_crop"]
                _draw_grid(ax_crop, crop_nw, crop_nh, cw, ch, color="white", alpha=0.5, linewidth=0.5)

            if (
                rinfo.get("pred_x") is not None and rinfo.get("pred_y") is not None
                and cx1 <= rinfo["pred_x"] <= cx2 and cy1 <= rinfo["pred_y"] <= cy2
            ):
                local_px = (rinfo["pred_x"] - cx1) / (cx2 - cx1) * cw
                local_py = (rinfo["pred_y"] - cy1) / (cy2 - cy1) * ch
                marker = "o" if rinfo.get("point_kind") == "click" else "*"
                ax_crop.plot(local_px, local_py, marker, color="cyan", markersize=16,
                             markeredgecolor="black", markeredgewidth=1.5)

            if gt_bbox is not None:
                gx1, gy1, gx2, gy2 = gt_bbox
                gt_cx = (gx1 + gx2) / 2
                gt_cy = (gy1 + gy2) / 2
                if cx1 <= gt_cx <= cx2 and cy1 <= gt_cy <= cy2:
                    local_gx = (gt_cx - cx1) / (cx2 - cx1) * cw
                    local_gy = (gt_cy - cy1) / (cy2 - cy1) * ch
                    ax_crop.plot(local_gx, local_gy, "r+", markersize=18, markeredgewidth=3)

            crop_title = f"Crop ({rinfo.get('action')})"
            if rinfo.get("selected_crop_round") is not None:
                crop_title += f" from round {rinfo['selected_crop_round']}"
            ax_crop.set_title(crop_title, fontsize=10)
        else:
            ax_crop.text(0.5, 0.5, "No crop view", ha="center", va="center",
                         transform=ax_crop.transAxes, fontsize=12, color="gray")
        reasoning = _shorten_reasoning(rinfo.get("reasoning_text", ""))
        if reasoning:
            ax_crop.text(
                0.02, -0.08,
                reasoning,
                transform=ax_crop.transAxes,
                fontsize=9,
                va="top",
                ha="left",
            )
        ax_crop.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="cyan",
               markeredgecolor="black", markersize=14, label="Prediction"),
    ]
    if gt_bbox is not None:
        legend_elements.append(
            Line2D([0], [0], marker="+", color="w", markerfacecolor="red",
                   markeredgecolor="red", markersize=14, markeredgewidth=3, label="GT center"),
        )
        legend_elements.append(
            patches.Patch(edgecolor="lime", facecolor="none", linewidth=2, label="GT bbox"),
        )
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")


def _upsample_heatmap(attn_2d: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Upsample a 2D attention map to target image size."""
    t = torch.from_numpy(attn_2d).float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return upsampled.squeeze().numpy()


def _draw_gt(ax, gt_bbox, img_w, img_h):
    """Draw GT bbox and center on an axis."""
    gx1, gy1, gx2, gy2 = gt_bbox
    rect = patches.Rectangle(
        (gx1 * img_w, gy1 * img_h),
        (gx2 - gx1) * img_w, (gy2 - gy1) * img_h,
        linewidth=2, edgecolor="lime", facecolor="none"
    )
    ax.add_patch(rect)
    gt_cx = (gx1 + gx2) / 2 * img_w
    gt_cy = (gy1 + gy2) / 2 * img_h
    ax.plot(gt_cx, gt_cy, "r+", markersize=18, markeredgewidth=3)


# ---------------------------------------------------------------------------
# ScreenSpot-Pro loader
# ---------------------------------------------------------------------------

def load_screenspot_sample(screenspot_dir: str, index: int):
    """Load a single sample from ScreenSpot-Pro dataset."""
    # Try multiple possible locations
    candidates = [
        os.path.join(screenspot_dir, "screenspot_pro.json"),
        os.path.join(screenspot_dir, "annotations", "all.json"),
    ]
    # Also try all json files in annotations/
    ann_dir = os.path.join(screenspot_dir, "annotations")
    if os.path.isdir(ann_dir):
        for fn in sorted(os.listdir(ann_dir)):
            if fn.endswith(".json"):
                candidates.append(os.path.join(ann_dir, fn))

    data = []
    for json_path in candidates:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
            if len(data) > index:
                break
    if not data:
        raise FileNotFoundError(f"No annotation JSON found in {screenspot_dir}. Tried: {candidates}")
    if index >= len(data):
        raise IndexError(f"Sample index {index} out of range (total {len(data)})")
    sample = data[index]

    # Try with and without images/ subdirectory
    img_path = os.path.join(screenspot_dir, sample["img_filename"])
    if not os.path.exists(img_path):
        img_path = os.path.join(screenspot_dir, "images", sample["img_filename"])
    instruction = sample["instruction"]
    bbox = sample["bbox"]  # [x1, y1, x2, y2] in pixels
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    gt_bbox = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)

    return img, img_path, instruction, gt_bbox, sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize current multi-round GUI inference")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--base_model", required=True, help="Path to base Qwen2.5-VL model")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--instruction", help="Text instruction")
    parser.add_argument("--gt_bbox", help="GT bbox as x1,y1,x2,y2 (normalised)")
    parser.add_argument("--dataset", default="pro", choices=["v1", "v2", "pro"],
                        help="Dataset: v1 (ScreenSpot), v2 (ScreenSpot-v2), pro (ScreenSpot-Pro)")
    parser.add_argument("--data_dir", help="Path to dataset directory (required for pro/v2)")
    parser.add_argument("--screenspot_dir", help="(Deprecated) Alias for --data_dir with --dataset pro")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to visualize (batch mode)")
    parser.add_argument("--output", default="viz_saccade.png", help="Output image path")
    parser.add_argument("--max_rounds", type=int, default=6, help="Max saccade rounds")
    parser.add_argument("--crop_ratio", type=float, default=0.0, help="Crop ratio (legacy, 0=use crop_size)")
    parser.add_argument("--crop_size", type=int, default=308, help="Fixed crop side length in pixels")
    parser.add_argument("--crop_upscale", type=int, default=3, help="Integer upscale factor")
    parser.add_argument("--crop_target_pixels", type=int, default=0, help="(Legacy) Crop target pixels")
    parser.add_argument("--low_res_max_pixels", type=int, default=1001600, help="Low-res max pixels")
    parser.add_argument("--reasoning_max_new_tokens", type=int, default=48, help="Max new tokens for reasoning generation")
    parser.add_argument("--proposal_topk", type=int, default=3, help="Number of basin proposals to overlay per panel")
    parser.add_argument("--show_basin_proposals", action=argparse.BooleanOptionalAction, default=True,
                        help="Overlay ranked attention-region proposals on low-res and crop views")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    from gui_attention.inputs.builder import MultiRoundInputBuilder
    from gui_attention.modeling.model import Qwen25VLWithDualHead

    # Load model
    print(f"Loading model from {args.checkpoint} (base: {args.base_model})")
    model, tokenizer = Qwen25VLWithDualHead.load_pretrained(
        args.checkpoint, args.base_model,
        attn_implementation="flash_attention_2",
        device=args.device,
    )
    model.eval()
    print("Model loaded.")

    # Build input builder
    crop_pixels = args.crop_target_pixels if args.crop_target_pixels > 0 else (args.crop_size * args.crop_upscale) ** 2
    builder = MultiRoundInputBuilder(
        model_path=args.base_model,
        tokenizer=tokenizer,
        low_res_max_pixels=args.low_res_max_pixels,
        high_res_max_pixels=crop_pixels,
    )

    # Backward compat: --screenspot_dir → --data_dir + --dataset pro
    if args.screenspot_dir and not args.data_dir:
        args.data_dir = args.screenspot_dir
        args.dataset = "pro"

    # Collect samples to visualize
    samples = []
    if args.data_dir or args.dataset in ("v1", "v2"):
        # Load full dataset then pick samples by index
        from eval_screenspot import load_screenspot_v1, load_screenspot_v2, load_screenspot_pro
        if args.dataset == "v1":
            all_samples = load_screenspot_v1(args.data_dir)
        elif args.dataset == "v2":
            all_samples = load_screenspot_v2(args.data_dir)
        else:
            all_samples = load_screenspot_pro(args.data_dir)

        for idx in range(args.sample_index, min(args.sample_index + args.num_samples, len(all_samples))):
            s = all_samples[idx]
            # Get image
            if s.get("image") is not None:
                img = s["image"].convert("RGB")
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                img_path = tmp.name
            else:
                img_path = s["image_path"]
                img = Image.open(img_path).convert("RGB")

            w, h = img.size
            # Get normalised GT bbox
            if "bbox_norm" in s:
                gt_bbox = tuple(s["bbox_norm"])
            elif "bbox_xywh_px" in s:
                x, y, bw, bh = s["bbox_xywh_px"]
                gt_bbox = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            elif "bbox_px" in s:
                b = s["bbox_px"]
                gt_bbox = (b[0] / w, b[1] / h, b[2] / w, b[3] / h)
            else:
                gt_bbox = None

            out_path = args.output.replace(".png", f"_{idx}.png") if args.num_samples > 1 else args.output
            samples.append((img, img_path, s["instruction"], gt_bbox, out_path, idx))

    elif args.image:
        image = Image.open(args.image).convert("RGB")
        image_path = args.image
        instruction = args.instruction or "Click the target element"
        gt_bbox = None
        if args.gt_bbox:
            gt_bbox = tuple(float(x) for x in args.gt_bbox.split(","))
        samples.append((image, image_path, instruction, gt_bbox, args.output, 0))
    else:
        parser.error("Provide --image or --data_dir")
        return

    # Process each sample
    for image, image_path, instruction, gt_bbox, output_path, idx in samples:
        print(f"\n--- Sample {idx}: {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        if gt_bbox:
            print(f"  GT bbox (norm): ({gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f})")

        result = run_saccade_with_recording(
            image=image,
            image_path=image_path,
            instruction=instruction,
            model=model,
            tokenizer=tokenizer,
            builder=builder,
            max_rounds=args.max_rounds,
            crop_ratio=args.crop_ratio,
            crop_size=args.crop_size,
            crop_upscale=args.crop_upscale,
            crop_target_pixels=args.crop_target_pixels,
            device=args.device,
            reasoning_max_new_tokens=args.reasoning_max_new_tokens,
        )

        trace = result.get("trace", [])
        print(f"  {len(trace)} traced panels:")
        for rinfo in trace:
            ri = rinfo["round"]
            px, py = rinfo.get("pred_x"), rinfo.get("pred_y")
            hit = ""
            if gt_bbox and px is not None and py is not None:
                in_box = gt_bbox[0] <= px <= gt_bbox[2] and gt_bbox[1] <= py <= gt_bbox[3]
                hit = " ✓ HIT" if in_box else " ✗ MISS"
            pred_text = f"pred=({px:.4f}, {py:.4f})" if px is not None and py is not None else "pred=(n/a)"
            print(
                f"    Round {ri}: action={rinfo.get('action')} {pred_text}, "
                f"attended={rinfo.get('attended_image', 'low')}, "
                f"fmt={'ok' if rinfo.get('format_ok') else 'bad'}{hit}"
            )
            reasoning = _shorten_reasoning(rinfo.get("reasoning_text", ""), limit=100)
            if reasoning:
                print(f"      reasoning: {reasoning}")

        final_point = result["topk_points"][0]
        final_hit = ""
        if gt_bbox:
            in_box = gt_bbox[0] <= final_point[0] <= gt_bbox[2] and gt_bbox[1] <= final_point[1] <= gt_bbox[3]
            final_hit = " ✓ HIT" if in_box else " ✗ MISS"
        print(f"  Final: ({final_point[0]:.4f}, {final_point[1]:.4f}){final_hit}")

        plot_saccade_rounds(
            image,
            result,
            gt_bbox=gt_bbox,
            instruction=instruction,
            output_path=output_path,
            show_basin_proposals=args.show_basin_proposals,
            proposal_topk=args.proposal_topk,
        )


if __name__ == "__main__":
    main()
