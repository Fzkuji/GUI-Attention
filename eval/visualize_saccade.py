"""Visualize multi-round saccade inference on a single image.

Produces a figure showing:
  - Each round's attention heatmap overlaid on the image
  - Crop bounding boxes and predicted click points
  - Final prediction vs GT (if provided)

Usage:
  # With a trained checkpoint
  python eval/visualize_saccade.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --image /path/to/screenshot.png \
      --instruction "Click the search button" \
      --output viz_output.png

  # With GT bbox (normalised x1,y1,x2,y2)
  python eval/visualize_saccade.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --image /path/to/screenshot.png \
      --instruction "Click the search button" \
      --gt_bbox 0.45,0.32,0.55,0.38 \
      --output viz_output.png

  # From ScreenSpot-Pro dataset (auto-loads image + instruction + GT)
  python eval/visualize_saccade.py \
      --checkpoint /path/to/checkpoint \
      --base_model /path/to/Qwen2.5-VL-3B-Instruct \
      --screenspot_dir /path/to/ScreenSpot-Pro \
      --sample_index 42 \
      --output viz_output.png
"""

import argparse
import json
import os
import sys

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

from gui_attention.attention import (
    extract_anchor_hidden_states,
    extract_visual_hidden_states,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.crop import crop_image
from gui_attention.foveation import SaccadeLoop
from gui_attention.labels import compute_overlap_mask
from gui_attention.model import Qwen25VLWithActionHead


# ---------------------------------------------------------------------------
# Saccade inference with per-round recording
# ---------------------------------------------------------------------------

def run_saccade_with_recording(
    image: Image.Image,
    image_path: str,
    instruction: str,
    model,
    tokenizer,
    builder: MultiRoundInputBuilder,
    max_rounds: int = 4,
    crop_ratio: float = 0.0,
    crop_size: int = 252,
    crop_upscale: int = 3,
    crop_target_pixels: int = 0,
    device: str = "cuda:0",
) -> list:
    """Run saccade inference and record per-round details for visualization.

    Returns:
        list of dicts, one per round:
        {
            "round": int,
            "attn_2d": np.ndarray (nh, nw),     # attention heatmap
            "pred_x": float, "pred_y": float,    # predicted click (global normalised)
            "crop_bbox": (x1,y1,x2,y2) or None,  # crop region (global normalised)
            "attended_image": str,                # "low" or "high"
            "grid_dims": (nh, nw),
            "image_info": {...},
        }
    """
    saccade = SaccadeLoop(max_rounds=max_rounds, crop_ratio=crop_ratio)
    state = saccade.new_state()
    builder.reset()

    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    _backbone = model.backbone
    if hasattr(_backbone, "base_model") and hasattr(_backbone.base_model, "model"):
        _inner = _backbone.base_model.model
    else:
        _inner = _backbone
    if hasattr(_inner, "visual"):
        _visual = _inner.visual
    elif hasattr(_inner, "model") and hasattr(_inner.model, "visual"):
        _visual = _inner.model.visual
    else:
        raise AttributeError(f"Cannot find visual module in {type(_inner)}")
    merge = _visual.spatial_merge_size

    rounds_info = []

    # ---- Round 0: low-res ----
    r0_inputs, cur_text, cur_images = builder.build_round0(image_path, instruction)
    inp = {k: v.to(device) for k, v in r0_inputs.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=inp["input_ids"],
            attention_mask=inp.get("attention_mask"),
            pixel_values=inp.get("pixel_values"),
            image_grid_thw=inp.get("image_grid_thw"),
        )

    last_hs = outputs.hidden_states[-1]
    vis_embeds = model.extract_visual_embeds(
        inp["input_ids"], inp.get("pixel_values"), inp.get("image_grid_thw"))
    _, vis_ranges = extract_visual_hidden_states(last_hs, inp["input_ids"], img_tok)
    anchor = extract_anchor_hidden_states(last_hs, inp["input_ids"], pp_id, n=0)

    if vis_embeds is None or anchor is None:
        return rounds_info

    grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
    nh0, nw0 = grid_dims[0]

    attn0, _, _, _, _ = model.action_head(vis_embeds, anchor)
    attn_1d = attn0.squeeze(0)

    # Low-res attention for round 0
    n_low = vis_ranges[0][1]
    low_attn = attn_1d[:n_low].detach().cpu().float().numpy().reshape(nh0, nw0)
    lx, ly = token_to_spatial(attn_1d[:n_low].argmax().item(), nw0, nh0,
                              attn_weights=attn_1d[:n_low])

    rounds_info.append({
        "round": 0,
        "attn_2d": low_attn,
        "pred_x": lx,
        "pred_y": ly,
        "crop_bbox": None,
        "attended_image": "low",
        "grid_dims": (nh0, nw0),
    })

    focus_x, focus_y = lx, ly
    decision = saccade.decide_round0(state, focus_x, focus_y)

    # ---- Subsequent rounds ----
    for ri in range(1, max_rounds):
        if not saccade.should_continue(state, ri):
            break

        cropped, crop_bbox = crop_image(image, focus_x, focus_y,
                                        crop_ratio=crop_ratio,
                                        crop_size=crop_size,
                                        crop_upscale=crop_upscale,
                                        crop_target_pixels=crop_target_pixels)
        try:
            ri_inputs, cur_text, cur_images = builder.extend_with_crop(
                cur_text, cur_images, cropped, crop_bbox)
        except Exception:
            break

        inp = {k: v.to(device) for k, v in ri_inputs.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
            )

        last_hs = outputs.hidden_states[-1]
        vis_embeds = model.extract_visual_embeds(
            inp["input_ids"], inp.get("pixel_values"), inp.get("image_grid_thw"))
        _, vis_ranges = extract_visual_hidden_states(last_hs, inp["input_ids"], img_tok)
        anchor = extract_anchor_hidden_states(last_hs, inp["input_ids"], pp_id, n=ri)

        if vis_embeds is None or anchor is None or len(vis_ranges) < 2:
            break

        grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
        nh_low, nw_low = grid_dims[0]
        latest_img_idx = len(vis_ranges) - 1

        # Mask: current crop overlap on low-res. Old crops NOT masked (matches training).
        this_crop_mask = compute_overlap_mask(nh_low, nw_low, crop_bbox).to(device)
        n_low = vis_ranges[0][1]
        n_total = sum(r[1] for r in vis_ranges)
        full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        full_mask[:n_low] = this_crop_mask

        attn_ri, _, _, _, _ = model.action_head(vis_embeds, anchor, mask=full_mask)
        attn_1d = attn_ri.squeeze(0)

        img_idx, local_idx = identify_attended_image(attn_1d, vis_ranges)
        info = builder.image_infos[img_idx]

        if img_idx < len(grid_dims):
            nh_a, nw_a = grid_dims[img_idx]
        else:
            break

        off_a, n_a = vis_ranges[img_idx]
        img_attn = attn_1d[off_a:off_a + n_a]
        lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
        bx1, by1, bx2, by2 = info.global_bbox
        global_x = bx1 + lx * (bx2 - bx1)
        global_y = by1 + ly * (by2 - by1)

        attended_source = "high" if info.resolution == "high" else "low"

        # Build attention heatmap for this round
        # Low-res attention (unmasked patches)
        low_attn_raw = attn_1d[:vis_ranges[0][1]].detach().cpu().float().numpy()
        low_attn_2d = low_attn_raw.reshape(grid_dims[0])

        # Crop attention
        crop_off, crop_n = vis_ranges[latest_img_idx]
        crop_attn_raw = attn_1d[crop_off:crop_off + crop_n].detach().cpu().float().numpy()
        crop_nh, crop_nw = grid_dims[latest_img_idx]
        crop_attn_2d = crop_attn_raw.reshape(crop_nh, crop_nw)

        rounds_info.append({
            "round": ri,
            "attn_2d_low": low_attn_2d,
            "attn_2d_crop": crop_attn_2d,
            "pred_x": global_x,
            "pred_y": global_y,
            "crop_bbox": crop_bbox,
            "attended_image": attended_source,
            "grid_dims_low": grid_dims[0],
            "grid_dims_crop": (crop_nh, crop_nw),
        })

        decision = saccade.decide_saccade(state, attended_source, global_x, global_y)
        if decision["action"] == "stop":
            break
        focus_x, focus_y = global_x, global_y

    return rounds_info


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


def plot_saccade_rounds(image: Image.Image, rounds_info: list, gt_bbox=None,
                        instruction: str = "", output_path: str = "viz_saccade.png"):
    """Plot multi-round saccade results.

    Layout: one row per round.
      - Round 0: full image + low-res heatmap + predicted point + patch grid
      - Round N: full image with crop box + crop image with heatmap + patch grid
    """
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

        if ri == 0:
            # Round 0: full image with low-res heatmap
            ax_full = axes[i, 0]
            ax_full.imshow(img_np)
            attn = rinfo["attn_2d"]
            nh, nw = rinfo["grid_dims"]
            # Upsample heatmap to image size
            heatmap = _upsample_heatmap(attn, img_w, img_h)
            ax_full.imshow(heatmap, alpha=0.5, cmap="jet", extent=[0, img_w, img_h, 0])
            # Draw patch grid
            _draw_grid(ax_full, nw, nh, img_w, img_h, color="white", alpha=0.4, linewidth=0.5)
            # Predicted point
            px, py = rinfo["pred_x"] * img_w, rinfo["pred_y"] * img_h
            ax_full.plot(px, py, "c*", markersize=18, markeredgecolor="black", markeredgewidth=1.5)
            # GT
            if gt_bbox is not None:
                _draw_gt(ax_full, gt_bbox, img_w, img_h)
            ax_full.set_title(f"Round {ri}: Low-res overview ({nw}×{nh} patches)", fontsize=11)
            ax_full.axis("off")

            # Empty right column for round 0
            axes[i, 1].axis("off")
            axes[i, 1].set_title("(no crop)", fontsize=10, color="gray")

        else:
            # Round N: left = full image with crop box, right = crop with heatmap
            ax_full = axes[i, 0]
            ax_crop = axes[i, 1]

            # Left: full image + low-res heatmap + crop box
            ax_full.imshow(img_np)
            if "attn_2d_low" in rinfo:
                heatmap_low = _upsample_heatmap(rinfo["attn_2d_low"], img_w, img_h)
                ax_full.imshow(heatmap_low, alpha=0.4, cmap="jet", extent=[0, img_w, img_h, 0])
                # Draw low-res grid on full image
                nh_low, nw_low = rinfo["grid_dims_low"]
                _draw_grid(ax_full, nw_low, nh_low, img_w, img_h, color="white", alpha=0.3, linewidth=0.5)

            crop_bbox = rinfo["crop_bbox"]
            if crop_bbox is not None:
                cx1, cy1, cx2, cy2 = crop_bbox
                rect = patches.Rectangle(
                    (cx1 * img_w, cy1 * img_h),
                    (cx2 - cx1) * img_w, (cy2 - cy1) * img_h,
                    linewidth=2.5, edgecolor="cyan", facecolor="none", linestyle="--"
                )
                ax_full.add_patch(rect)

            # Predicted point on full image
            px, py = rinfo["pred_x"] * img_w, rinfo["pred_y"] * img_h
            ax_full.plot(px, py, "c*", markersize=18, markeredgecolor="black", markeredgewidth=1.5)
            if gt_bbox is not None:
                _draw_gt(ax_full, gt_bbox, img_w, img_h)
            attended = rinfo.get("attended_image", "?")
            ax_full.set_title(f"Round {ri}: Full image (attended={attended})", fontsize=11)
            ax_full.axis("off")

            # Right: crop image with crop heatmap + grid
            if crop_bbox is not None and "attn_2d_crop" in rinfo:
                cx1, cy1, cx2, cy2 = crop_bbox
                crop_pil = image.crop((
                    int(cx1 * img_w), int(cy1 * img_h),
                    int(cx2 * img_w), int(cy2 * img_h),
                ))
                crop_np = np.array(crop_pil)
                cw, ch = crop_pil.size
                ax_crop.imshow(crop_np)
                crop_heatmap = _upsample_heatmap(rinfo["attn_2d_crop"], cw, ch)
                ax_crop.imshow(crop_heatmap, alpha=0.5, cmap="jet", extent=[0, cw, ch, 0])
                # Draw crop patch grid
                crop_nh, crop_nw = rinfo["grid_dims_crop"]
                _draw_grid(ax_crop, crop_nw, crop_nh, cw, ch, color="white", alpha=0.5, linewidth=0.5)

                # Predicted point in crop coords
                if cx1 <= rinfo["pred_x"] <= cx2 and cy1 <= rinfo["pred_y"] <= cy2:
                    local_px = (rinfo["pred_x"] - cx1) / (cx2 - cx1) * cw
                    local_py = (rinfo["pred_y"] - cy1) / (cy2 - cy1) * ch
                    ax_crop.plot(local_px, local_py, "c*", markersize=18,
                                 markeredgecolor="black", markeredgewidth=1.5)

                # GT in crop coords
                if gt_bbox is not None:
                    gx1, gy1, gx2, gy2 = gt_bbox
                    gt_cx = (gx1 + gx2) / 2
                    gt_cy = (gy1 + gy2) / 2
                    if cx1 <= gt_cx <= cx2 and cy1 <= gt_cy <= cy2:
                        local_gx = (gt_cx - cx1) / (cx2 - cx1) * cw
                        local_gy = (gt_cy - cy1) / (cy2 - cy1) * ch
                        ax_crop.plot(local_gx, local_gy, "r+", markersize=18,
                                     markeredgewidth=3)

                ax_crop.set_title(f"Round {ri}: Crop region ({crop_nw}×{crop_nh} patches)", fontsize=11)
            else:
                ax_crop.text(0.5, 0.5, "No crop data", ha="center", va="center",
                             transform=ax_crop.transAxes, fontsize=12, color="gray")
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
    parser = argparse.ArgumentParser(description="Visualize saccade inference")
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
    parser.add_argument("--max_rounds", type=int, default=4, help="Max saccade rounds")
    parser.add_argument("--crop_ratio", type=float, default=0.0, help="Crop ratio (legacy, 0=use crop_size)")
    parser.add_argument("--crop_size", type=int, default=252, help="Fixed crop side length in pixels")
    parser.add_argument("--crop_upscale", type=int, default=3, help="Integer upscale factor")
    parser.add_argument("--crop_target_pixels", type=int, default=0, help="(Legacy) Crop target pixels")
    parser.add_argument("--low_res_max_pixels", type=int, default=400000, help="Low-res max pixels")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint} (base: {args.base_model})")
    model, tokenizer = Qwen25VLWithActionHead.load_pretrained(
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
        parser.error("Provide --image or --screenspot_dir")
        return

    # Process each sample
    for image, image_path, instruction, gt_bbox, output_path, idx in samples:
        print(f"\n--- Sample {idx}: {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        if gt_bbox:
            print(f"  GT bbox (norm): ({gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f})")

        rounds_info = run_saccade_with_recording(
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
        )

        print(f"  {len(rounds_info)} rounds:")
        for rinfo in rounds_info:
            ri = rinfo["round"]
            hit = ""
            if gt_bbox:
                px, py = rinfo["pred_x"], rinfo["pred_y"]
                in_box = gt_bbox[0] <= px <= gt_bbox[2] and gt_bbox[1] <= py <= gt_bbox[3]
                hit = " ✓ HIT" if in_box else " ✗ MISS"
            print(f"    Round {ri}: pred=({rinfo['pred_x']:.4f}, {rinfo['pred_y']:.4f}), "
                  f"attended={rinfo.get('attended_image', 'low')}{hit}")

        plot_saccade_rounds(image, rounds_info, gt_bbox=gt_bbox,
                            instruction=instruction, output_path=output_path)


if __name__ == "__main__":
    main()
