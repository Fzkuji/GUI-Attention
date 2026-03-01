"""Inference utilities for saccade foveation (v4).

Includes BFS connected region prediction (from GUI-Actor) and multi-round
saccade inference loop.
"""

from typing import List, Tuple

import torch

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


def get_prediction_region_point(
    attn_scores: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]], List[float]]:
    """BFS connected region prediction (adapted from GUI-Actor).

    1. Threshold activated patches (> activation_threshold * max_score).
    2. BFS to find connected regions (4-directional).
    3. Return weighted center of highest-scoring region.

    Args:
        attn_scores: (n_vis,) attention weights.
        n_width: grid width.
        n_height: grid height.
        activation_threshold: fraction of max score for thresholding.

    Returns:
        best_point: (x, y) normalised coordinates.
        sorted_centers: all region centers sorted by score.
        sorted_scores: corresponding average activation scores.
    """
    scores = attn_scores.float().cpu()
    max_score = scores.max().item()
    threshold = max_score * activation_threshold

    # Build set of activated patches
    activated = {}  # flat_idx → score
    for idx in range(scores.numel()):
        if scores[idx].item() > threshold:
            activated[idx] = scores[idx].item()

    if not activated:
        # Fallback: argmax
        idx = scores.argmax().item()
        x = (idx % n_width + 0.5) / n_width
        y = (idx // n_width + 0.5) / n_height
        return (x, y), [(x, y)], [scores[idx].item()]

    # BFS to find connected regions
    visited = set()
    regions = []

    for flat_idx, score in activated.items():
        if flat_idx in visited:
            continue
        region = []
        queue = [flat_idx]
        visited.add(flat_idx)
        while queue:
            curr = queue.pop(0)
            row = curr // n_width
            col = curr % n_width
            region.append((row, col, activated[curr]))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                nidx = nr * n_width + nc
                if 0 <= nr < n_height and 0 <= nc < n_width and nidx in activated and nidx not in visited:
                    visited.add(nidx)
                    queue.append(nidx)
        regions.append(region)

    # Score each region and compute weighted center
    region_info = []
    for region in regions:
        total_w = sum(s for _, _, s in region)
        avg_score = total_w / len(region)
        cx = sum(((c + 0.5) / n_width) * s for _, c, s in region) / total_w
        cy = sum(((r + 0.5) / n_height) * s for r, _, s in region) / total_w
        region_info.append((avg_score, cx, cy))

    region_info.sort(key=lambda x: x[0], reverse=True)
    sorted_scores = [r[0] for r in region_info]
    sorted_centers = [(r[1], r[2]) for r in region_info]
    best_point = sorted_centers[0]

    return best_point, sorted_centers, sorted_scores


def run_saccade_inference(
    image,
    image_path,
    instruction: str,
    model,
    tokenizer,
    builder: MultiRoundInputBuilder,
    max_rounds: int = 3,
    crop_ratio: float = 0.3,
    crop_upsample_pixels: int = 0,
    crop_target_pixels: int = 200704,
    device: str = "cuda:0",
    activation_threshold: float = 0.3,
) -> dict:
    """Multi-round saccade inference.

    Round 0: low-res full image → action head → select focus.
    Round 1+: low-res (masked) + high-res crop → action head
              → argmax in high → click; argmax in low → saccade.

    Returns:
        dict with topk_points, n_width, n_height, num_rounds, attended_source.
    """
    saccade = SaccadeLoop(max_rounds=max_rounds, crop_ratio=crop_ratio)
    state = saccade.new_state()
    builder.reset()

    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    # Navigate to spatial_merge_size (handle PEFT-wrapped, plain, and DeepSpeed-wrapped)
    _backbone = model.backbone
    if hasattr(_backbone, 'base_model') and hasattr(_backbone.base_model, 'model'):
        _inner = _backbone.base_model.model
    else:
        _inner = _backbone
    if hasattr(_inner, 'visual'):
        _visual = _inner.visual
    elif hasattr(_inner, 'model') and hasattr(_inner.model, 'visual'):
        _visual = _inner.model.visual
    else:
        raise AttributeError(f"Cannot find visual module in {type(_inner)}")
    merge = _visual.spatial_merge_size

    # Round 0: low-res only
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
    vis_hidden, vis_ranges = extract_visual_hidden_states(last_hs, inp["input_ids"], img_tok)
    anchor = extract_anchor_hidden_states(last_hs, inp["input_ids"], pp_id, n=0)

    if vis_hidden is None or anchor is None:
        return {"topk_points": [(0.5, 0.5)], "n_width": 1, "n_height": 1, "num_rounds": 0}

    grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
    nh0, nw0 = grid_dims[0]

    attn0, _ = model.action_head(vis_hidden, anchor)
    attn_1d = attn0.squeeze(0)

    # Use BFS prediction for round 0
    best_pt, _, _ = get_prediction_region_point(attn_1d, nw0, nh0, activation_threshold)
    focus_x, focus_y = best_pt

    decision = saccade.decide_round0(state, focus_x, focus_y)
    nw_final, nh_final = nw0, nh0
    final_point = (focus_x, focus_y)

    # Subsequent rounds: LLM sees full history (all crops accumulated).
    # Action head only considers unmasked low-res + latest crop; old crops masked.
    total_vis_tokens = vis_hidden.shape[0] if vis_hidden is not None else 0

    for ri in range(1, max_rounds):
        if not saccade.should_continue(state, ri):
            break

        # Crop around current focus
        cropped, crop_bbox = crop_image(image, focus_x, focus_y, crop_ratio,
                                        upsample_pixels=crop_upsample_pixels,
                                        crop_target_pixels=crop_target_pixels)

        # Extend context (accumulate all crops for LLM context)
        try:
            ri_inputs, cur_text, cur_images = builder.extend_with_crop(
                cur_text, cur_images, cropped, crop_bbox,
            )
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
        vis_hidden, vis_ranges = extract_visual_hidden_states(last_hs, inp["input_ids"], img_tok)
        anchor = extract_anchor_hidden_states(last_hs, inp["input_ids"], pp_id, n=ri)

        if vis_hidden is None or anchor is None or len(vis_ranges) < 2:
            break

        grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
        nh_low, nw_low = grid_dims[0]
        latest_img_idx = len(vis_ranges) - 1

        # Action head mask: current crop masks low-res + ALL old crops masked out
        this_crop_mask = compute_overlap_mask(nh_low, nw_low, crop_bbox).to(device)
        n_low = vis_ranges[0][1]
        n_total = sum(r[1] for r in vis_ranges)
        full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        full_mask[:n_low] = this_crop_mask
        # Mask all previous crop tokens (indices 1 to latest-1)
        for prev_i in range(1, latest_img_idx):
            off, ntok = vis_ranges[prev_i]
            full_mask[off:off + ntok] = True

        total_vis_tokens = n_total

        attn_ri, _ = model.action_head(vis_hidden, anchor, mask=full_mask)
        attn_1d = attn_ri.squeeze(0)

        # Identify which image has argmax
        img_idx, local_idx = identify_attended_image(attn_1d, vis_ranges)
        info = builder.image_infos[img_idx]

        if img_idx < len(grid_dims):
            nh_a, nw_a = grid_dims[img_idx]
        else:
            break

        off_a, n_a = vis_ranges[img_idx]
        img_attn = attn_1d[off_a:off_a+n_a]
        lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
        bx1, by1, bx2, by2 = info.global_bbox
        global_x = bx1 + lx * (bx2 - bx1)
        global_y = by1 + ly * (by2 - by1)

        attended_source = "high" if info.resolution == "high" else "low"
        decision = saccade.decide_saccade(state, attended_source, global_x, global_y)

        nw_final, nh_final = nw_a, nh_a
        final_point = (global_x, global_y)

        if decision["action"] == "stop":
            break

        # Saccade: update focus
        focus_x, focus_y = global_x, global_y

    return {
        "topk_points": [final_point],
        "n_width": nw_final,
        "n_height": nh_final,
        "num_rounds": len(state.history),
        "total_vis_tokens": total_vis_tokens,
    }
