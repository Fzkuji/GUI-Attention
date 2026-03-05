"""Inference utilities for saccade foveation with Dual Head.

Multi-round saccade:
  Round 0: low-res full image → LookHead → argmax → crop around it
  Round N: low-res + ALL historical crops (visible) → LookHead
           → argmax in high-res crop → ClickHead on crop tokens → precise (x,y) → stop
           → argmax in low-res → saccade (continue)
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
    """
    scores = attn_scores.float().cpu()
    max_score = scores.max().item()
    threshold = max_score * activation_threshold

    activated = {}
    for idx in range(scores.numel()):
        if scores[idx].item() > threshold:
            activated[idx] = scores[idx].item()

    if not activated:
        idx = scores.argmax().item()
        x = (idx % n_width + 0.5) / n_width
        y = (idx // n_width + 0.5) / n_height
        return (x, y), [(x, y)], [scores[idx].item()]

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
    crop_size: int = 308,
    crop_upscale: int = 3,
    device: str = "cuda:0",
    activation_threshold: float = 0.3,
    use_click_head: bool = True,
) -> dict:
    """Multi-round saccade inference with LookHead + ClickHead.

    Round 0: low-res full image → LookHead → BFS region → crop center.
    Round 1+: low-res + all crops (visible) → LookHead
              → argmax in high-res → ClickHead on crop → precise (x,y) → stop
              → argmax in low-res → saccade.

    Args:
        use_click_head: if True, use ClickHead for precise positioning when
                       LookHead selects a high-res crop. If False, use
                       LookHead's attention directly (Phase 1 behavior).

    Returns:
        dict with topk_points, n_width, n_height, num_rounds, total_vis_tokens.
    """
    saccade = SaccadeLoop(max_rounds=max_rounds, crop_size=crop_size, crop_upscale=crop_upscale)
    state = saccade.new_state()
    builder.reset()

    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id

    # Find visual module for merge_size
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

    img_w, img_h = image.size

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
    vis_embeds = model.extract_visual_embeds(
        inp["input_ids"], inp.get("pixel_values"), inp.get("image_grid_thw"))
    _, vis_ranges = extract_visual_hidden_states(last_hs, inp["input_ids"], img_tok)
    anchor = extract_anchor_hidden_states(last_hs, inp["input_ids"], pp_id, n=0)

    if vis_embeds is None or anchor is None:
        return {"topk_points": [(0.5, 0.5)], "n_width": 1, "n_height": 1, "num_rounds": 0}

    grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
    nh0, nw0 = grid_dims[0]

    # LookHead on low-res
    attn0, _, _ = model.dual_head.look(vis_embeds, anchor)
    attn_1d = attn0.squeeze(0)

    n_low = vis_ranges[0][1]
    low_attn = attn_1d[:n_low]
    best_pt, _, _ = get_prediction_region_point(low_attn, nw0, nh0, activation_threshold)
    focus_x, focus_y = best_pt

    decision = saccade.decide_round0(state, focus_x, focus_y)
    nw_final, nh_final = nw0, nh0
    final_point = (focus_x, focus_y)
    total_vis_tokens = vis_embeds.shape[0] if vis_embeds is not None else 0

    for ri in range(1, max_rounds):
        if not saccade.should_continue(state, ri):
            break

        # Crop around current focus
        cropped, crop_bbox = crop_image(image, focus_x, focus_y,
                                        crop_size=crop_size,
                                        crop_upscale=crop_upscale)

        # Extend context
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
        vis_embeds = model.extract_visual_embeds(
            inp["input_ids"], inp.get("pixel_values"), inp.get("image_grid_thw"))
        _, vis_ranges = extract_visual_hidden_states(last_hs, inp["input_ids"], img_tok)
        anchor = extract_anchor_hidden_states(last_hs, inp["input_ids"], pp_id, n=ri)

        if vis_embeds is None or anchor is None or len(vis_ranges) < 2:
            break

        grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
        nh_low, nw_low = grid_dims[0]

        # Mask: current crop overlap on low-res
        this_crop_mask = compute_overlap_mask(nh_low, nw_low, crop_bbox).to(device)
        n_low = vis_ranges[0][1]
        n_total = sum(r[1] for r in vis_ranges)
        full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        full_mask[:n_low] = this_crop_mask

        total_vis_tokens = n_total

        # LookHead: decide where attention goes
        attn_ri, _, _ = model.dual_head.look(vis_embeds, anchor, mask=full_mask)
        attn_1d = attn_ri.squeeze(0)

        img_idx, local_idx = identify_attended_image(attn_1d, vis_ranges)
        info = builder.image_infos[img_idx]

        if img_idx < len(grid_dims):
            nh_a, nw_a = grid_dims[img_idx]
        else:
            break

        attended_source = "high" if info.resolution == "high" else "low"

        if attended_source == "high" and use_click_head:
            # LookHead chose a high-res crop → ClickHead on ALL crop tokens
            # Collect all high-res crop tokens
            crop_vis_list = []
            crop_meta = []  # (n_tokens, grid_w, grid_h, global_bbox)
            for ci_idx in range(1, len(vis_ranges)):
                ci_off, ci_n = vis_ranges[ci_idx]
                ci_info = builder.image_infos[ci_idx]
                if ci_info.resolution == "high":
                    crop_vis_list.append(vis_embeds[ci_off:ci_off + ci_n])
                    ci_nh, ci_nw = grid_dims[ci_idx] if ci_idx < len(grid_dims) else (1, 1)
                    crop_meta.append((ci_n, ci_nw, ci_nh, ci_info.global_bbox))

            if crop_vis_list:
                combined_crop_vis = torch.cat(crop_vis_list, dim=0)
                click_attn, _, _ = model.dual_head.click(combined_crop_vis, anchor)
                click_1d = click_attn.squeeze(0)

                # Find which crop token has highest attention
                global_argmax = click_1d.argmax().item()
                running = 0
                for ci_n, ci_nw, ci_nh, ci_bbox in crop_meta:
                    if running + ci_n > global_argmax:
                        local_tok = global_argmax - running
                        lx = ((local_tok % ci_nw) + 0.5) / ci_nw
                        ly = ((local_tok // ci_nw) + 0.5) / ci_nh
                        bx1, by1, bx2, by2 = ci_bbox
                        global_x = bx1 + lx * (bx2 - bx1)
                        global_y = by1 + ly * (by2 - by1)
                        break
                    running += ci_n
                else:
                    # Fallback: use LookHead attention
                    off_a, n_a = vis_ranges[img_idx]
                    img_attn = attn_1d[off_a:off_a + n_a]
                    lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
                    bx1, by1, bx2, by2 = info.global_bbox
                    global_x = bx1 + lx * (bx2 - bx1)
                    global_y = by1 + ly * (by2 - by1)
            else:
                # No crop tokens available, fallback to LookHead
                off_a, n_a = vis_ranges[img_idx]
                img_attn = attn_1d[off_a:off_a + n_a]
                lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
                bx1, by1, bx2, by2 = info.global_bbox
                global_x = bx1 + lx * (bx2 - bx1)
                global_y = by1 + ly * (by2 - by1)
        else:
            # Use LookHead attention directly
            off_a, n_a = vis_ranges[img_idx]
            img_attn = attn_1d[off_a:off_a + n_a]
            lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)

            bx1, by1, bx2, by2 = info.global_bbox
            global_x = bx1 + lx * (bx2 - bx1)
            global_y = by1 + ly * (by2 - by1)

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
