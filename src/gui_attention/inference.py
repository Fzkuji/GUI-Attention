"""Inference utilities for saccade foveation with Dual Head.

Multi-round saccade:
  Round 0: low-res full image → LookHead
  Round N: low-res + ALL historical crops (visible) → LookHead → saccade
  End: ClickHead on ALL accumulated crop tokens for final precise (x, y)
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
    use_dual_tokens: bool = True,
) -> dict:
    """Multi-round saccade inference with LookHead + ClickHead.

    Round 0: low-res full image → LookHead → argmax token decode.
    Round 1+: low-res + all crops (visible) → LookHead
              → decode attended token to global point → continue saccade.
    After all rounds, run ClickHead once on all crop tokens for final output.

    Args:
        use_click_head: if True, run final ClickHead over accumulated crops.
                        If False, return the last LookHead prediction.

    Returns:
        dict with topk_points, n_width, n_height, num_rounds, total_vis_tokens.
    """
    saccade = SaccadeLoop(max_rounds=max_rounds, crop_size=crop_size, crop_upscale=crop_upscale)
    state = saccade.new_state()
    builder.reset()

    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    # Support dual tokens: anchor can be look_pad, click_pad, or pointer_pad
    if use_dual_tokens and hasattr(model.config, 'look_pad_token_id') and hasattr(model.config, 'click_pad_token_id'):
        look_id = model.config.look_pad_token_id
        click_id = model.config.click_pad_token_id
        pp_id = [look_id, click_id, pp_id]

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

    # Round 0: low-res only
    r0_inputs, _, _ = builder.build_round0(
        image_path, instruction, use_dual_tokens=use_dual_tokens
    )
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
    best_idx = low_attn.argmax().item()
    focus_x, focus_y = token_to_spatial(best_idx, nw0, nh0, attn_weights=low_attn)

    saccade.decide_round0(state, focus_x, focus_y)
    nw_final, nh_final = nw0, nh0
    final_point = (focus_x, focus_y)
    total_vis_tokens = vis_embeds.shape[0] if vis_embeds is not None else 0
    last_look_point = final_point
    last_look_dims = (nw0, nh0)
    round_preds = [final_point]

    # Keep the latest full context for final ClickHead on accumulated crops.
    last_vis_embeds = None
    last_vis_ranges = None
    last_grid_dims = None
    last_anchor = None

    for ri in range(1, max_rounds):
        # Crop around current focus
        cropped, crop_bbox = crop_image(image, focus_x, focus_y,
                                        crop_size=crop_size,
                                        crop_upscale=crop_upscale)

        # Rebuild full context to match training: low-res + all historical crops + latest crop.
        builder.reset()
        ri_inputs, cur_text, cur_images = builder.build_round0(
            image_path, instruction, use_dual_tokens=use_dual_tokens
        )
        rebuild_failed = False
        for prev_ri in range(1, ri):
            prev_px, prev_py = round_preds[prev_ri - 1]
            prev_crop, prev_bbox = crop_image(
                image, prev_px, prev_py, crop_size=crop_size, crop_upscale=crop_upscale
            )
            try:
                ri_inputs, cur_text, cur_images = builder.extend_with_crop(
                    cur_text,
                    cur_images,
                    prev_crop,
                    prev_bbox,
                    gt_in_crop=None,
                    use_dual_tokens=use_dual_tokens,
                )
            except Exception:
                rebuild_failed = True
                break
        if rebuild_failed:
            break
        # Build context ending with generation prompt (no suffix)
        try:
            ri_inputs, cur_text, cur_images = builder.extend_with_crop(
                cur_text,
                cur_images,
                cropped,
                crop_bbox,
                gt_in_crop=None,
                use_dual_tokens=use_dual_tokens,
                for_generation=True,
            )
        except Exception:
            break

        inp = {k: v.to(device) for k, v in ri_inputs.items()}

        # Autoregressive generation: let model decide look vs click
        click_pad_id = getattr(model.config, 'click_pad_token_id',
                               model.config.pointer_pad_token_id)
        look_pad_id = getattr(model.config, 'look_pad_token_id', None)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
                max_new_tokens=30,
                do_sample=False,
            )
        # Extract only newly generated tokens
        new_tokens = gen_ids[0, inp["input_ids"].shape[1]:]
        new_token_list = new_tokens.tolist()

        # Check which pad token the model generated
        model_chose_click = click_pad_id in new_token_list
        model_chose_look = look_pad_id in new_token_list if look_pad_id else False

        # Now do a full forward to get hidden states for the action heads
        # Use the FULL generated sequence (input + generated) to match training
        # But we need the suffix to be included for proper anchor extraction
        # Re-build with the correct suffix based on model's decision
        builder.reset()
        ri_inputs_full, cur_text, cur_images = builder.build_round0(
            image_path, instruction, use_dual_tokens=use_dual_tokens
        )
        for prev_ri in range(1, ri):
            prev_px, prev_py = round_preds[prev_ri - 1]
            prev_crop, prev_bbox = crop_image(
                image, prev_px, prev_py, crop_size=crop_size, crop_upscale=crop_upscale
            )
            ri_inputs_full, cur_text, cur_images = builder.extend_with_crop(
                cur_text, cur_images, prev_crop, prev_bbox,
                gt_in_crop=False, use_dual_tokens=use_dual_tokens,
            )
        # Current crop with correct suffix
        ri_inputs_full, cur_text, cur_images = builder.extend_with_crop(
            cur_text, cur_images, cropped, crop_bbox,
            gt_in_crop=model_chose_click, use_dual_tokens=use_dual_tokens,
        )

        inp = {k: v.to(device) for k, v in ri_inputs_full.items()}

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

        if vis_embeds is None or anchor is None or len(vis_ranges) < ri + 1:
            break

        grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)

        # Mask: current crop overlap on low-res + old crop tokens
        this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox).to(device)
        n_low = vis_ranges[0][1]
        n_total = vis_embeds.shape[0]
        latest_img_idx = len(vis_ranges) - 1
        full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        full_mask[:n_low] = this_crop_mask
        for prev_i in range(1, latest_img_idx):
            off_prev, ntok_prev = vis_ranges[prev_i]
            full_mask[off_prev:off_prev + ntok_prev] = True

        total_vis_tokens = n_total

        # Save state for ClickHead
        last_vis_embeds = vis_embeds
        last_vis_ranges = vis_ranges
        last_grid_dims = grid_dims
        last_anchor = anchor

        if model_chose_click:
            # Model generated <pointer_pad> → ClickHead → stop
            last_look_point = (focus_x, focus_y)
            last_look_dims = (nw0, nh0)
            break
        else:
            # Model generated <look_pad> (or neither) → LookHead → saccade
            attn_ri, _, _ = model.dual_head.look(vis_embeds, anchor, mask=full_mask)
            attn_1d = attn_ri.squeeze(0)

            img_idx, local_idx = identify_attended_image(attn_1d, vis_ranges)
            if img_idx >= len(grid_dims) or img_idx >= len(builder.image_infos):
                break
            info = builder.image_infos[img_idx]

            nh_a, nw_a = grid_dims[img_idx]
            off_a, n_a = vis_ranges[img_idx]
            img_attn = attn_1d[off_a:off_a + n_a]
            lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)

            bx1, by1, bx2, by2 = info.global_bbox
            global_x = bx1 + lx * (bx2 - bx1)
            global_y = by1 + ly * (by2 - by1)

            focus_x, focus_y = global_x, global_y
            round_preds.append((focus_x, focus_y))
            last_look_point = (global_x, global_y)
            last_look_dims = (nw_a, nh_a)

    # Final prediction: ClickHead on all accumulated crop tokens.
    if (
        use_click_head
        and last_vis_embeds is not None
        and last_anchor is not None
        and last_vis_ranges is not None
        and last_grid_dims is not None
        and len(last_vis_ranges) > 1
    ):
        crop_vis_list = []
        crop_meta = []  # (n_tokens, grid_w, grid_h, global_bbox)
        for ci_idx in range(1, len(last_vis_ranges)):
            if ci_idx >= len(last_grid_dims) or ci_idx >= len(builder.image_infos):
                continue
            ci_off, ci_n = last_vis_ranges[ci_idx]
            ci_info = builder.image_infos[ci_idx]
            if ci_info.resolution != "high":
                continue
            ci_nh, ci_nw = last_grid_dims[ci_idx]
            crop_vis_list.append(last_vis_embeds[ci_off:ci_off + ci_n])
            crop_meta.append((ci_n, ci_nw, ci_nh, ci_info.global_bbox))

        if crop_vis_list:
            combined_crop_vis = torch.cat(crop_vis_list, dim=0)
            click_attn, _, _ = model.dual_head.click(combined_crop_vis, last_anchor)
            click_1d = click_attn.squeeze(0)
            global_argmax = click_1d.argmax().item()

            running = 0
            for ci_n, ci_nw, ci_nh, ci_bbox in crop_meta:
                if running + ci_n > global_argmax:
                    local_tok = global_argmax - running
                    lx = ((local_tok % ci_nw) + 0.5) / ci_nw
                    ly = ((local_tok // ci_nw) + 0.5) / ci_nh
                    bx1, by1, bx2, by2 = ci_bbox
                    final_point = (
                        bx1 + lx * (bx2 - bx1),
                        by1 + ly * (by2 - by1),
                    )
                    # Return clicked crop grid dims, not low-res dims.
                    nw_final, nh_final = ci_nw, ci_nh
                    break
                running += ci_n
        else:
            final_point = last_look_point
            nw_final, nh_final = last_look_dims
    else:
        # Fallback when no crop exists: use last LookHead prediction.
        final_point = last_look_point
        nw_final, nh_final = last_look_dims

    return {
        "topk_points": [final_point],
        "n_width": nw_final,
        "n_height": nh_final,
        "num_rounds": len(state.history),
        "total_vis_tokens": total_vis_tokens,
    }
