"""Inference utilities for saccade foveation with Dual Head.

Multi-round saccade:
  Round 0: low-res full image → LookHead
  Round N: low-res + ALL historical crops (visible) → LookHead → saccade
  End: ClickHead on ALL accumulated crop tokens for final precise (x, y)
"""

from typing import List, Tuple

import torch
from transformers import StoppingCriteriaList

from gui_attention.modeling.attention import (
    extract_anchor_hidden_states,
    extract_visual_hidden_states,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.inputs.builder import MultiRoundInputBuilder
from gui_attention.inputs.crop import crop_image, crop_image_bbox
from gui_attention.inputs.labels import compute_overlap_mask
from gui_attention.runtime.foveation import SaccadeLoop
from gui_attention.runtime.proposals import box_center, select_top_proposal_bbox
from gui_attention.runtime.reasoning import (
    ActionSpanStoppingCriteria,
    decode_reasoning_content,
    parse_reasoning_action,
)


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
    reasoning_max_new_tokens: int = 48,
    return_trace: bool = False,
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
    else:
        look_id = getattr(model.config, "look_pad_token_id", None)
        click_id = getattr(model.config, "click_pad_token_id", model.config.pointer_pad_token_id)
    pointer_end_id = model.config.pointer_end_token_id

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

    def _to_device(inputs):
        return {k: v.to(device) for k, v in inputs.items()}

    look_crop_pixels = builder.low_res_max_pixels

    def _attn_np(attn: torch.Tensor):
        return attn.detach().cpu().float().numpy()

    def _fallback_bbox_from_point(center_x, center_y):
        _, fallback_bbox = crop_image(
            image,
            center_x,
            center_y,
            crop_size=crop_size,
            crop_upscale=crop_upscale,
        )
        return fallback_bbox

    def _proposal_bbox_from_attn(
        img_attn_2d: torch.Tensor,
        *,
        parent_bbox=(0.0, 0.0, 1.0, 1.0),
        fallback_center=None,
    ):
        try:
            proposal_bbox, _, _ = select_top_proposal_bbox(
                _attn_np(img_attn_2d),
                parent_bbox=parent_bbox,
            )
        except Exception:
            if fallback_center is None:
                raise
            proposal_bbox = _fallback_bbox_from_point(fallback_center[0], fallback_center[1])
        _, proposal_bbox = crop_image_bbox(
            image,
            proposal_bbox,
            target_pixels=look_crop_pixels,
        )
        return proposal_bbox

    def _build_context(history_used_responses, history_crop_bboxes, *, current_crop=None, current_bbox=None,
                       current_response=None, for_generation=False):
        builder.reset()
        if history_used_responses:
            inputs, cur_text, cur_images = builder.build_round0(
                image_path,
                instruction,
                use_dual_tokens=use_dual_tokens,
                free_reasoning=True,
                assistant_response=history_used_responses[0],
            )
        else:
            inputs, cur_text, cur_images = builder.build_round0(
                image_path,
                instruction,
                use_dual_tokens=use_dual_tokens,
                free_reasoning=True,
                for_generation=(for_generation and current_crop is None),
                assistant_response=(current_response if current_crop is None else None),
            )

        for hist_idx in range(1, len(history_used_responses)):
            prev_crop_bbox = history_crop_bboxes[hist_idx - 1]
            prev_crop, prev_bbox = crop_image_bbox(
                image,
                prev_crop_bbox,
                target_pixels=look_crop_pixels,
            )
            inputs, cur_text, cur_images = builder.extend_with_crop(
                cur_text,
                cur_images,
                prev_crop,
                prev_bbox,
                use_dual_tokens=use_dual_tokens,
                free_reasoning=True,
                assistant_response=history_used_responses[hist_idx],
            )

        if current_crop is not None:
            inputs, cur_text, cur_images = builder.extend_with_crop(
                cur_text,
                cur_images,
                current_crop,
                current_bbox,
                use_dual_tokens=use_dual_tokens,
                free_reasoning=True,
                assistant_response=current_response,
                for_generation=for_generation,
            )
        return inputs, cur_text, cur_images

    def _generate_reasoning(inputs, *, allow_click: bool):
        inp = _to_device(inputs)
        stop = StoppingCriteriaList([
            ActionSpanStoppingCriteria(
                prompt_len=inp["input_ids"].shape[1],
                pointer_end_token_id=pointer_end_id,
                allowed_action_token_ids=[look_id, click_id],
            )
        ])
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
                max_new_tokens=reasoning_max_new_tokens,
                do_sample=False,
                stopping_criteria=stop,
            )
        new_ids = gen_ids[0, inp["input_ids"].shape[1]:].tolist()
        raw_content = decode_reasoning_content(tokenizer, new_ids)
        parsed = parse_reasoning_action(raw_content, allow_click=allow_click)
        return parsed, new_ids

    round_actions = []
    round_raw_responses = []
    round_used_responses = []
    round_format_oks = []
    trace = []

    def _maybe_numpy(attn):
        return attn.detach().cpu().float().numpy() if return_trace and attn is not None else None

    # Round 0: generate reasoning, then always look on low-res.
    r0_prompt, _, _ = _build_context([], [], for_generation=True)
    parsed0, _ = _generate_reasoning(r0_prompt, allow_click=False)
    round_actions.append(parsed0.action)
    round_raw_responses.append(parsed0.raw_content)
    round_used_responses.append(parsed0.used_content)
    round_format_oks.append(parsed0.format_ok)

    r0_inputs, _, _ = _build_context([], [], current_response=parsed0.used_content)
    inp = _to_device(r0_inputs)

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
        return {
            "topk_points": [(0.5, 0.5)],
            "n_width": 1,
            "n_height": 1,
            "num_rounds": 0,
            "round_actions": round_actions,
            "reasoning_texts": round_raw_responses,
            "format_ok_rate": 0.0,
        }

    grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
    nh0, nw0 = grid_dims[0]

    attn0, _, _ = model.dual_head.look(vis_embeds, anchor)
    attn_1d = attn0.squeeze(0)

    n_low = vis_ranges[0][1]
    low_attn = attn_1d[:n_low]
    best_idx = low_attn.argmax().item()
    low_attn_2d = low_attn.view(nh0, nw0)
    peak_x, peak_y = token_to_spatial(best_idx, nw0, nh0, attn_weights=low_attn)
    proposal_bbox = _proposal_bbox_from_attn(
        low_attn_2d,
        parent_bbox=(0.0, 0.0, 1.0, 1.0),
        fallback_center=(peak_x, peak_y),
    )
    focus_x, focus_y = box_center(proposal_bbox)
    round_crop_bboxes = [proposal_bbox]

    saccade.decide_round0(state, focus_x, focus_y)
    nw_final, nh_final = nw0, nh0
    final_point = (focus_x, focus_y)
    total_vis_tokens = vis_embeds.shape[0] if vis_embeds is not None else 0
    last_look_point = final_point
    last_look_dims = (nw0, nh0)
    round_preds = [final_point]
    if return_trace:
        trace.append({
            "round": 0,
            "action": parsed0.action,
            "reasoning_text": parsed0.raw_content,
            "used_reasoning_text": parsed0.used_content,
            "format_ok": parsed0.format_ok,
            "crop_bbox": None,
            "decision_crop_bbox": proposal_bbox,
            "pred_x": focus_x,
            "pred_y": focus_y,
            "attended_image": "low",
            "grid_dims_low": (nh0, nw0),
            "grid_dims_crop": None,
            "low_attn_2d": _maybe_numpy(low_attn_2d),
            "crop_attn_2d": None,
            "point_kind": "look",
        })

    # Keep the latest full context for final ClickHead on accumulated crops.
    last_vis_embeds = None
    last_vis_ranges = None
    last_grid_dims = None
    last_anchor = None

    click_trace_idx = None
    decision_crop_bbox = None
    for ri in range(1, max_rounds):
        current_request_bbox = round_crop_bboxes[ri - 1]
        cropped, crop_bbox = crop_image_bbox(
            image,
            current_request_bbox,
            target_pixels=look_crop_pixels,
        )
        decision_crop_bbox = crop_bbox
        history_responses = list(round_used_responses)
        try:
            ri_prompt, _, _ = _build_context(
                history_responses,
                round_crop_bboxes[:ri - 1],
                current_crop=cropped,
                current_bbox=crop_bbox,
                for_generation=True,
            )
        except Exception:
            break
        parsed, _ = _generate_reasoning(ri_prompt, allow_click=True)
        round_actions.append(parsed.action)
        round_raw_responses.append(parsed.raw_content)
        round_used_responses.append(parsed.used_content)
        round_format_oks.append(parsed.format_ok)

        try:
            ri_inputs_full, _, _ = _build_context(
                history_responses,
                round_crop_bboxes[:ri - 1],
                current_crop=cropped,
                current_bbox=crop_bbox,
                current_response=parsed.used_content,
            )
        except Exception:
            break

        inp = _to_device(ri_inputs_full)

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

        if parsed.action == "click":
            last_look_point = (focus_x, focus_y)
            last_look_dims = (nw0, nh0)
            if return_trace:
                click_trace_idx = len(trace)
                trace.append({
                    "round": ri,
                    "action": parsed.action,
                    "reasoning_text": parsed.raw_content,
                    "used_reasoning_text": parsed.used_content,
                    "format_ok": parsed.format_ok,
                    "crop_bbox": crop_bbox,
                    "decision_crop_bbox": crop_bbox,
                    "pred_x": None,
                    "pred_y": None,
                    "attended_image": "click",
                    "grid_dims_low": (nh0, nw0),
                    "grid_dims_crop": None,
                    "low_attn_2d": None,
                    "crop_attn_2d": None,
                    "point_kind": "click",
                })
            break
        else:
            attn_ri, _, _ = model.dual_head.look(vis_embeds, anchor, mask=full_mask)
            attn_1d = attn_ri.squeeze(0)

            img_idx, local_idx = identify_attended_image(attn_1d, vis_ranges)
            if img_idx >= len(grid_dims) or img_idx >= len(builder.image_infos):
                break
            info = builder.image_infos[img_idx]

            nh_a, nw_a = grid_dims[img_idx]
            off_a, n_a = vis_ranges[img_idx]
            img_attn = attn_1d[off_a:off_a + n_a]
            bx1, by1, bx2, by2 = info.global_bbox
            lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
            fallback_center = (
                bx1 + lx * (bx2 - bx1),
                by1 + ly * (by2 - by1),
            )
            next_bbox = _proposal_bbox_from_attn(
                img_attn.view(nh_a, nw_a),
                parent_bbox=info.global_bbox,
                fallback_center=fallback_center,
            )
            focus_x, focus_y = box_center(next_bbox)
            decision_crop_bbox = next_bbox
            round_crop_bboxes.append(next_bbox)
            round_preds.append((focus_x, focus_y))
            last_look_point = (focus_x, focus_y)
            last_look_dims = (nw_a, nh_a)
            if return_trace:
                crop_off, crop_n = vis_ranges[latest_img_idx]
                crop_attn = attn_1d[crop_off:crop_off + crop_n]
                trace.append({
                    "round": ri,
                    "action": parsed.action,
                    "reasoning_text": parsed.raw_content,
                    "used_reasoning_text": parsed.used_content,
                    "format_ok": parsed.format_ok,
                    "crop_bbox": crop_bbox,
                    "decision_crop_bbox": next_bbox,
                    "pred_x": focus_x,
                    "pred_y": focus_y,
                    "attended_image": info.resolution,
                    "grid_dims_low": (nh0, nw0),
                    "grid_dims_crop": (grid_dims[latest_img_idx][0], grid_dims[latest_img_idx][1]),
                    "low_attn_2d": _maybe_numpy(attn_1d[:n_low].view(nh0, nw0)),
                    "crop_attn_2d": _maybe_numpy(crop_attn.view(grid_dims[latest_img_idx][0], grid_dims[latest_img_idx][1])),
                    "point_kind": "look",
                })

    # Final prediction: ClickHead on all accumulated crop tokens.
    selected_crop_bbox = None
    selected_crop_grid = None
    selected_crop_attn = None
    selected_crop_round = None
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
            for crop_meta_idx, (ci_n, ci_nw, ci_nh, ci_bbox) in enumerate(crop_meta, start=1):
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
                    selected_crop_bbox = ci_bbox
                    selected_crop_grid = (ci_nh, ci_nw)
                    selected_crop_attn = click_1d[running:running + ci_n].view(ci_nh, ci_nw)
                    selected_crop_round = crop_meta_idx
                    break
                running += ci_n
        else:
            final_point = last_look_point
            nw_final, nh_final = last_look_dims
    else:
        # Fallback when no crop exists: use last LookHead prediction.
        final_point = last_look_point
        nw_final, nh_final = last_look_dims

    if return_trace:
        if click_trace_idx is not None and click_trace_idx < len(trace):
            trace[click_trace_idx]["pred_x"] = final_point[0]
            trace[click_trace_idx]["pred_y"] = final_point[1]
            trace[click_trace_idx]["crop_bbox"] = selected_crop_bbox or trace[click_trace_idx]["crop_bbox"]
            trace[click_trace_idx]["grid_dims_crop"] = selected_crop_grid
            trace[click_trace_idx]["crop_attn_2d"] = _maybe_numpy(selected_crop_attn)
            trace[click_trace_idx]["selected_crop_round"] = selected_crop_round
        elif selected_crop_bbox is not None:
            trace.append({
                "round": len(trace),
                "action": "final_click",
                "reasoning_text": "",
                "used_reasoning_text": "",
                "format_ok": True,
                "crop_bbox": selected_crop_bbox,
                "decision_crop_bbox": decision_crop_bbox,
                "pred_x": final_point[0],
                "pred_y": final_point[1],
                "attended_image": "click",
                "grid_dims_low": (nh0, nw0),
                "grid_dims_crop": selected_crop_grid,
                "low_attn_2d": None,
                "crop_attn_2d": _maybe_numpy(selected_crop_attn),
                "point_kind": "click",
                "selected_crop_round": selected_crop_round,
            })

    result = {
        "topk_points": [final_point],
        "n_width": nw_final,
        "n_height": nh_final,
        "num_rounds": len(round_actions),
        "total_vis_tokens": total_vis_tokens,
        "round_actions": round_actions,
        "reasoning_texts": round_raw_responses,
        "used_reasoning_texts": round_used_responses,
        "format_ok_rate": (
            sum(1 for ok in round_format_oks if ok) / len(round_format_oks)
            if round_format_oks else 0.0
        ),
    }
    if return_trace:
        result["trace"] = trace
    return result
