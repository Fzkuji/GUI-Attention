"""Attention extraction helpers for multi-round foveated inference.

Shared by training (gui_attention.train) and evaluation
(eval/eval_screenspot_pro_aligned.py).
"""

import math

import torch
import torch.nn.functional as F

from gui_aima.inference import calculate_attention_from_qk


def _find_nth_image_visual_range(input_ids, image_token_id, n):
    """Return (start, end) indices of the n-th (0-based) contiguous block of image tokens."""
    ids = input_ids.tolist()
    blocks = []
    in_block = False
    start = 0
    for i, tid in enumerate(ids):
        if tid == image_token_id:
            if not in_block:
                start = i
                in_block = True
        else:
            if in_block:
                blocks.append((start, i))
                in_block = False
    if in_block:
        blocks.append((start, len(ids)))
    return blocks[n] if n < len(blocks) else None


def _find_nth_pointer_pad(input_ids, pointer_pad_id, n):
    """Return index of the n-th pointer_pad token (0-based)."""
    pad_set = set(pointer_pad_id) if isinstance(pointer_pad_id, list) else {pointer_pad_id}
    count = 0
    for i, tid in enumerate(input_ids.tolist()):
        if tid in pad_set:
            if count == n:
                return i
            count += 1
    return None


def get_round_attention(model, input_ids, pixel_values, image_grid_thw,
                        attention_mask, round_idx):
    """Forward the full multi-round sequence and extract attention for a specific
    round's pointer_pad over that round's visual tokens.

    Returns dict with attn_weights (1, n_vis), n_width, n_height  --  or None.
    """
    device = input_ids.device
    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    ps_id = model.config.pointer_start_token_id

    vis_range = _find_nth_image_visual_range(input_ids[0], img_tok, round_idx)
    if vis_range is None:
        return None
    vis_start, vis_end = vis_range
    visual_indices = torch.arange(vis_start, vis_end, device=device)

    target_pos = _find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
    if target_pos is None:
        return None
    target_indices = torch.tensor([target_pos], device=device)

    ps_positions = (input_ids[0] == ps_id).nonzero(as_tuple=False).squeeze(-1)
    ps_before = ps_positions[ps_positions < target_pos]
    query_end = ps_before[-1].item() if len(ps_before) > 0 else target_pos
    query_start = vis_end

    query_indices = torch.arange(query_start, query_end, device=device)
    if getattr(model.config, 'part_query_weighting', False) and len(query_indices) > 12:
        query_indices = query_indices[:-12]
    if query_indices.numel() == 0:
        query_indices = torch.arange(max(0, target_pos - 10), target_pos, device=device)

    merged_indices = torch.cat([query_indices, target_indices], dim=0)

    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=attention_mask,
    )
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )

    hs_per_layer = list(outputs.hidden_states)
    calculated_attention = calculate_attention_from_qk(
        model=model, all_hidden_states=[hs_per_layer],
        all_position_ids=position_ids, query_indices=merged_indices,
        all_attention_mask=attention_mask,
    )

    all_layer_hs = torch.stack(hs_per_layer[1:], dim=0)
    sample_hs = all_layer_hs[:, 0, :, :]
    q_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
    v_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
    sim = torch.einsum('lqd,lvd->lqv', q_hs, v_hs)
    attn_per_query = sim.sum(dim=-1)

    topk_query_indices = None
    global_pattern = None
    if not getattr(model.config, 'kl_query_weighting', False):
        k = getattr(model.config, 'query_topk', 1)
        agg = attn_per_query.sum(dim=0)
        _, topk_query_indices = torch.topk(agg, min(k, len(query_indices)), largest=True)
    else:
        global_pattern = attn_per_query.sum(dim=0).softmax(dim=-1)

    attn_weights, _ = model.multi_patch_pointer_head_attention(
        query_indices, visual_indices, target_indices,
        calculated_attention[0], topk_query_indices, global_pattern,
        batch_idx=0,
    )

    merge = model.visual.spatial_merge_size
    if round_idx < image_grid_thw.shape[0]:
        _, nh, nw = (image_grid_thw[round_idx] // merge).tolist()
    else:
        n_vis = visual_indices.numel()
        nw = nh = int(math.sqrt(n_vis))

    return {"attn_weights": attn_weights, "n_width": int(nw), "n_height": int(nh)}


def forward_for_attention(model, input_ids, pixel_values, image_grid_thw, attention_mask):
    """Run model forward and cache outputs for multi-round attention extraction.

    Use with extract_attention_for_round to avoid redundant forward passes.
    """
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=attention_mask,
    )
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )
    hs_per_layer = list(outputs.hidden_states)
    all_layer_hs = torch.stack(hs_per_layer[1:], dim=0)
    return {
        "position_ids": position_ids,
        "hs_per_layer": hs_per_layer,
        "all_layer_hs": all_layer_hs,
    }


def extract_attention_for_round(model, input_ids, image_grid_thw, attention_mask,
                                cache, round_idx):
    """Extract attention for a specific round from pre-computed forward outputs."""
    device = input_ids.device
    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    ps_id = model.config.pointer_start_token_id

    vis_range = _find_nth_image_visual_range(input_ids[0], img_tok, round_idx)
    if vis_range is None:
        return None
    vis_start, vis_end = vis_range
    visual_indices = torch.arange(vis_start, vis_end, device=device)

    target_pos = _find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
    if target_pos is None:
        return None
    target_indices = torch.tensor([target_pos], device=device)

    ps_positions = (input_ids[0] == ps_id).nonzero(as_tuple=False).squeeze(-1)
    ps_before = ps_positions[ps_positions < target_pos]
    query_end = ps_before[-1].item() if len(ps_before) > 0 else target_pos
    query_start = vis_end

    query_indices = torch.arange(query_start, query_end, device=device)
    if getattr(model.config, 'part_query_weighting', False) and len(query_indices) > 12:
        query_indices = query_indices[:-12]
    if query_indices.numel() == 0:
        query_indices = torch.arange(max(0, target_pos - 10), target_pos, device=device)

    merged_indices = torch.cat([query_indices, target_indices], dim=0)

    position_ids = cache["position_ids"]
    hs_per_layer = cache["hs_per_layer"]
    all_layer_hs = cache["all_layer_hs"]

    calculated_attention = calculate_attention_from_qk(
        model=model, all_hidden_states=[hs_per_layer],
        all_position_ids=position_ids, query_indices=merged_indices,
        all_attention_mask=attention_mask,
    )

    sample_hs = all_layer_hs[:, 0, :, :]
    q_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
    v_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
    sim = torch.einsum('lqd,lvd->lqv', q_hs, v_hs)
    attn_per_query = sim.sum(dim=-1)

    topk_query_indices = None
    global_pattern = None
    if not getattr(model.config, 'kl_query_weighting', False):
        k = getattr(model.config, 'query_topk', 1)
        agg = attn_per_query.sum(dim=0)
        _, topk_query_indices = torch.topk(agg, min(k, len(query_indices)), largest=True)
    else:
        global_pattern = attn_per_query.sum(dim=0).softmax(dim=-1)

    attn_weights, _ = model.multi_patch_pointer_head_attention(
        query_indices, visual_indices, target_indices,
        calculated_attention[0], topk_query_indices, global_pattern,
        batch_idx=0,
    )

    merge = model.visual.spatial_merge_size
    if round_idx < image_grid_thw.shape[0]:
        _, nh, nw = (image_grid_thw[round_idx] // merge).tolist()
    else:
        n_vis = visual_indices.numel()
        nw = nh = int(math.sqrt(n_vis))

    return {"attn_weights": attn_weights, "n_width": int(nw), "n_height": int(nh)}
