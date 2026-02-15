"""Simplified attention extraction: last-layer Q*K with max-peak head selection.

No dependency on gui_aima's grounding head. Works directly with Qwen2.5-VL's
transformer layers, handling GQA and M-RoPE.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# M-RoPE helpers (extracted from transformers to avoid version coupling)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_mrope(x, cos, sin, mrope_section):
    """Apply M-RoPE to a single tensor (Q or K independently)."""
    sec2 = [s * 2 for s in mrope_section]
    # cos/sin shape: (3, batch, seq_len, head_dim)
    # Split head_dim into mrope sections, pick temporal/height/width dims
    cos_c = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(sec2, dim=-1))], dim=-1
    ).unsqueeze(1)  # (batch, 1, seq_len, head_dim) — broadcast over heads
    sin_c = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(sec2, dim=-1))], dim=-1
    ).unsqueeze(1)
    return (x * cos_c) + (_rotate_half(x) * sin_c)


# ---------------------------------------------------------------------------
# Token-range helpers
# ---------------------------------------------------------------------------

def find_image_visual_ranges(input_ids, image_token_id):
    """Return list of (start, end) for each contiguous block of image tokens."""
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
    return blocks


def find_nth_pointer_pad(input_ids, pointer_pad_id, n):
    """Return index of the n-th pointer_pad token (0-based)."""
    pad_set = set(pointer_pad_id) if isinstance(pointer_pad_id, list) else {pointer_pad_id}
    count = 0
    for i, tid in enumerate(input_ids.tolist()):
        if tid in pad_set:
            if count == n:
                return i
            count += 1
    return None


# ---------------------------------------------------------------------------
# Core: last-layer attention extraction
# ---------------------------------------------------------------------------

def extract_last_layer_attention(
    model,
    hidden_states: list,
    position_ids: torch.Tensor,
    pointer_pos: int,
    visual_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Extract the last transformer layer's attention from pointer_pad to visual tokens.

    Args:
        model: Qwen2.5-VL model (or AIMA variant).
        hidden_states: list from model(..., output_hidden_states=True).hidden_states.
            hidden_states[0] = embedding output, hidden_states[i] = output of layer i-1.
        position_ids: (3, batch, seq_len) M-RoPE position IDs.
        pointer_pos: scalar index of pointer_pad in the sequence.
        visual_indices: 1-D LongTensor of visual token positions.

    Returns:
        per_head_attn: (num_heads, n_vis) — attention distribution per Q-head.
        selected_attn: (n_vis,) — attention from the max-peak head.
        best_head_idx: int — index of the selected head.
    """
    last_layer = model.model.layers[-1]
    attn = last_layer.self_attn

    # Input to last layer = output of previous layer, then apply input_layernorm
    h = hidden_states[-2]  # (batch, seq_len, hidden_dim)
    h = last_layer.input_layernorm(h)

    # Project Q (pointer only) and K (visual tokens only)
    ptr_h = h[:, pointer_pos : pointer_pos + 1, :]     # (1, 1, D)
    vis_h = h[:, visual_indices, :]                      # (1, n_vis, D)

    q = attn.q_proj(ptr_h)    # (1, 1, num_heads * head_dim)
    k = attn.k_proj(vis_h)    # (1, n_vis, num_kv_heads * head_dim)

    num_heads = attn.num_heads
    num_kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim
    num_kv_groups = num_heads // num_kv_heads

    q = q.view(1, 1, num_heads, head_dim).transpose(1, 2)         # (1, H, 1, d)
    k = k.view(1, -1, num_kv_heads, head_dim).transpose(1, 2)     # (1, Hkv, n_vis, d)

    # Compute M-RoPE cos/sin for the subset of positions we need
    device = visual_indices.device
    all_idx = torch.cat([torch.tensor([pointer_pos], device=device), visual_indices])
    subset_pos = position_ids[:, :, all_idx]   # (3, batch, 1+n_vis)

    cos, sin = attn.rotary_emb(k, subset_pos)
    # cos, sin: (3, batch, 1+n_vis, head_dim)

    mrope_section = attn.rope_scaling["mrope_section"]

    # Apply RoPE separately to Q (pos 0) and K (pos 1:)
    q = _apply_mrope(q, cos[:, :, :1, :], sin[:, :, :1, :], mrope_section)
    k = _apply_mrope(k, cos[:, :, 1:, :], sin[:, :, 1:, :], mrope_section)

    # GQA: expand K heads to match Q heads
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)   # (1, H, n_vis, d)

    # Scaled dot-product attention
    scaling = getattr(attn, 'scaling', head_dim ** -0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling  # (1, H, 1, n_vis)
    attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)

    per_head = attn_weights[0, :, 0, :]   # (H, n_vis)

    # Select head with the highest peak attention value
    best_head_idx = per_head.max(dim=-1).values.argmax().item()
    selected = per_head[best_head_idx]     # (n_vis,)

    return per_head, selected, best_head_idx


# ---------------------------------------------------------------------------
# High-level: forward + extract for a specific round
# ---------------------------------------------------------------------------

def forward_and_extract(model, input_ids, pixel_values, image_grid_thw,
                        attention_mask, round_idx):
    """Run model forward and extract attention for one round's pointer_pad.

    Convenience wrapper combining forward pass + last-layer extraction.

    Returns:
        dict with keys:
            attn_weights  — (1, n_vis) selected-head attention (for compat)
            per_head_attn — (H, n_vis) all heads
            best_head     — int
            n_width, n_height — spatial grid dims of the attended image
            visual_range  — (start, end) of this round's visual tokens
        or None if the round's tokens are not found.
    """
    device = input_ids.device
    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id

    vis_ranges = find_image_visual_ranges(input_ids[0], img_tok)
    if round_idx >= len(vis_ranges):
        return None
    vis_start, vis_end = vis_ranges[round_idx]
    visual_indices = torch.arange(vis_start, vis_end, device=device)

    target_pos = find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
    if target_pos is None:
        return None

    # Position IDs
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=attention_mask,
    )

    # Forward
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )

    per_head, selected, best_head = extract_last_layer_attention(
        model, list(outputs.hidden_states), position_ids,
        target_pos, visual_indices,
    )

    # Spatial grid dimensions
    merge = model.visual.spatial_merge_size
    if round_idx < image_grid_thw.shape[0]:
        _, nh, nw = (image_grid_thw[round_idx] // merge).tolist()
    else:
        n_vis = visual_indices.numel()
        nw = nh = int(math.sqrt(n_vis))

    return {
        "attn_weights": selected.unsqueeze(0),   # (1, n_vis)
        "per_head_attn": per_head,                # (H, n_vis)
        "best_head": best_head,
        "n_width": int(nw),
        "n_height": int(nh),
        "visual_range": (vis_start, vis_end),
    }


def forward_for_cache(model, input_ids, pixel_values, image_grid_thw, attention_mask):
    """Run model forward and cache outputs for multi-round extraction."""
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=attention_mask,
    )
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )
    return {
        "position_ids": position_ids,
        "hidden_states": list(outputs.hidden_states),
    }


def extract_round_from_cache(model, input_ids, image_grid_thw, cache, round_idx):
    """Extract attention for a specific round from pre-computed forward outputs.

    Returns dict like forward_and_extract, or None.
    """
    device = input_ids.device
    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id

    vis_ranges = find_image_visual_ranges(input_ids[0], img_tok)
    if round_idx >= len(vis_ranges):
        return None
    vis_start, vis_end = vis_ranges[round_idx]
    visual_indices = torch.arange(vis_start, vis_end, device=device)

    target_pos = find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
    if target_pos is None:
        return None

    per_head, selected, best_head = extract_last_layer_attention(
        model, cache["hidden_states"], cache["position_ids"],
        target_pos, visual_indices,
    )

    merge = model.visual.spatial_merge_size
    if round_idx < image_grid_thw.shape[0]:
        _, nh, nw = (image_grid_thw[round_idx] // merge).tolist()
    else:
        n_vis = visual_indices.numel()
        nw = nh = int(math.sqrt(n_vis))

    return {
        "attn_weights": selected.unsqueeze(0),
        "per_head_attn": per_head,
        "best_head": best_head,
        "n_width": int(nw),
        "n_height": int(nh),
        "visual_range": (vis_start, vis_end),
    }


# ---------------------------------------------------------------------------
# Multi-image helpers: identify which image an attended token belongs to
# ---------------------------------------------------------------------------

def identify_attended_image(
    attn: torch.Tensor,
    visual_ranges: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Given attention over ALL visual tokens, find which image has the max-attended token.

    Args:
        attn: (n_total_vis,) attention weights over concatenated visual tokens.
        visual_ranges: list of (start, end) per image in the input_ids sequence.

    Returns:
        image_idx: which image (0-based) the max token belongs to.
        local_token_idx: offset within that image's visual tokens.
    """
    global_argmax = attn.argmax().item()

    # Map global offset into image-specific offset
    cumulative = 0
    for img_idx, (vs, ve) in enumerate(visual_ranges):
        n_tokens = ve - vs
        if cumulative + n_tokens > global_argmax:
            return img_idx, global_argmax - cumulative
        cumulative += n_tokens

    # Fallback: last image
    return len(visual_ranges) - 1, global_argmax - cumulative


def token_to_spatial(local_token_idx: int, n_width: int, n_height: int):
    """Convert a flat visual token index to normalised (x, y) coordinates."""
    col = local_token_idx % n_width
    row = local_token_idx // n_width
    return (col + 0.5) / n_width, (row + 0.5) / n_height
