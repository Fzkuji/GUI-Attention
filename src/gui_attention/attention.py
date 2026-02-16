"""Attention extraction using gui_aima's multi-layer multi-head aggregation.

Replaces the v2 last-layer Q*K approach with gui_aima's full pipeline:
  calculate_attention_from_qk -> visual-sink head weighting ->
  multi_patch_pointer_head_attention -> aggregated (1, n_vis) distribution.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from gui_aima.model_utils import calculate_attention_from_qk


# ---------------------------------------------------------------------------
# Token-range helpers (unchanged from v2)
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
# Core: extract attention via gui_aima multi-layer aggregation
# ---------------------------------------------------------------------------

def extract_attention(model, outputs, input_ids, position_ids, attention_mask,
                      visual_indices, query_indices, target_index):
    """Extract aggregated attention using gui_aima's multi-layer pipeline.

    Steps:
        1. calculate_attention_from_qk -> per-layer per-head attention
        2. Visual-sink query weighting -> head weights
        3. model.multi_patch_pointer_head_attention -> aggregated attention

    Args:
        model: Qwen2_5_VLForConditionalGenerationWithPointer.
        outputs: model forward outputs (with output_hidden_states=True).
        input_ids: (1, seq_len).
        position_ids: (3, 1, seq_len) M-RoPE position IDs.
        attention_mask: (1, seq_len).
        visual_indices: list of int, positions of visual tokens.
        query_indices: list of int, positions of text query tokens.
        target_index: int, position of the pointer_pad token.

    Returns:
        attn_weights: (1, n_vis) normalised attention distribution.
        loss: KL loss if labels were passed via the grounding head, else None.
    """
    hidden_states = list(outputs.hidden_states)

    # 1. Multi-layer attention extraction
    all_attentions = calculate_attention_from_qk(
        model,
        all_hidden_states=[hidden_states],
        all_position_ids=position_ids,
        all_attention_mask=attention_mask,
        query_indices=[target_index],
    )
    # all_attentions[0] = list of per-layer tensors, each (B, H, 1, seq_len)
    layer_attns = all_attentions[0]

    # 2. Visual-sink query weighting
    topk_query_indices = _compute_visual_sink_queries(
        model, hidden_states, visual_indices, query_indices,
    )

    # 3. Aggregate via grounding head
    attn_weights, loss = model.multi_patch_pointer_head_attention(
        query_indices=query_indices,
        visual_indices=visual_indices,
        target_indices=[target_index],
        self_attentions=layer_attns,
        topk_query_indices=topk_query_indices,
        batch_idx=0,
    )

    return attn_weights, loss


def _compute_visual_sink_queries(model, hidden_states, visual_indices, query_indices, topk=None):
    """Compute top-K visual-sink query indices via cosine similarity.

    For each text query token, compute sum of cosine similarities with all
    visual tokens across all layers. Select the top-K queries with highest
    aggregate similarity.

    Returns:
        topk_indices: LongTensor of query_indices positions, or None.
    """
    if not query_indices or not visual_indices:
        return None

    k = topk
    if k is None:
        k = getattr(model.config, 'query_topk', 1)
    if k is None or k <= 0:
        return None

    # Stack hidden states from all layers (skip embedding layer at index 0)
    num_layers = len(hidden_states) - 1
    device = hidden_states[1].device

    q_idx = torch.tensor(query_indices, device=device)
    v_idx = torch.tensor(visual_indices, device=device)

    agg_sim = torch.zeros(len(query_indices), device=device)

    for layer_idx in range(1, len(hidden_states)):
        hs = hidden_states[layer_idx][0]  # (seq_len, hidden_dim)
        q_hs = F.normalize(hs[q_idx], dim=-1)  # (n_query, D)
        v_hs = F.normalize(hs[v_idx], dim=-1)  # (n_visual, D)
        sim = torch.matmul(q_hs, v_hs.T)  # (n_query, n_visual)
        agg_sim += sim.sum(dim=-1)  # (n_query,)

    k = min(k, len(query_indices))
    _, topk_local = torch.topk(agg_sim, k, largest=True)
    return topk_local


# ---------------------------------------------------------------------------
# Multi-image helpers (unchanged from v2)
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

    cumulative = 0
    for img_idx, (vs, ve) in enumerate(visual_ranges):
        n_tokens = ve - vs
        if cumulative + n_tokens > global_argmax:
            return img_idx, global_argmax - cumulative
        cumulative += n_tokens

    return len(visual_ranges) - 1, global_argmax - cumulative


def token_to_spatial(local_token_idx: int, n_width: int, n_height: int):
    """Convert a flat visual token index to normalised (x, y) coordinates."""
    col = local_token_idx % n_width
    row = local_token_idx // n_width
    return (col + 0.5) / n_width, (row + 0.5) / n_height
