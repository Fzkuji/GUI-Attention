"""Attention utilities for action head (v4, no gui_aima dependency).

Extracts LLM last-layer hidden states and feeds them through the ActionHead.
"""

from typing import List, Tuple

import torch

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
# Hidden state extraction for action head
# ---------------------------------------------------------------------------

def extract_visual_hidden_states(hidden_states, input_ids, image_token_id):
    """Extract visual token hidden states from LLM last layer.

    Args:
        hidden_states: last-layer hidden states (1, seq_len, d_model).
        input_ids: (1, seq_len).
        image_token_id: token ID for image patches.

    Returns:
        visual_hidden: (n_vis, d_model) hidden states at visual positions.
        visual_ranges: list of (start_in_flat, n_tokens) per image block.
    """
    hs = hidden_states[0]  # (seq_len, d_model)
    ids_1d = input_ids[0]
    ranges = find_image_visual_ranges(ids_1d, image_token_id)

    parts = []
    range_offsets = []
    offset = 0
    for vs, ve in ranges:
        n = ve - vs
        parts.append(hs[vs:ve])
        range_offsets.append((offset, n))
        offset += n

    if not parts:
        return None, []
    return torch.cat(parts, dim=0), range_offsets


def extract_anchor_hidden_states(hidden_states, input_ids, pointer_pad_id, n=0):
    """Extract the n-th pointer_pad token's hidden state from LLM last layer.

    Args:
        hidden_states: last-layer hidden states (1, seq_len, d_model).
        input_ids: (1, seq_len).
        pointer_pad_id: token ID(s) for pointer pad.
        n: which pointer_pad to extract (0-based).

    Returns:
        anchor_hidden: (1, d_model) or None if not found.
    """
    pos = find_nth_pointer_pad(input_ids[0], pointer_pad_id, n)
    if pos is None:
        return None
    return hidden_states[0, pos:pos + 1]  # (1, d_model)


# ---------------------------------------------------------------------------
# Multi-image helpers
# ---------------------------------------------------------------------------

def identify_attended_image(
    attn: torch.Tensor,
    visual_ranges: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Given attention over ALL visual tokens, find which image has the max-attended token.

    Args:
        attn: (n_total_vis,) attention weights over concatenated visual tokens.
        visual_ranges: list of (offset, n_tokens) per image.

    Returns:
        image_idx: which image (0-based) the max token belongs to.
        local_token_idx: offset within that image's visual tokens.
    """
    global_argmax = attn.argmax().item()

    cumulative = 0
    for img_idx, (offset, n_tokens) in enumerate(visual_ranges):
        if cumulative + n_tokens > global_argmax:
            return img_idx, global_argmax - cumulative
        cumulative += n_tokens

    return len(visual_ranges) - 1, global_argmax - cumulative


def token_to_spatial(local_token_idx: int, n_width: int, n_height: int,
                     attn_weights=None):
    """Convert a flat visual token index to normalised (x, y) coordinates.

    If *attn_weights* (1-D tensor of length n_width*n_height) is provided,
    use a 3×3 neighbourhood around the argmax token to compute a
    weighted-average position for sub-patch precision.
    """
    col = local_token_idx % n_width
    row = local_token_idx // n_width

    if attn_weights is not None:
        # 3×3 weighted refinement
        attn_2d = attn_weights.detach().float().view(n_height, n_width)
        w_col = 0.0
        w_row = 0.0
        total = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < n_height and 0 <= c < n_width:
                    w = attn_2d[r, c].item()
                    w_row += w * (r + 0.5)
                    w_col += w * (c + 0.5)
                    total += w
        if total > 0:
            return w_col / total / n_width, w_row / total / n_height

    return (col + 0.5) / n_width, (row + 0.5) / n_height
