"""Attention utilities for action head (v4, no gui_aima dependency).

Extracts LLM last-layer hidden states and feeds them through the ActionHead.
Includes M-RoPE position adjustment for crop images.
"""

from typing import List, Optional, Tuple

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
                     attn_weights=None, blur_sigma: float = 1.0,
                     pixel_size: int = 56):
    """Convert a flat visual token index to normalised (x, y) coordinates.

    If *attn_weights* (1-D tensor of length n_width*n_height) is provided,
    upsample attention to pixel level, apply Gaussian blur, then compute
    a weighted-average coordinate for sub-patch precision.

    Args:
        local_token_idx: flat index of the argmax token (used as fallback).
        n_width: number of tokens per row.
        n_height: number of token rows.
        attn_weights: 1-D attention tensor (n_width * n_height).
        blur_sigma: sigma for Gaussian blur in pixel space.
        pixel_size: pixels per token (spatial_merge_size * patch_size = 2*28 = 56).
    """
    col = local_token_idx % n_width
    row = local_token_idx // n_width

    if attn_weights is not None:
        import torch.nn.functional as F

        attn_2d = attn_weights.detach().float().view(1, 1, n_height, n_width)
        # Upsample to pixel level via nearest
        h_px = n_height * pixel_size
        w_px = n_width * pixel_size
        pixel_attn = F.interpolate(attn_2d, size=(h_px, w_px), mode='nearest')

        # Gaussian blur
        if blur_sigma > 0:
            kernel_size = max(3, int(blur_sigma * 6) | 1)  # ensure odd
            # 1D Gaussian kernel
            x = torch.arange(kernel_size, dtype=torch.float32,
                             device=pixel_attn.device) - kernel_size // 2
            kernel_1d = torch.exp(-0.5 * (x / blur_sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            # Separable 2D convolution
            k_h = kernel_1d.view(1, 1, -1, 1)
            k_w = kernel_1d.view(1, 1, 1, -1)
            pad_h = kernel_size // 2
            pad_w = kernel_size // 2
            pixel_attn = F.pad(pixel_attn, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            pixel_attn = F.conv2d(pixel_attn, k_h)
            pixel_attn = F.conv2d(pixel_attn, k_w)

        # Argmax on blurred pixel-level attention → sub-patch precision
        pixel_attn = pixel_attn.squeeze()  # (h_px, w_px)
        peak_idx = pixel_attn.argmax().item()
        peak_row = peak_idx // w_px
        peak_col = peak_idx % w_px
        # Normalise to [0, 1] — use pixel center (+0.5)
        return (peak_col + 0.5) / w_px, (peak_row + 0.5) / h_px

    return (col + 0.5) / n_width, (row + 0.5) / n_height


# ---------------------------------------------------------------------------
# M-RoPE position adjustment for crop images
# ---------------------------------------------------------------------------

def adjust_crop_position_ids(
    position_ids: torch.Tensor,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int,
    crop_bbox: Tuple[float, float, float, float],
    image_token_id: int,
) -> torch.Tensor:
    """Adjust M-RoPE position_ids so crop image's spatial positions map to
    the corresponding positions in the low-res image's coordinate space.

    The crop's h/w positions are remapped from [0, nh1) to [y1*nh0, y2*nh0)
    and [x1*nw0, x2*nw0) respectively, where (nh0, nw0) is the low-res grid
    and (x1, y1, x2, y2) is the crop bbox in normalised coordinates.

    Args:
        position_ids: (3, B, seq_len) — temporal, height, width.
        input_ids: (B, seq_len) or (seq_len,).
        image_grid_thw: (num_images, 3) — grid dims for each image.
        merge_size: spatial merge size (typically 2).
        crop_bbox: (x1, y1, x2, y2) normalised crop region in original image.
        image_token_id: token id for image padding tokens.

    Returns:
        Modified position_ids with crop spatial positions adjusted.
    """
    if image_grid_thw.shape[0] < 2:
        return position_ids  # No crop image, nothing to adjust

    # Get grid dims for low-res (image 0) and crop (image 1)
    nh0 = (image_grid_thw[0][1] // merge_size).item()
    nw0 = (image_grid_thw[0][2] // merge_size).item()
    nh1 = (image_grid_thw[1][1] // merge_size).item()
    nw1 = (image_grid_thw[1][2] // merge_size).item()

    x1, y1, x2, y2 = crop_bbox

    # Find the crop image's token range (second contiguous block of image tokens)
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    ranges = find_image_visual_ranges(ids, image_token_id)
    if len(ranges) < 2:
        return position_ids  # No second image found

    crop_start, crop_end = ranges[1]
    n_crop_tokens = crop_end - crop_start
    expected = nh1 * nw1
    if n_crop_tokens != expected:
        return position_ids  # Mismatch, skip adjustment

    # Get the base offset: the position value at the start of crop tokens
    # position_ids shape: (3, B, seq_len)
    base_h = position_ids[1, 0, crop_start].item()
    base_w = position_ids[2, 0, crop_start].item()

    # Original h positions for crop: base_h + [0, 0, ..., 1, 1, ..., nh1-1]
    # Original w positions for crop: base_w + [0, 1, ..., nw1-1, 0, 1, ...]
    # New h positions: base_h + linspace(y1*nh0, y2*nh0, nh1)
    # New w positions: base_w + linspace(x1*nw0, x2*nw0, nw1)

    h_mapped = torch.linspace(y1 * nh0, y2 * nh0, nh1).round().long()
    w_mapped = torch.linspace(x1 * nw0, x2 * nw0, nw1).round().long()

    # Build full grid: h_mapped repeated for each column, w_mapped tiled for each row
    # Layout: row-major, token[r * nw1 + c] = grid position (r, c)
    h_grid = h_mapped.unsqueeze(1).expand(nh1, nw1).flatten()  # (nh1*nw1,)
    w_grid = w_mapped.unsqueeze(0).expand(nh1, nw1).flatten()  # (nh1*nw1,)

    # Apply with base offset
    position_ids = position_ids.clone()
    position_ids[1, 0, crop_start:crop_end] = h_grid.to(position_ids.device) + base_h
    position_ids[2, 0, crop_start:crop_end] = w_grid.to(position_ids.device) + base_w

    return position_ids
