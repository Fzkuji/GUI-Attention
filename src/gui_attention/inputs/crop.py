"""Image cropping and coordinate helpers."""

import math

from PIL import Image
from qwen_vl_utils import smart_resize


def crop_image(image: Image.Image, cx_norm, cy_norm, crop_ratio=0.0,
               upsample_pixels: int = 0, crop_target_pixels: int = 0,
               crop_size: int = 0, crop_upscale: int = 1):
    """Crop around normalised centre and optionally upscale.

    Two crop modes:
    1. **Fixed pixel crop** (preferred): ``crop_size > 0`` crops a square of
       ``crop_size × crop_size`` original pixels, then upscales by
       ``crop_upscale`` (integer, e.g. 4 → 168→672). Both crop_size and
       crop_size*crop_upscale should be divisible by 28 for clean token grids.
    2. **Ratio crop** (legacy): ``crop_ratio > 0`` crops a fraction of the
       image dimensions, then resizes via ``crop_target_pixels``.

    Args:
        image: source PIL image.
        cx_norm, cy_norm: normalised crop centre (0-1).
        crop_ratio: fraction of width/height to crop (legacy mode).
        upsample_pixels: (legacy) enlarge small crops to this size.
        crop_target_pixels: (legacy) resize crop to this pixel budget.
        crop_size: fixed crop side length in pixels (e.g. 168). 0=use crop_ratio.
        crop_upscale: integer upscale factor (e.g. 4). Only used with crop_size.

    Returns:
        (cropped_pil, (x1, y1, x2, y2) normalised bbox in original coords).
    """
    W, H = image.size

    if crop_size > 0:
        # --- Fixed pixel crop mode ---
        cw = ch = min(crop_size, W, H)  # clamp to image size
        cx, cy = int(cx_norm * W), int(cy_norm * H)
        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(W, x1 + cw)
        y2 = min(H, y1 + ch)
        # Shift if we hit a boundary
        if x2 - x1 < cw:
            x1 = max(0, x2 - cw)
        if y2 - y1 < ch:
            y1 = max(0, y2 - ch)

        cropped = image.crop((x1, y1, x2, y2))

        # Integer upscale
        if crop_upscale > 1:
            new_w = cropped.size[0] * crop_upscale
            new_h = cropped.size[1] * crop_upscale
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
    else:
        # --- Legacy ratio crop mode ---
        cw, ch = int(W * crop_ratio), int(H * crop_ratio)
        cx, cy = int(cx_norm * W), int(cy_norm * H)
        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(W, x1 + cw)
        y2 = min(H, y1 + ch)
        if x2 - x1 < cw:
            x1 = max(0, x2 - cw)
        if y2 - y1 < ch:
            y1 = max(0, y2 - ch)

        cropped = image.crop((x1, y1, x2, y2))

        # Resize crop to target pixel budget
        if crop_target_pixels > 0:
            cur_w, cur_h = cropped.size
            new_h, new_w = smart_resize(
                cur_h, cur_w, factor=28,
                min_pixels=crop_target_pixels, max_pixels=crop_target_pixels,
            )
            if (new_w, new_h) != (cur_w, cur_h):
                cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
        elif upsample_pixels > 0:
            cur_w, cur_h = cropped.size
            cur_pixels = cur_w * cur_h
            if cur_pixels < upsample_pixels:
                new_h, new_w = smart_resize(
                    cur_h, cur_w, factor=28,
                    min_pixels=upsample_pixels, max_pixels=upsample_pixels,
                )
                cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    return cropped, (x1 / W, y1 / H, x2 / W, y2 / H)


def get_patch_bbox(px_norm, py_norm, n_width, n_height):
    """Return the normalised bbox (x1,y1,x2,y2) of the patch containing (px_norm, py_norm)."""
    col = min(int(px_norm * n_width), n_width - 1)
    row = min(int(py_norm * n_height), n_height - 1)
    return (col / n_width, row / n_height, (col + 1) / n_width, (row + 1) / n_height)


def point_in_bbox(px, py, bbox):
    """Check if point (px, py) falls within bbox (x1, y1, x2, y2)."""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]
