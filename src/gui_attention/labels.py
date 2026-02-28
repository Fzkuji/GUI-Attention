"""Label generation for action head supervision.

Supports binary overlap labels (v4) and Gaussian-weighted soft labels (v5).
Soft labels encode sub-patch positional information: patches closer to the
GT bbox centre receive higher weights, encouraging the model to concentrate
attention precisely on the target rather than spreading it uniformly across
all overlapping patches.
"""

import math
import torch


def compute_binary_labels(n_height: int, n_width: int, gt_bbox: tuple,
                          soft: bool = True, sigma_scale: float = 2.0) -> torch.Tensor:
    """Generate labels over a visual token grid.

    When *soft=False* (legacy): binary labels — any patch overlapping GT = 1.
    When *soft=True* (default): Gaussian-weighted labels centred on the GT
    bbox centre.  Overlapping patches get weight ∝ exp(-d²/2σ²) where d is
    the distance from the patch centre to the GT centre (in grid units) and
    σ = sigma_scale (default 2.0 grid cells).  Non-overlapping patches = 0.
    The result is normalised so max = 1.

    Args:
        n_height: number of rows in the visual token grid.
        n_width: number of columns in the visual token grid.
        gt_bbox: (x1, y1, x2, y2) normalised ground-truth bounding box.
        soft: if True, use Gaussian weighting; otherwise binary.
        sigma_scale: σ in grid-cell units for the Gaussian (only if soft=True).

    Returns:
        (n_height * n_width,) float tensor.
    """
    x1, y1, x2, y2 = gt_bbox
    gt_cx = (x1 + x2) / 2
    gt_cy = (y1 + y2) / 2
    patch_w = 1.0 / n_width
    patch_h = 1.0 / n_height

    labels = torch.zeros(n_height * n_width)
    for row in range(n_height):
        py1 = row * patch_h
        py2 = py1 + patch_h
        if py2 <= y1 or py1 >= y2:
            continue
        for col in range(n_width):
            px1 = col * patch_w
            px2 = px1 + patch_w
            if px2 <= x1 or px1 >= x2:
                continue

            if soft:
                # Patch centre in normalised coords
                pcx = (col + 0.5) / n_width
                pcy = (row + 0.5) / n_height
                # Distance in grid-cell units
                dx = (pcx - gt_cx) / patch_w
                dy = (pcy - gt_cy) / patch_h
                dist_sq = dx * dx + dy * dy
                labels[row * n_width + col] = math.exp(-dist_sq / (2.0 * sigma_scale * sigma_scale))
            else:
                labels[row * n_width + col] = 1.0

    # Fallback: if no overlap, set closest patch
    if labels.sum() == 0:
        closest_col = min(max(int(gt_cx * n_width), 0), n_width - 1)
        closest_row = min(max(int(gt_cy * n_height), 0), n_height - 1)
        labels[closest_row * n_width + closest_col] = 1.0

    # Normalise so max = 1
    max_val = labels.max()
    if max_val > 0:
        labels = labels / max_val

    return labels


def compute_overlap_mask(n_height: int, n_width: int, crop_bbox: tuple) -> torch.Tensor:
    """Compute a boolean mask for low-res patches covered by a high-res crop.

    Returns True for patches that overlap with crop_bbox (should be masked out
    in the action head logits when high-res crop is available).

    Args:
        n_height: number of rows in the low-res visual token grid.
        n_width: number of columns in the low-res visual token grid.
        crop_bbox: (x1, y1, x2, y2) normalised crop bounding box.

    Returns:
        (n_height * n_width,) bool tensor. True = masked.
    """
    x1, y1, x2, y2 = crop_bbox
    patch_w = 1.0 / n_width
    patch_h = 1.0 / n_height

    mask = torch.zeros(n_height * n_width, dtype=torch.bool)
    for row in range(n_height):
        py1 = row * patch_h
        py2 = py1 + patch_h
        if py2 <= y1 or py1 >= y2:
            continue
        for col in range(n_width):
            px1 = col * patch_w
            px2 = px1 + patch_w
            if px2 <= x1 or px1 >= x2:
                continue
            mask[row * n_width + col] = True

    return mask
