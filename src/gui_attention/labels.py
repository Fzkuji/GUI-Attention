"""Binary overlap label generation for action head supervision.

Replaces the v3 IoU x Gaussian soft labels with simple binary overlap masks
matching GUI-Actor's label scheme: any patch overlapping with GT bbox = 1.
"""

import torch


def compute_binary_labels(n_height: int, n_width: int, gt_bbox: tuple) -> torch.Tensor:
    """Generate binary overlap labels over a visual token grid.

    A patch is positive (1) if it has ANY overlap with gt_bbox, else 0.

    Args:
        n_height: number of rows in the visual token grid.
        n_width: number of columns in the visual token grid.
        gt_bbox: (x1, y1, x2, y2) normalised ground-truth bounding box.

    Returns:
        (n_height * n_width,) binary tensor.
    """
    x1, y1, x2, y2 = gt_bbox
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
            labels[row * n_width + col] = 1.0

    # Fallback: if no overlap, set closest patch to 1
    if labels.sum() == 0:
        gt_cx = (x1 + x2) / 2
        gt_cy = (y1 + y2) / 2
        closest_col = min(max(int(gt_cx * n_width), 0), n_width - 1)
        closest_row = min(max(int(gt_cy * n_height), 0), n_height - 1)
        labels[closest_row * n_width + closest_col] = 1.0

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
