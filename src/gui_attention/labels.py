"""Label generation for action head supervision.

Supports binary overlap labels (v4) and pixel-integrated Gaussian soft
labels (v5).  Soft labels are computed by:
  1. Creating a pixel-level Gaussian heatmap centred on the GT point.
  2. Averaging the pixel values within each patch's spatial extent.
This encodes sub-patch positional information: the relative weights of
neighbouring patches implicitly represent where within the patch grid
the GT target lies, enabling blur+argmax at inference to recover
pixel-level coordinates.
"""

import math
import torch


def compute_binary_labels(n_height: int, n_width: int, gt_bbox: tuple,
                          soft: bool = True, sigma_scale: float = 2.0,
                          pixel_size: int = 56) -> torch.Tensor:
    """Generate labels over a visual token grid.

    When *soft=False* (legacy): binary labels — any patch overlapping GT = 1.
    When *soft=True* (default): pixel-integrated Gaussian labels.
      1. Build a pixel-level Gaussian heatmap (n_height*pixel_size,
         n_width*pixel_size) centred on the GT bbox centre.
      2. Average-pool each patch region to get the patch weight.
    σ is defined in pixel space as sigma_scale * pixel_size.

    Args:
        n_height: number of rows in the visual token grid.
        n_width: number of columns in the visual token grid.
        gt_bbox: (x1, y1, x2, y2) normalised ground-truth bounding box.
        soft: if True, use pixel-integrated Gaussian; otherwise binary.
        sigma_scale: σ multiplier (σ_pixels = sigma_scale * pixel_size).
        pixel_size: pixels per token (spatial_merge_size * patch_size).

    Returns:
        (n_height * n_width,) float tensor.
    """
    x1, y1, x2, y2 = gt_bbox
    gt_cx = (x1 + x2) / 2
    gt_cy = (y1 + y2) / 2

    if not soft:
        # Legacy binary labels
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

        if labels.sum() == 0:
            closest_col = min(max(int(gt_cx * n_width), 0), n_width - 1)
            closest_row = min(max(int(gt_cy * n_height), 0), n_height - 1)
            labels[closest_row * n_width + closest_col] = 1.0
        return labels

    # --- Pixel-integrated Gaussian soft labels ---
    h_px = n_height * pixel_size
    w_px = n_width * pixel_size
    sigma = sigma_scale * pixel_size

    # GT centre in pixel coordinates
    cx_px = gt_cx * w_px
    cy_px = gt_cy * h_px

    # Build 1D Gaussians (separable) for efficiency
    xs = torch.arange(w_px, dtype=torch.float32) + 0.5  # pixel centres
    ys = torch.arange(h_px, dtype=torch.float32) + 0.5
    gx = torch.exp(-0.5 * ((xs - cx_px) / sigma) ** 2)
    gy = torch.exp(-0.5 * ((ys - cy_px) / sigma) ** 2)

    # 2D Gaussian via outer product: (h_px, w_px)
    heatmap = gy.unsqueeze(1) * gx.unsqueeze(0)

    # Average-pool into patch grid: reshape → (n_height, pixel_size, n_width, pixel_size) → mean
    heatmap = heatmap.view(n_height, pixel_size, n_width, pixel_size)
    labels = heatmap.mean(dim=(1, 3))  # (n_height, n_width)
    labels = labels.reshape(-1)  # (n_height * n_width,)

    # Normalise so max = 1
    max_val = labels.max()
    if max_val > 0:
        labels = labels / max_val

    # Fallback: if all zeros (shouldn't happen with Gaussian, but safety)
    if labels.sum() == 0:
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
