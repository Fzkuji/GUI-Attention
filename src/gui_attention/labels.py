"""IoU x Gaussian soft label generation for attention supervision."""

import math
import torch


def compute_soft_labels(n_height: int, n_width: int, gt_bbox: tuple,
                        sigma_scale: float = 0.8) -> torch.Tensor:
    """Generate IoU x Gaussian soft labels over a visual token grid.

    For each visual token at grid position (row, col), computes:
        p(v_i) = IoU(patch_i, gt_bbox) * Gaussian(center_i; bbox_center, sigma)
    then normalises to a probability distribution.

    Args:
        n_height: number of rows in the visual token grid.
        n_width: number of columns in the visual token grid.
        gt_bbox: (x1, y1, x2, y2) normalised ground-truth bounding box.
        sigma_scale: sigma as a fraction of the bbox diagonal.

    Returns:
        (n_height * n_width,) normalised probability distribution.
    """
    x1, y1, x2, y2 = gt_bbox
    gt_cx = (x1 + x2) / 2
    gt_cy = (y1 + y2) / 2
    gt_w = max(x2 - x1, 0)
    gt_h = max(y2 - y1, 0)
    diag = math.sqrt(gt_w ** 2 + gt_h ** 2)
    sigma = max(diag * sigma_scale, 1e-6)
    two_sigma_sq = 2 * sigma ** 2

    patch_w = 1.0 / n_width
    patch_h = 1.0 / n_height
    gt_area = gt_w * gt_h

    labels = torch.zeros(n_height * n_width)

    for row in range(n_height):
        for col in range(n_width):
            px1 = col * patch_w
            py1 = row * patch_h
            px2 = px1 + patch_w
            py2 = py1 + patch_h

            # IoU
            inter_w = max(0.0, min(px2, x2) - max(px1, x1))
            inter_h = max(0.0, min(py2, y2) - max(py1, y1))
            inter_area = inter_w * inter_h
            patch_area = patch_w * patch_h
            union_area = patch_area + gt_area - inter_area
            iou = inter_area / max(union_area, 1e-8)

            # Gaussian
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            dist_sq = (pcx - gt_cx) ** 2 + (pcy - gt_cy) ** 2
            gauss = math.exp(-dist_sq / two_sigma_sq)

            labels[row * n_width + col] = iou * gauss

    total = labels.sum()
    if total > 0:
        labels = labels / total
    else:
        closest_col = min(max(int(gt_cx * n_width), 0), n_width - 1)
        closest_row = min(max(int(gt_cy * n_height), 0), n_height - 1)
        labels[closest_row * n_width + closest_col] = 1.0

    return labels
