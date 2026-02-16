"""Image cropping and coordinate helpers."""

from PIL import Image


def crop_image(image: Image.Image, cx_norm, cy_norm, crop_ratio):
    """Crop around normalised centre. Returns (cropped_pil, (x1,y1,x2,y2) normalised)."""
    W, H = image.size
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
    return image.crop((x1, y1, x2, y2)), (x1 / W, y1 / H, x2 / W, y2 / H)


def get_patch_bbox(px_norm, py_norm, n_width, n_height):
    """Return the normalised bbox (x1,y1,x2,y2) of the patch containing (px_norm, py_norm)."""
    col = min(int(px_norm * n_width), n_width - 1)
    row = min(int(py_norm * n_height), n_height - 1)
    return (col / n_width, row / n_height, (col + 1) / n_width, (row + 1) / n_height)


def point_in_bbox(px, py, bbox):
    """Check if point (px, py) falls within bbox (x1, y1, x2, y2)."""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]
