"""
Unit tests for GUI-Attention utility functions.

Tests the core helper functions that don't depend on gui_aima or model weights.
Run with: python -m pytest tests/test_utils.py -v
"""

import math
import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Test crop_image ──────────────────────────────────────────────────────────

def _crop_image(image, cx_norm, cy_norm, crop_ratio):
    """Copy of crop_image from train_grpo_multi_round.py for testing."""
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


def test_crop_image_center():
    img = Image.new("RGB", (1000, 800))
    cropped, bbox = _crop_image(img, 0.5, 0.5, 0.3)
    assert cropped.size == (300, 240)
    assert abs(bbox[0] - 0.35) < 0.01
    assert abs(bbox[1] - 0.35) < 0.01


def test_crop_image_corner():
    """Crop near corner should clamp to image boundary."""
    img = Image.new("RGB", (1000, 800))
    cropped, bbox = _crop_image(img, 0.0, 0.0, 0.3)
    assert bbox[0] == 0.0
    assert bbox[1] == 0.0
    assert cropped.size[0] == 300
    assert cropped.size[1] == 240


def test_crop_image_bottom_right():
    img = Image.new("RGB", (1000, 800))
    cropped, bbox = _crop_image(img, 1.0, 1.0, 0.3)
    assert abs(bbox[2] - 1.0) < 0.01
    assert abs(bbox[3] - 1.0) < 0.01


# ── Test get_patch_bbox ──────────────────────────────────────────────────────

def _get_patch_bbox(px_norm, py_norm, n_width, n_height):
    col = min(int(px_norm * n_width), n_width - 1)
    row = min(int(py_norm * n_height), n_height - 1)
    x1 = col / n_width
    y1 = row / n_height
    x2 = (col + 1) / n_width
    y2 = (row + 1) / n_height
    return (x1, y1, x2, y2)


def test_get_patch_bbox_center():
    bbox = _get_patch_bbox(0.5, 0.5, 10, 10)
    assert bbox == (0.5, 0.5, 0.6, 0.6)


def test_get_patch_bbox_origin():
    bbox = _get_patch_bbox(0.0, 0.0, 10, 10)
    assert bbox == (0.0, 0.0, 0.1, 0.1)


def test_get_patch_bbox_edge():
    """Point at exactly 1.0 should map to last patch, not out of bounds."""
    bbox = _get_patch_bbox(1.0, 1.0, 10, 10)
    assert bbox == (0.9, 0.9, 1.0, 1.0)


# ── Test point_in_bbox ───────────────────────────────────────────────────────

def _point_in_bbox(px, py, bbox):
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


def test_point_in_bbox():
    assert _point_in_bbox(0.5, 0.5, (0.0, 0.0, 1.0, 1.0))
    assert not _point_in_bbox(-0.1, 0.5, (0.0, 0.0, 1.0, 1.0))
    assert _point_in_bbox(0.0, 0.0, (0.0, 0.0, 1.0, 1.0))  # boundary inclusive


# ── Test position_reward ─────────────────────────────────────────────────────

def _position_reward(pred_x, pred_y, bbox_gt, img_w, img_h):
    gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
    gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2
    dist_px = math.sqrt(((pred_x - gt_cx) * img_w) ** 2 +
                        ((pred_y - gt_cy) * img_h) ** 2)
    bbox_diag = max(math.sqrt(((bbox_gt[2] - bbox_gt[0]) * img_w) ** 2 +
                              ((bbox_gt[3] - bbox_gt[1]) * img_h) ** 2), 1.0)
    return 1.0 / (dist_px / bbox_diag + 1.0)


def test_position_reward_perfect():
    """Prediction at GT center should give reward close to 1."""
    r = _position_reward(0.5, 0.5, [0.4, 0.4, 0.6, 0.6], 1920, 1080)
    assert r > 0.99


def test_position_reward_far():
    """Prediction far from GT should give low reward."""
    r = _position_reward(0.0, 0.0, [0.8, 0.8, 1.0, 1.0], 1920, 1080)
    assert r < 0.2


def test_position_reward_monotonic():
    """Closer predictions should get higher rewards."""
    bbox = [0.4, 0.4, 0.6, 0.6]
    r_close = _position_reward(0.48, 0.48, bbox, 1920, 1080)
    r_far = _position_reward(0.1, 0.1, bbox, 1920, 1080)
    assert r_close > r_far


# ── Test sample_from_attention ───────────────────────────────────────────────

def _sample_from_attention(attn_weights, n_w, n_h, temperature=1.0):
    if temperature != 1.0:
        logits = torch.log(attn_weights.clamp(min=1e-10)) / temperature
        probs = F.softmax(logits, dim=-1)
    else:
        probs = attn_weights
    p = probs.squeeze(0).float()
    p = p / p.sum().clamp(min=1e-8)
    dist = torch.distributions.Categorical(probs=p)
    idx = dist.sample()
    lp = dist.log_prob(idx)
    px = idx.item() % n_w
    py = idx.item() // n_w
    return (px + 0.5) / n_w, (py + 0.5) / n_h, lp, idx.item()


def test_sample_from_attention_range():
    """Output coordinates should be in [0, 1]."""
    n_w, n_h = 10, 8
    attn = torch.rand(1, n_w * n_h)
    attn = attn / attn.sum()
    for _ in range(20):
        px, py, lp, idx = _sample_from_attention(attn, n_w, n_h)
        assert 0 <= px <= 1, f"px={px} out of range"
        assert 0 <= py <= 1, f"py={py} out of range"
        assert idx < n_w * n_h


def test_sample_from_attention_peaked():
    """With peaked distribution, should almost always sample the peak."""
    n_w, n_h = 5, 5
    attn = torch.zeros(1, 25)
    attn[0, 12] = 1.0  # center
    px, py, _, idx = _sample_from_attention(attn, n_w, n_h)
    assert idx == 12
    assert abs(px - 0.5) < 0.11
    assert abs(py - 0.5) < 0.11


def test_sample_from_attention_temperature():
    """Low temperature should make distribution more peaked."""
    n_w, n_h = 5, 5
    attn = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 5.0, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1]])
    attn = attn / attn.sum()
    # With very low temperature, should almost always pick idx=12
    hits = 0
    for _ in range(50):
        _, _, _, idx = _sample_from_attention(attn, n_w, n_h, temperature=0.01)
        if idx == 12:
            hits += 1
    assert hits > 45, f"Expected mostly idx=12 with low temp, got {hits}/50"


# ── Test _compute_round_log_prob ─────────────────────────────────────────────

def _compute_round_log_prob(attn_weights, local_coords, nw, nh, temperature):
    lx, ly = local_coords
    px = max(0, min(round(lx * nw - 0.5), nw - 1))
    py = max(0, min(round(ly * nh - 0.5), nh - 1))
    sidx = py * nw + px
    if temperature != 1.0:
        logits = torch.log(attn_weights.clamp(min=1e-10)) / temperature
        log_p = F.log_softmax(logits, dim=-1)
    else:
        log_p = torch.log(attn_weights.clamp(min=1e-10))
    return log_p[0, sidx]


def test_compute_round_log_prob_peaked():
    """Log prob at peaked position should be close to 0."""
    attn = torch.zeros(1, 25)
    attn[0, 12] = 1.0
    lp = _compute_round_log_prob(attn, (0.5, 0.5), 5, 5, 1.0)
    assert lp.item() > -0.01  # log(1) = 0


def test_compute_round_log_prob_uniform():
    """Log prob under uniform should be -log(N)."""
    n = 25
    attn = torch.ones(1, n) / n
    lp = _compute_round_log_prob(attn, (0.5, 0.5), 5, 5, 1.0)
    assert abs(lp.item() - (-math.log(n))) < 0.01


# ── Test _find_nth_image_visual_range ────────────────────────────────────────

def _find_nth_image_visual_range(input_ids, image_token_id, n):
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
    if n < len(blocks):
        return blocks[n]
    return None


def test_find_nth_image_visual_range():
    IMG_TOK = 151655
    # [text, img, img, img, text, img, img, text]
    ids = torch.tensor([100, IMG_TOK, IMG_TOK, IMG_TOK, 200, IMG_TOK, IMG_TOK, 300])
    assert _find_nth_image_visual_range(ids, IMG_TOK, 0) == (1, 4)
    assert _find_nth_image_visual_range(ids, IMG_TOK, 1) == (5, 7)
    assert _find_nth_image_visual_range(ids, IMG_TOK, 2) is None


def test_find_nth_trailing_block():
    """Image block at the end of sequence."""
    IMG_TOK = 151655
    ids = torch.tensor([100, 200, IMG_TOK, IMG_TOK])
    assert _find_nth_image_visual_range(ids, IMG_TOK, 0) == (2, 4)


# ── Test precision_for_round ─────────────────────────────────────────────────

def test_precision_for_round():
    PRECISION_LOW = 1_003_520
    PRECISION_HIGH = 5_760_000

    def precision_for_round(round_idx):
        return PRECISION_LOW if round_idx == 0 else PRECISION_HIGH

    assert precision_for_round(0) == PRECISION_LOW
    assert precision_for_round(1) == PRECISION_HIGH
    assert precision_for_round(5) == PRECISION_HIGH


# ── Test metrics truncation ──────────────────────────────────────────────────

def test_metrics_truncation():
    """Simulate metrics accumulation and truncation logic."""
    from collections import defaultdict
    metrics = defaultdict(list)

    # Accumulate 500 entries
    for i in range(500):
        metrics["reward"].append(float(i))
        metrics["hit_rate"].append(1 if i % 2 == 0 else 0)

    assert len(metrics["reward"]) == 500

    # Apply truncation (same logic as in train loop)
    for key in list(metrics.keys()):
        if len(metrics[key]) > 200:
            metrics[key] = metrics[key][-200:]

    assert len(metrics["reward"]) == 200
    assert metrics["reward"][0] == 300.0  # kept last 200 of 0..499


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
