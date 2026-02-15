"""
Unit tests for GUI-Attention utility functions.

Tests the core helper functions that don't depend on gui_aima or model weights.
Run with: python -m pytest tests/test_utils.py -v
"""

import math
import torch
import torch.nn.functional as F
from PIL import Image


# ── Test crop_image ──────────────────────────────────────────────────────────

def test_crop_image_center():
    from gui_attention.crop import crop_image
    img = Image.new("RGB", (1000, 800))
    cropped, bbox = crop_image(img, 0.5, 0.5, 0.3)
    assert cropped.size == (300, 240)
    assert abs(bbox[0] - 0.35) < 0.01
    assert abs(bbox[1] - 0.35) < 0.01


def test_crop_image_corner():
    from gui_attention.crop import crop_image
    img = Image.new("RGB", (1000, 800))
    cropped, bbox = crop_image(img, 0.0, 0.0, 0.3)
    assert bbox[0] == 0.0
    assert bbox[1] == 0.0
    assert cropped.size[0] == 300
    assert cropped.size[1] == 240


def test_crop_image_bottom_right():
    from gui_attention.crop import crop_image
    img = Image.new("RGB", (1000, 800))
    cropped, bbox = crop_image(img, 1.0, 1.0, 0.3)
    assert abs(bbox[2] - 1.0) < 0.01
    assert abs(bbox[3] - 1.0) < 0.01


# ── Test get_patch_bbox ──────────────────────────────────────────────────────

def test_get_patch_bbox_center():
    from gui_attention.crop import get_patch_bbox
    bbox = get_patch_bbox(0.5, 0.5, 10, 10)
    assert bbox == (0.5, 0.5, 0.6, 0.6)


def test_get_patch_bbox_origin():
    from gui_attention.crop import get_patch_bbox
    bbox = get_patch_bbox(0.0, 0.0, 10, 10)
    assert bbox == (0.0, 0.0, 0.1, 0.1)


def test_get_patch_bbox_edge():
    from gui_attention.crop import get_patch_bbox
    bbox = get_patch_bbox(1.0, 1.0, 10, 10)
    assert bbox == (0.9, 0.9, 1.0, 1.0)


# ── Test point_in_bbox ───────────────────────────────────────────────────────

def test_point_in_bbox():
    from gui_attention.crop import point_in_bbox
    assert point_in_bbox(0.5, 0.5, (0.0, 0.0, 1.0, 1.0))
    assert not point_in_bbox(-0.1, 0.5, (0.0, 0.0, 1.0, 1.0))
    assert point_in_bbox(0.0, 0.0, (0.0, 0.0, 1.0, 1.0))


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
    r = _position_reward(0.5, 0.5, [0.4, 0.4, 0.6, 0.6], 1920, 1080)
    assert r > 0.99


def test_position_reward_far():
    r = _position_reward(0.0, 0.0, [0.8, 0.8, 1.0, 1.0], 1920, 1080)
    assert r < 0.2


def test_position_reward_monotonic():
    bbox = [0.4, 0.4, 0.6, 0.6]
    r_close = _position_reward(0.48, 0.48, bbox, 1920, 1080)
    r_far = _position_reward(0.1, 0.1, bbox, 1920, 1080)
    assert r_close > r_far


# ── Test sample_from_attention ───────────────────────────────────────────────

def test_sample_from_attention_range():
    from gui_attention.sampling import sample_from_attention
    n_w, n_h = 10, 8
    attn = torch.rand(1, n_w * n_h)
    attn = attn / attn.sum()
    for _ in range(20):
        px, py, lp, idx = sample_from_attention(attn, n_w, n_h)
        assert 0 <= px <= 1, f"px={px} out of range"
        assert 0 <= py <= 1, f"py={py} out of range"
        assert idx < n_w * n_h


def test_sample_from_attention_peaked():
    from gui_attention.sampling import sample_from_attention
    n_w, n_h = 5, 5
    attn = torch.zeros(1, 25)
    attn[0, 12] = 1.0
    px, py, _, idx = sample_from_attention(attn, n_w, n_h)
    assert idx == 12
    assert abs(px - 0.5) < 0.11
    assert abs(py - 0.5) < 0.11


def test_sample_from_attention_temperature():
    from gui_attention.sampling import sample_from_attention
    n_w, n_h = 5, 5
    attn = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 5.0, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1,
                          0.1, 0.1, 0.1, 0.1, 0.1]])
    attn = attn / attn.sum()
    hits = 0
    for _ in range(50):
        _, _, _, idx = sample_from_attention(attn, n_w, n_h, temperature=0.01)
        if idx == 12:
            hits += 1
    assert hits > 45, f"Expected mostly idx=12 with low temp, got {hits}/50"


# ── Test compute_round_log_prob ──────────────────────────────────────────────

def test_compute_round_log_prob_peaked():
    from gui_attention.sampling import compute_round_log_prob
    attn = torch.zeros(1, 25)
    attn[0, 12] = 1.0
    lp = compute_round_log_prob(attn, (0.5, 0.5), 5, 5, 1.0)
    assert lp.item() > -0.01


def test_compute_round_log_prob_uniform():
    from gui_attention.sampling import compute_round_log_prob
    n = 25
    attn = torch.ones(1, n) / n
    lp = compute_round_log_prob(attn, (0.5, 0.5), 5, 5, 1.0)
    assert abs(lp.item() - (-math.log(n))) < 0.01


# ── Test argmax_prediction ───────────────────────────────────────────────────

def test_argmax_prediction():
    from gui_attention.sampling import argmax_prediction
    attn = torch.zeros(1, 25)
    attn[0, 12] = 1.0
    x, y = argmax_prediction(attn, 5, 5)
    assert abs(x - 0.5) < 0.11
    assert abs(y - 0.5) < 0.11


def test_argmax_prediction_corner():
    from gui_attention.sampling import argmax_prediction
    attn = torch.zeros(1, 25)
    attn[0, 0] = 1.0
    x, y = argmax_prediction(attn, 5, 5)
    assert abs(x - 0.1) < 0.01
    assert abs(y - 0.1) < 0.01


# ── Test find_image_visual_ranges ────────────────────────────────────────────

def test_find_image_visual_ranges():
    from gui_attention.attention import find_image_visual_ranges
    IMG_TOK = 151655
    ids = torch.tensor([100, IMG_TOK, IMG_TOK, IMG_TOK, 200, IMG_TOK, IMG_TOK, 300])
    ranges = find_image_visual_ranges(ids, IMG_TOK)
    assert ranges[0] == (1, 4)
    assert ranges[1] == (5, 7)
    assert len(ranges) == 2


def test_find_image_visual_ranges_trailing():
    from gui_attention.attention import find_image_visual_ranges
    IMG_TOK = 151655
    ids = torch.tensor([100, 200, IMG_TOK, IMG_TOK])
    ranges = find_image_visual_ranges(ids, IMG_TOK)
    assert ranges[0] == (2, 4)


# ── Test find_nth_pointer_pad ────────────────────────────────────────────────

def test_find_nth_pointer_pad():
    from gui_attention.attention import find_nth_pointer_pad
    PP_TOK = 99999
    ids = torch.tensor([100, PP_TOK, 200, PP_TOK, 300])
    assert find_nth_pointer_pad(ids, PP_TOK, 0) == 1
    assert find_nth_pointer_pad(ids, PP_TOK, 1) == 3
    assert find_nth_pointer_pad(ids, PP_TOK, 2) is None


# ── Test precision_for_level ─────────────────────────────────────────────────

def test_precision_for_level():
    from gui_attention.constants import precision_for_level, PRECISION_LEVELS
    assert precision_for_level(0) == PRECISION_LEVELS[0]
    assert precision_for_level(1) == PRECISION_LEVELS[1]
    assert precision_for_level(3) == PRECISION_LEVELS[3]
    # Clamped
    assert precision_for_level(99) == PRECISION_LEVELS[3]


# ── Test FoveationState ──────────────────────────────────────────────────────

def test_foveation_state_round_key():
    from gui_attention.foveation import FoveationState
    state = FoveationState()
    k1 = state._round_key(0.51, 0.49)
    k2 = state._round_key(0.52, 0.48)
    assert k1 == k2, "Close points should map to same key"

    k3 = state._round_key(0.8, 0.2)
    assert k1 != k3, "Distant points should map to different keys"


# ── Test identify_attended_image ─────────────────────────────────────────────

def test_identify_attended_image():
    from gui_attention.attention import identify_attended_image
    attn = torch.zeros(30)
    attn[15] = 1.0
    # 3 images: 10, 10, 10 tokens
    ranges = [(0, 10), (10, 20), (20, 30)]
    img_idx, local = identify_attended_image(attn, ranges)
    assert img_idx == 1
    assert local == 5


# ── Test token_to_spatial ────────────────────────────────────────────────────

def test_token_to_spatial():
    from gui_attention.attention import token_to_spatial
    x, y = token_to_spatial(0, 10, 10)
    assert abs(x - 0.05) < 0.01
    assert abs(y - 0.05) < 0.01

    x, y = token_to_spatial(55, 10, 10)
    assert abs(x - 0.55) < 0.01
    assert abs(y - 0.55) < 0.01


# ── Test metrics truncation ──────────────────────────────────────────────────

def test_metrics_truncation():
    from collections import defaultdict
    metrics = defaultdict(list)
    for i in range(500):
        metrics["reward"].append(float(i))
        metrics["hit_rate"].append(1 if i % 2 == 0 else 0)
    assert len(metrics["reward"]) == 500
    for key in list(metrics.keys()):
        if len(metrics[key]) > 200:
            metrics[key] = metrics[key][-200:]
    assert len(metrics["reward"]) == 200
    assert metrics["reward"][0] == 300.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
