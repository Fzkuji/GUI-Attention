"""
Integration test for training logic.

Tests the core logic (reward computation, advantage, loss structure, foveation
decisions) without needing actual model weights.

Run with: python tests/test_train_logic.py
"""

import math
import torch
import torch.nn.functional as F


def test_advantage_computation():
    """Test that GRPO advantage normalization works correctly.

    Key property: constant reward offset does NOT affect advantages.
    """
    rewards_base = torch.tensor([0.3, 0.5, 0.7, 0.4])
    adv_base = (rewards_base - rewards_base.mean()) / (rewards_base.std() + 1e-8)

    rewards_offset = rewards_base + 0.05
    adv_offset = (rewards_offset - rewards_offset.mean()) / (rewards_offset.std() + 1e-8)

    assert torch.allclose(adv_base, adv_offset, atol=1e-6), \
        "Constant reward offset should not affect advantages"
    print("  PASS: advantage computation invariant to constant offset")


def test_advantage_with_2_generations():
    """With only 2 generations, advantage is essentially +/-."""
    rewards = torch.tensor([0.3, 0.7])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    assert adv[0] < 0 and adv[1] > 0
    print(f"  PASS: 2-generation advantages = {adv.tolist()}")


def test_log_prob_gradient_flow():
    """Test that NLL at a target index produces differentiable output."""
    attn = torch.rand(1, 25, requires_grad=True)
    attn_norm = attn / attn.sum()

    # Simulate _nll_at_target
    target_idx = 12
    log_p = torch.log(attn_norm.clamp(min=1e-10))
    nll = -log_p[0, target_idx]

    nll.backward()
    assert attn.grad is not None
    assert attn.grad.abs().sum() > 0
    print("  PASS: NLL gradient flows correctly")


def test_loss_structure():
    """Test GRPO loss = -advantage * sum(log_probs) averaged over valid generations."""
    rewards = torch.tensor([0.3, 0.5, 0.8])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    log_probs = [
        torch.tensor(-2.0),
        torch.tensor(-1.5),
        torch.tensor(-1.0),
    ]

    total_loss = torch.tensor(0.0)
    for adv, lp in zip(advantages, log_probs):
        total_loss = total_loss + (-adv * lp)
    total_loss = total_loss / len(rewards)

    assert total_loss.isfinite()
    print(f"  PASS: loss = {total_loss.item():.4f} (finite and structured)")


def test_foveation_loop_stop_at_high():
    """FoveationLoop should stop when attending to a stop level."""
    from gui_attention.foveation import FoveationLoop

    fov = FoveationLoop(max_rounds=5)
    state = fov.new_state()

    # Attend to level 0 → should crop to level 1
    decision = fov.decide(state, attended_level=0, global_x=0.5, global_y=0.5)
    assert decision["action"] == "crop"
    assert decision["level"] == 1

    # Attend to level 2 (a stop level) → should stop
    decision = fov.decide(state, attended_level=2, global_x=0.5, global_y=0.5)
    assert decision["action"] == "stop"
    assert state.stopped

    print("  PASS: FoveationLoop stops at high precision level")


def test_foveation_loop_skip_visited():
    """FoveationLoop should skip a level when re-visiting a point."""
    from gui_attention.foveation import FoveationLoop

    fov = FoveationLoop(max_rounds=5)
    state = fov.new_state()

    # First visit to (0.5, 0.5) at level 0 → crop to level 1
    decision = fov.decide(state, attended_level=0, global_x=0.5, global_y=0.5)
    assert decision["action"] == "crop"
    assert decision["level"] == 1

    # Re-visit same point at level 0 → skip to level 2 (stop level)
    decision = fov.decide(state, attended_level=0, global_x=0.5, global_y=0.5)
    # Level 2 is a stop level, but the skip logic adds the crop first
    assert decision["action"] == "crop" or decision["action"] == "stop"
    if decision["action"] == "crop":
        assert decision["level"] == 2

    print("  PASS: FoveationLoop skips levels for visited points")


def test_foveation_loop_max_rounds():
    """FoveationLoop should respect max_rounds."""
    from gui_attention.foveation import FoveationLoop

    fov = FoveationLoop(max_rounds=2)
    state = fov.new_state()

    # Round 0
    fov.decide(state, attended_level=0, global_x=0.3, global_y=0.3)
    assert fov.should_continue(state, current_round=1)

    # Round 1
    fov.decide(state, attended_level=1, global_x=0.4, global_y=0.4)
    assert not fov.should_continue(state, current_round=2)

    print("  PASS: FoveationLoop respects max_rounds")


def test_sft_loss_structure():
    """Test SFT loss: NLL on GT token, averaged across rounds."""
    torch.manual_seed(42)
    nw, nh = 8, 6
    n_tokens = nw * nh

    attn_r0 = torch.rand(1, n_tokens, requires_grad=True)
    attn_r0_norm = attn_r0 / attn_r0.sum()
    attn_r1 = torch.rand(1, n_tokens, requires_grad=True)
    attn_r1_norm = attn_r1 / attn_r1.sum()

    gt_x, gt_y = 0.4, 0.6

    def compute_nll(attn_norm, lx, ly, nw, nh):
        px = max(0, min(round(lx * nw - 0.5), nw - 1))
        py = max(0, min(round(ly * nh - 0.5), nh - 1))
        sidx = py * nw + px
        log_p = torch.log(attn_norm.clamp(min=1e-10))
        return -log_p[0, sidx]

    # Round 0: GT in full image
    loss_r0 = compute_nll(attn_r0_norm, gt_x, gt_y, nw, nh)

    # Round 1: GT in local crop coords
    bx1, by1, bx2, by2 = 0.25, 0.45, 0.55, 0.75
    local_gt_x = (gt_x - bx1) / (bx2 - bx1)
    local_gt_y = (gt_y - by1) / (by2 - by1)
    loss_r1 = compute_nll(attn_r1_norm, local_gt_x, local_gt_y, nw, nh)

    total_loss = (loss_r0 + loss_r1) / 2

    assert total_loss.isfinite()
    assert total_loss.item() > 0

    total_loss.backward()
    assert attn_r0.grad is not None and attn_r0.grad.abs().sum() > 0
    assert attn_r1.grad is not None and attn_r1.grad.abs().sum() > 0

    print(f"  PASS: SFT loss = {total_loss.item():.4f} (positive, finite, grads flow)")


def test_sft_gt_local_coord_mapping():
    """Test that GT coordinate mapping from global to crop-local is correct."""
    gt_x, gt_y = 0.4, 0.6
    crop_ratio = 0.3
    W, H = 100, 100
    cw, ch = int(W * crop_ratio), int(H * crop_ratio)
    cx, cy = int(gt_x * W), int(gt_y * H)
    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(W, x1 + cw)
    y2 = min(H, y1 + ch)
    bx1, by1, bx2, by2 = x1 / W, y1 / H, x2 / W, y2 / H

    local_gt_x = (gt_x - bx1) / (bx2 - bx1)
    local_gt_y = (gt_y - by1) / (by2 - by1)

    assert 0.3 < local_gt_x < 0.7, f"Expected ~0.5, got {local_gt_x}"
    assert 0.3 < local_gt_y < 0.7, f"Expected ~0.5, got {local_gt_y}"
    assert 0.0 <= local_gt_x <= 1.0
    assert 0.0 <= local_gt_y <= 1.0

    print(f"  PASS: GT local coords = ({local_gt_x:.3f}, {local_gt_y:.3f}) (centered in crop)")


def test_sft_peaked_attention_low_loss():
    """SFT loss should be low when attention is peaked on the GT token."""
    nw, nh = 8, 6
    gt_x, gt_y = 0.4, 0.6

    px = max(0, min(round(gt_x * nw - 0.5), nw - 1))
    py = max(0, min(round(gt_y * nh - 0.5), nh - 1))
    gt_idx = py * nw + px

    attn_peaked = torch.ones(1, nw * nh) * 0.001
    attn_peaked[0, gt_idx] = 10.0
    attn_peaked = attn_peaked / attn_peaked.sum()

    attn_uniform = torch.ones(1, nw * nh) / (nw * nh)

    nll_peaked = -torch.log(attn_peaked[0, gt_idx])
    nll_uniform = -torch.log(attn_uniform[0, gt_idx])

    assert nll_peaked < nll_uniform
    assert nll_peaked < 0.5

    print(f"  PASS: peaked NLL={nll_peaked.item():.4f} < uniform NLL={nll_uniform.item():.4f}")


def test_gt_token_in_image():
    """Test _gt_token_in_image helper maps GT coords to correct token index."""
    # Inline the logic from train.py to avoid gui_aima dependency via builder
    def _gt_token_in_image(gt_x, gt_y, global_bbox, nw, nh):
        bx1, by1, bx2, by2 = global_bbox
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:
            return None
        local_x = (gt_x - bx1) / bw
        local_y = (gt_y - by1) / bh
        if not (0 <= local_x <= 1 and 0 <= local_y <= 1):
            return None
        px = max(0, min(round(local_x * nw - 0.5), nw - 1))
        py = max(0, min(round(local_y * nh - 0.5), nh - 1))
        return py * nw + px

    nw, nh = 10, 10

    # Full image: GT at (0.55, 0.55) → token at (5, 5) → flat = 55
    idx = _gt_token_in_image(0.55, 0.55, (0.0, 0.0, 1.0, 1.0), nw, nh)
    assert idx == 55, f"Expected 55, got {idx}"

    # GT outside crop bbox → None
    idx2 = _gt_token_in_image(0.1, 0.1, (0.3, 0.3, 0.6, 0.6), nw, nh)
    assert idx2 is None, "GT outside bbox should return None"

    # GT inside crop: (0.45, 0.45) in bbox (0.3, 0.3, 0.6, 0.6)
    idx3 = _gt_token_in_image(0.45, 0.45, (0.3, 0.3, 0.6, 0.6), nw, nh)
    assert idx3 is not None

    print("  PASS: GT token mapping correct")


def test_identify_attended_image():
    """Test that identify_attended_image maps global argmax to correct image."""
    from gui_attention.attention import identify_attended_image

    # 3 images: 10 tokens, 20 tokens, 15 tokens
    attn = torch.zeros(45)
    attn[25] = 1.0  # max in image 2 (offset 10+20=30 > 25, so image 1, local=15)
    # Wait: image 0 has 10 tokens (0-9), image 1 has 20 tokens (10-29), image 2 has 15 (30-44)
    # Token 25 is in image 1 (local index 25-10=15)

    ranges = [(0, 10), (0, 20), (0, 15)]  # (offset, count) format
    img_idx, local_idx = identify_attended_image(attn, ranges)
    assert img_idx == 1, f"Expected image 1, got {img_idx}"
    assert local_idx == 15, f"Expected local 15, got {local_idx}"

    print("  PASS: identify_attended_image correct")


def test_token_to_spatial():
    """Test flat token index → (x, y) normalized coordinate mapping."""
    from gui_attention.attention import token_to_spatial

    # 5x5 grid, token 12 = center
    x, y = token_to_spatial(12, 5, 5)
    assert abs(x - 0.5) < 0.01
    assert abs(y - 0.5) < 0.01

    # Token 0 = top-left
    x, y = token_to_spatial(0, 10, 10)
    assert abs(x - 0.05) < 0.01
    assert abs(y - 0.05) < 0.01

    # Token 99 = bottom-right in 10x10
    x, y = token_to_spatial(99, 10, 10)
    assert abs(x - 0.95) < 0.01
    assert abs(y - 0.95) < 0.01

    print("  PASS: token_to_spatial correct")


def test_precision_levels():
    """Test precision level constants and helper."""
    from gui_attention.constants import (
        PRECISION_LEVELS, STOP_LEVELS, precision_for_level,
    )

    assert len(PRECISION_LEVELS) == 4
    assert PRECISION_LEVELS[0] < PRECISION_LEVELS[1] < PRECISION_LEVELS[2] < PRECISION_LEVELS[3]
    assert 2 in STOP_LEVELS
    assert 3 in STOP_LEVELS
    assert 0 not in STOP_LEVELS

    assert precision_for_level(0) == PRECISION_LEVELS[0]
    assert precision_for_level(3) == PRECISION_LEVELS[3]
    # Clamped
    assert precision_for_level(10) == PRECISION_LEVELS[3]

    print("  PASS: precision levels correct")


if __name__ == "__main__":
    print("Running training logic tests...\n")
    test_advantage_computation()
    test_advantage_with_2_generations()
    test_log_prob_gradient_flow()
    test_loss_structure()
    test_foveation_loop_stop_at_high()
    test_foveation_loop_skip_visited()
    test_foveation_loop_max_rounds()
    test_sft_loss_structure()
    test_sft_gt_local_coord_mapping()
    test_sft_peaked_attention_low_loss()
    test_gt_token_in_image()
    test_identify_attended_image()
    test_token_to_spatial()
    test_precision_levels()
    print("\nAll tests passed!")
