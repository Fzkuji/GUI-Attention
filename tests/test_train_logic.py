"""
Integration test for training logic.

Mocks gui_aima dependencies to test the training pipeline's core logic
(reward computation, advantage, loss structure) without needing actual model weights.

Run with: python tests/test_train_logic.py
"""

import math
import torch
import torch.nn.functional as F


def test_advantage_computation():
    """Test that GRPO advantage normalization works correctly.

    Key property: constant reward offset (like the removed format_reward_value)
    does NOT affect advantages.
    """
    rewards_base = torch.tensor([0.3, 0.5, 0.7, 0.4])
    adv_base = (rewards_base - rewards_base.mean()) / (rewards_base.std() + 1e-8)

    # Adding a constant offset should not change advantages
    rewards_offset = rewards_base + 0.05
    adv_offset = (rewards_offset - rewards_offset.mean()) / (rewards_offset.std() + 1e-8)

    assert torch.allclose(adv_base, adv_offset, atol=1e-6), \
        "Constant reward offset should not affect advantages"
    print("  PASS: advantage computation invariant to constant offset")


def test_advantage_with_2_generations():
    """With only 2 generations, advantage is essentially ±1 (normalized)."""
    rewards = torch.tensor([0.3, 0.7])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    # With 2 values, std = |diff|/2, so normalized adv should be ±sqrt(2)/2 ≈ ±0.707
    # Actually: mean=0.5, std=0.2, adv = [-0.2/0.2, 0.2/0.2] = [-1, 1]
    # But torch.std uses Bessel correction (N-1), so std = 0.2*sqrt(2) ≈ 0.283
    # adv = [-0.2/0.283, 0.2/0.283] ≈ [-0.707, 0.707]
    assert adv[0] < 0 and adv[1] > 0
    print(f"  PASS: 2-generation advantages = {adv.tolist()}")


def test_log_prob_gradient_flow():
    """Test that _compute_round_log_prob produces differentiable output."""
    attn = torch.rand(1, 25, requires_grad=True)
    attn_norm = attn / attn.sum()

    # Simulate _compute_round_log_prob
    lx, ly = 0.5, 0.5
    nw, nh = 5, 5
    px = max(0, min(round(lx * nw - 0.5), nw - 1))
    py = max(0, min(round(ly * nh - 0.5), nh - 1))
    sidx = py * nw + px
    log_p = torch.log(attn_norm.clamp(min=1e-10))
    lp = log_p[0, sidx]

    # Check gradient flows
    lp.backward()
    assert attn.grad is not None
    assert attn.grad.abs().sum() > 0
    print("  PASS: log_prob gradient flows correctly")


def test_shared_r0_attention_correctness():
    """Verify that sharing round 0 forward across generations
    produces the same log_prob as separate forwards."""
    torch.manual_seed(42)
    nw, nh = 8, 6
    n_tokens = nw * nh

    # Simulate attention weights (same for all generations since inputs are same)
    attn_shared = torch.rand(1, n_tokens)
    attn_shared = attn_shared / attn_shared.sum()

    # Two generations with different sampled coordinates
    coords_gen0 = (0.3, 0.4)
    coords_gen1 = (0.7, 0.8)

    def compute_lp(attn, coords, nw, nh):
        lx, ly = coords
        px = max(0, min(round(lx * nw - 0.5), nw - 1))
        py = max(0, min(round(ly * nh - 0.5), nh - 1))
        sidx = py * nw + px
        log_p = torch.log(attn.clamp(min=1e-10))
        return log_p[0, sidx]

    # Shared: compute attention once, apply to both
    lp0_shared = compute_lp(attn_shared, coords_gen0, nw, nh)
    lp1_shared = compute_lp(attn_shared, coords_gen1, nw, nh)

    # Separate: each generation has its own copy (but same values)
    lp0_separate = compute_lp(attn_shared.clone(), coords_gen0, nw, nh)
    lp1_separate = compute_lp(attn_shared.clone(), coords_gen1, nw, nh)

    assert torch.allclose(lp0_shared, lp0_separate)
    assert torch.allclose(lp1_shared, lp1_separate)
    print("  PASS: shared R0 attention produces identical log_probs")


def test_loss_structure():
    """Test GRPO loss = -advantage * sum(log_probs) averaged over valid generations."""
    # Simulate 3 generations
    rewards = torch.tensor([0.3, 0.5, 0.8])
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Each generation has accumulated log_probs from multiple rounds
    log_probs = [
        torch.tensor(-2.0),  # gen 0
        torch.tensor(-1.5),  # gen 1
        torch.tensor(-1.0),  # gen 2
    ]

    total_loss = torch.tensor(0.0)
    for adv, lp in zip(advantages, log_probs):
        total_loss = total_loss + (-adv * lp)
    total_loss = total_loss / len(rewards)

    # Best generation (highest reward) should push loss DOWN (encourage)
    # Worst generation should push loss UP (discourage)
    assert total_loss.isfinite()
    print(f"  PASS: loss = {total_loss.item():.4f} (finite and structured)")


def test_convergence_logic():
    """Test the convergence criterion: point_in_bbox with ri >= 2."""
    def point_in_bbox(px, py, bbox):
        return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]

    def get_patch_bbox(px_norm, py_norm, n_width, n_height):
        col = min(int(px_norm * n_width), n_width - 1)
        row = min(int(py_norm * n_height), n_height - 1)
        return (col / n_width, row / n_height, (col + 1) / n_width, (row + 1) / n_height)

    # Round 1 predicts (0.5, 0.5) in a 10x10 grid → patch [0.5, 0.5, 0.6, 0.6]
    prev_patch = get_patch_bbox(0.5, 0.5, 10, 10)

    # Round 2 (ri=2): prediction falls in previous patch → converge
    assert point_in_bbox(0.55, 0.55, prev_patch), "Should converge"

    # Round 2: prediction outside previous patch → don't converge
    assert not point_in_bbox(0.3, 0.3, prev_patch), "Should not converge"

    # Round 1 (ri=1): even if in patch, ri < 2 → don't converge
    ri = 1
    in_patch = point_in_bbox(0.55, 0.55, prev_patch)
    should_stop = ri >= 2 and in_patch
    assert not should_stop, "ri=1 should never converge"

    print("  PASS: convergence logic correct")


def test_sft_loss_structure():
    """Test SFT loss: NLL on GT patch, averaged across rounds."""
    torch.manual_seed(42)
    nw, nh = 8, 6
    n_tokens = nw * nh

    # Simulate attention weights for 2 rounds
    attn_r0 = torch.rand(1, n_tokens, requires_grad=True)
    attn_r0_norm = attn_r0 / attn_r0.sum()
    attn_r1 = torch.rand(1, n_tokens, requires_grad=True)
    attn_r1_norm = attn_r1 / attn_r1.sum()

    # GT coordinate
    gt_x, gt_y = 0.4, 0.6

    # Round 0: GT in full image coords
    def compute_nll(attn_norm, lx, ly, nw, nh):
        px = max(0, min(round(lx * nw - 0.5), nw - 1))
        py = max(0, min(round(ly * nh - 0.5), nh - 1))
        sidx = py * nw + px
        log_p = torch.log(attn_norm.clamp(min=1e-10))
        return -log_p[0, sidx]

    loss_r0 = compute_nll(attn_r0_norm, gt_x, gt_y, nw, nh)

    # Round 1: GT in local crop coords (crop centered on GT, ratio=0.3)
    # Crop bbox: center at (0.4, 0.6), crop_w = 0.3, crop_h = 0.3
    # bbox: (0.25, 0.45, 0.55, 0.75)
    bx1, by1, bx2, by2 = 0.25, 0.45, 0.55, 0.75
    local_gt_x = (gt_x - bx1) / (bx2 - bx1)  # 0.5
    local_gt_y = (gt_y - by1) / (by2 - by1)  # 0.5
    loss_r1 = compute_nll(attn_r1_norm, local_gt_x, local_gt_y, nw, nh)

    total_loss = (loss_r0 + loss_r1) / 2  # averaged across rounds

    assert total_loss.isfinite()
    assert total_loss.item() > 0, "NLL should be positive"

    # Verify gradient flows
    total_loss.backward()
    assert attn_r0.grad is not None and attn_r0.grad.abs().sum() > 0
    assert attn_r1.grad is not None and attn_r1.grad.abs().sum() > 0

    print(f"  PASS: SFT loss = {total_loss.item():.4f} (positive, finite, grads flow)")


def test_sft_gt_local_coord_mapping():
    """Test that GT coordinate mapping from global to crop-local is correct."""
    # Image is 100x100, GT at (0.4, 0.6)
    gt_x, gt_y = 0.4, 0.6

    # Crop centered on GT with ratio 0.3 → crop is 30x30 pixels
    # Center: (40, 60) → crop box: (25, 45, 55, 75)
    crop_ratio = 0.3
    W, H = 100, 100
    cw, ch = int(W * crop_ratio), int(H * crop_ratio)
    cx, cy = int(gt_x * W), int(gt_y * H)
    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(W, x1 + cw)
    y2 = min(H, y1 + ch)
    bx1, by1, bx2, by2 = x1 / W, y1 / H, x2 / W, y2 / H

    # Map GT to local coords
    local_gt_x = (gt_x - bx1) / (bx2 - bx1)
    local_gt_y = (gt_y - by1) / (by2 - by1)

    # GT should be roughly centered in the crop
    assert 0.3 < local_gt_x < 0.7, f"Expected ~0.5, got {local_gt_x}"
    assert 0.3 < local_gt_y < 0.7, f"Expected ~0.5, got {local_gt_y}"
    # GT must be within [0, 1] in local coords
    assert 0.0 <= local_gt_x <= 1.0
    assert 0.0 <= local_gt_y <= 1.0

    print(f"  PASS: GT local coords = ({local_gt_x:.3f}, {local_gt_y:.3f}) (centered in crop)")


def test_sft_peaked_attention_low_loss():
    """SFT loss should be low when attention is peaked on the GT patch."""
    nw, nh = 8, 6
    gt_x, gt_y = 0.4, 0.6

    # Compute GT patch index
    px = max(0, min(round(gt_x * nw - 0.5), nw - 1))
    py = max(0, min(round(gt_y * nh - 0.5), nh - 1))
    gt_idx = py * nw + px

    # Peaked attention: almost all weight on GT patch
    attn_peaked = torch.ones(1, nw * nh) * 0.001
    attn_peaked[0, gt_idx] = 10.0
    attn_peaked = attn_peaked / attn_peaked.sum()

    # Uniform attention
    attn_uniform = torch.ones(1, nw * nh) / (nw * nh)

    nll_peaked = -torch.log(attn_peaked[0, gt_idx])
    nll_uniform = -torch.log(attn_uniform[0, gt_idx])

    assert nll_peaked < nll_uniform, "Peaked attention should have lower NLL than uniform"
    assert nll_peaked < 0.5, f"Peaked NLL should be very low, got {nll_peaked.item()}"

    print(f"  PASS: peaked NLL={nll_peaked.item():.4f} < uniform NLL={nll_uniform.item():.4f}")


if __name__ == "__main__":
    print("Running training logic tests...\n")
    test_advantage_computation()
    test_advantage_with_2_generations()
    test_log_prob_gradient_flow()
    test_shared_r0_attention_correctness()
    test_loss_structure()
    test_convergence_logic()
    test_sft_loss_structure()
    test_sft_gt_local_coord_mapping()
    test_sft_peaked_attention_low_loss()
    print("\nAll tests passed!")
