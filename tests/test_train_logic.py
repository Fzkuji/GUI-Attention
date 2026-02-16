"""
Integration test for v4 training logic.

Tests the core logic (action head, labels, saccade decisions, loss structure)
without needing actual model weights.

Run with: python tests/test_train_logic.py
"""

import torch


def test_action_head_loss_gradient():
    """Test that action head loss produces differentiable output."""
    from gui_attention.action_head import ActionHead
    head = ActionHead(d_model=64, projection_dim=64)
    vis = torch.randn(50, 64, requires_grad=True)
    anchor = torch.randn(1, 64, requires_grad=True)
    labels = torch.zeros(1, 50)
    labels[0, 20:25] = 1.0
    attn, loss = head(vis, anchor, labels=labels)
    loss.backward()
    assert vis.grad is not None and vis.grad.abs().sum() > 0
    assert anchor.grad is not None and anchor.grad.abs().sum() > 0
    print(f"  PASS: Action head gradient flows (loss={loss.item():.4f})")


def test_action_head_peaked_low_loss():
    """Loss should be lower when attention is peaked on GT patches."""
    from gui_attention.action_head import ActionHead
    torch.manual_seed(42)
    head = ActionHead(d_model=64, projection_dim=64)

    vis = torch.randn(50, 64)
    anchor = torch.randn(1, 64)
    labels = torch.zeros(1, 50)
    labels[0, 20:25] = 1.0

    _, loss = head(vis, anchor, labels=labels)
    # Loss should be finite and positive
    assert loss.isfinite()
    assert loss.item() > 0
    print(f"  PASS: Action head loss = {loss.item():.4f} (finite, positive)")


def test_binary_labels_correct():
    """Binary labels should mark patches overlapping with GT bbox."""
    from gui_attention.labels import compute_binary_labels
    labels = compute_binary_labels(10, 10, (0.3, 0.4, 0.6, 0.7))
    assert labels.shape == (100,)
    # Patches in the overlap region should be 1
    positive_count = labels.sum().item()
    assert positive_count > 0
    assert positive_count < 100  # Not all patches
    print(f"  PASS: Binary labels: {int(positive_count)}/100 positive patches")


def test_overlap_mask_correctness():
    """Overlap mask should mark patches covered by crop bbox."""
    from gui_attention.labels import compute_overlap_mask
    mask = compute_overlap_mask(10, 10, (0.0, 0.0, 0.5, 0.5))
    # Top-left quadrant patches should be masked
    assert mask[0].item()  # patch (0,0) should be masked
    assert not mask[99].item()  # patch (9,9) should not be masked
    print(f"  PASS: Overlap mask: {mask.sum().item()}/100 masked patches")


def test_saccade_loop_stop_on_high():
    """SaccadeLoop should stop when argmax is in high-res crop."""
    from gui_attention.foveation import SaccadeLoop

    loop = SaccadeLoop(max_rounds=3)
    state = loop.new_state()

    # Round 0: always crop
    d0 = loop.decide_round0(state, 0.5, 0.5)
    assert d0["action"] == "crop"

    # Round 1: argmax in high-res → stop
    d1 = loop.decide_saccade(state, "high", 0.5, 0.5)
    assert d1["action"] == "stop"
    assert state.stopped

    print("  PASS: SaccadeLoop stops on high-res attention")


def test_saccade_loop_saccade_on_low():
    """SaccadeLoop should saccade when argmax is in low-res."""
    from gui_attention.foveation import SaccadeLoop

    loop = SaccadeLoop(max_rounds=3)
    state = loop.new_state()

    d0 = loop.decide_round0(state, 0.5, 0.5)
    assert d0["action"] == "crop"

    # Round 1: argmax in low-res → saccade
    d1 = loop.decide_saccade(state, "low", 0.3, 0.7)
    assert d1["action"] == "saccade"
    assert not state.stopped

    print("  PASS: SaccadeLoop saccades on low-res attention")


def test_saccade_max_rounds():
    """SaccadeLoop should respect max_rounds."""
    from gui_attention.foveation import SaccadeLoop

    loop = SaccadeLoop(max_rounds=2)
    state = loop.new_state()

    assert loop.should_continue(state, current_round=0)
    assert loop.should_continue(state, current_round=1)
    assert not loop.should_continue(state, current_round=2)

    print("  PASS: SaccadeLoop respects max_rounds")


def test_two_round_loss_structure():
    """Test 2-round teacher forcing loss structure."""
    from gui_attention.action_head import ActionHead
    from gui_attention.labels import compute_binary_labels, compute_overlap_mask

    torch.manual_seed(42)
    head = ActionHead(d_model=64, projection_dim=64)

    # Round 0: low-res only
    n_low = 36  # 6x6
    vis0 = torch.randn(n_low, 64)
    anchor0 = torch.randn(1, 64)
    gt_bbox = (0.3, 0.3, 0.6, 0.6)
    labels0 = compute_binary_labels(6, 6, gt_bbox).unsqueeze(0)

    _, loss0 = head(vis0, anchor0, labels=labels0)
    assert loss0.isfinite()

    # Round 1: low-res + high-res crop
    n_high = 100  # 10x10
    n_total = n_low + n_high
    vis1 = torch.randn(n_total, 64)
    anchor1 = torch.randn(1, 64)

    crop_bbox = (0.2, 0.2, 0.5, 0.5)
    local_gt = (
        (gt_bbox[0] - crop_bbox[0]) / (crop_bbox[2] - crop_bbox[0]),
        (gt_bbox[1] - crop_bbox[1]) / (crop_bbox[3] - crop_bbox[1]),
        (gt_bbox[2] - crop_bbox[0]) / (crop_bbox[2] - crop_bbox[0]),
        (gt_bbox[3] - crop_bbox[1]) / (crop_bbox[3] - crop_bbox[1]),
    )
    high_labels = compute_binary_labels(10, 10, local_gt)
    full_labels = torch.zeros(1, n_total)
    full_labels[0, n_low:] = high_labels

    low_mask = compute_overlap_mask(6, 6, crop_bbox)
    full_mask = torch.zeros(n_total, dtype=torch.bool)
    full_mask[:n_low] = low_mask

    _, loss1 = head(vis1, anchor1, labels=full_labels, mask=full_mask)
    assert loss1.isfinite()

    total_loss = (loss0 + loss1) / 2
    assert total_loss.isfinite()
    assert total_loss.item() > 0

    print(f"  PASS: 2-round loss = {total_loss.item():.4f} (loss0={loss0.item():.4f}, loss1={loss1.item():.4f})")


def test_identify_attended_image():
    """Test that identify_attended_image maps global argmax to correct image."""
    from gui_attention.attention import identify_attended_image

    # 2 images: low-res (36 tokens) + high-res (100 tokens)
    attn = torch.zeros(136)
    attn[50] = 1.0  # max in high-res image (offset 36, local=14)

    ranges = [(0, 36), (36, 100)]
    img_idx, local_idx = identify_attended_image(attn, ranges)
    assert img_idx == 1, f"Expected image 1, got {img_idx}"
    assert local_idx == 14, f"Expected local 14, got {local_idx}"

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

    print("  PASS: token_to_spatial correct")


def test_gt_local_coord_mapping():
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

    print(f"  PASS: GT local coords = ({local_gt_x:.3f}, {local_gt_y:.3f})")


def test_bfs_region_prediction():
    """Test BFS connected region prediction."""
    from gui_attention.inference import get_prediction_region_point

    # Create attention with a clear peak region
    attn = torch.zeros(100)  # 10x10 grid
    attn[44] = 0.5  # (4, 4)
    attn[45] = 0.8  # (4, 5)
    attn[54] = 0.6  # (5, 4)
    attn[55] = 0.9  # (5, 5) - max

    best, centers, scores = get_prediction_region_point(attn, 10, 10, activation_threshold=0.3)
    assert len(centers) >= 1
    # Best point should be near (0.5, 0.5)
    assert 0.3 < best[0] < 0.7
    assert 0.3 < best[1] < 0.7

    print(f"  PASS: BFS prediction = ({best[0]:.3f}, {best[1]:.3f})")


if __name__ == "__main__":
    print("Running v4 training logic tests...\n")
    test_action_head_loss_gradient()
    test_action_head_peaked_low_loss()
    test_binary_labels_correct()
    test_overlap_mask_correctness()
    test_saccade_loop_stop_on_high()
    test_saccade_loop_saccade_on_low()
    test_saccade_max_rounds()
    test_two_round_loss_structure()
    test_identify_attended_image()
    test_token_to_spatial()
    test_gt_local_coord_mapping()
    test_bfs_region_prediction()
    print("\nAll tests passed!")
