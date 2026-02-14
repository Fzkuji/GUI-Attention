"""Attention sampling and prediction functions."""

import torch
import torch.nn.functional as F

from gui_aima.inference import get_prediction_region_point


def sample_from_attention(attn_weights, n_w, n_h, temperature=1.0):
    """Sample a coordinate from the attention distribution.
    Returns (x_norm, y_norm, log_prob, flat_index)."""
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


def compute_round_log_prob(attn_weights, local_coords, nw, nh, temperature=1.0):
    """Compute log probability of a position under the attention distribution."""
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


def argmax_from_attention(attn_weights, n_w, n_h):
    """Deterministic prediction: argmax of attention distribution."""
    p = attn_weights.squeeze(0).float()
    idx = p.argmax().item()
    px = idx % n_w
    py = idx // n_w
    return (px + 0.5) / n_w, (py + 0.5) / n_h


def region_from_attention(attn_weights, n_w, n_h):
    """Deterministic prediction using GUI-AIMA's region-based method.
    More robust than argmax: finds densest attention cluster via BFS."""
    best_point, _, _, _ = get_prediction_region_point(
        attn_weights, n_w, n_h, return_all_regions=True,
    )
    return best_point
