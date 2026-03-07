"""Attention-region proposals for visualization and future runtime use.

The core idea is:
1. Treat the attention map as a spatial probability distribution.
2. Partition the map into peak-attraction basins with no hard threshold.
3. Rank basin-local rectangles by attention mass, then density.

This is intentionally deterministic and lightweight so it can be reused in
visualization before being wired into actual inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RectStats:
    center_x: float
    center_y: float
    width: float
    height: float
    var_x: float
    var_y: float


@dataclass(frozen=True)
class BasinProposal:
    basin_id: int
    mass: float
    density: float
    score: float
    rect: RectStats
    peak_yx: tuple[int, int]


def normalize_attention(attn: np.ndarray) -> np.ndarray:
    """Normalize an attention map into a probability distribution."""
    arr = np.asarray(attn, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    total = arr.sum()
    if total <= 0:
        raise ValueError("attention map sum must be positive")
    return arr / total


def _next_higher_neighbor(attn: np.ndarray, y: int, x: int) -> tuple[int, int]:
    """Return the neighboring cell with the highest strictly larger value."""
    h, w = attn.shape
    best_y, best_x = y, x
    best_v = attn[y, x]
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            v = attn[ny, nx]
            if v > best_v:
                best_y, best_x = ny, nx
                best_v = v
    return best_y, best_x


def assign_basins(attn: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Partition the attention map into peak-attraction basins.

    Each cell repeatedly follows its highest-valued neighboring cell until it
    reaches a local maximum. Cells ending at the same peak share a basin label.
    """
    p = normalize_attention(attn)
    h, w = p.shape
    labels = -np.ones((h, w), dtype=np.int32)
    peak_cache: dict[tuple[int, int], tuple[int, int]] = {}
    peaks: list[tuple[int, int]] = []
    peak_to_id: dict[tuple[int, int], int] = {}

    def trace(y: int, x: int) -> tuple[int, int]:
        path: list[tuple[int, int]] = []
        cur = (y, x)
        while cur not in peak_cache:
            path.append(cur)
            nxt = _next_higher_neighbor(p, *cur)
            if nxt == cur:
                peak_cache[cur] = cur
                break
            cur = nxt
        peak = peak_cache[cur]
        for node in path:
            peak_cache[node] = peak
        return peak

    for y in range(h):
        for x in range(w):
            peak = trace(y, x)
            if peak not in peak_to_id:
                peak_to_id[peak] = len(peaks)
                peaks.append(peak)
            labels[y, x] = peak_to_id[peak]
    return labels, peaks


def compute_basin_proposals(attn: np.ndarray, top_k: int = 3) -> tuple[list[BasinProposal], np.ndarray]:
    """Return top-k basin-local rectangle proposals ranked by importance."""
    p = normalize_attention(attn)
    labels, peaks = assign_basins(p)

    h, w = p.shape
    xs = (np.arange(w) + 0.5) / w
    ys = (np.arange(h) + 0.5) / h
    xx, yy = np.meshgrid(xs, ys)
    min_width = 1.0 / max(w, 1)
    min_height = 1.0 / max(h, 1)

    proposals: list[BasinProposal] = []
    for basin_id, peak in enumerate(peaks):
        mask = labels == basin_id
        basin = np.where(mask, p, 0.0)
        mass = float(basin.sum())
        if mass <= 0:
            continue
        basin = basin / mass

        center_x = float((basin * xx).sum())
        center_y = float((basin * yy).sum())
        var_x = float((basin * (xx - center_x) ** 2).sum())
        var_y = float((basin * (yy - center_y) ** 2).sum())
        width = min(max(math.sqrt(max(0.0, 12.0 * var_x)), min_width), 1.0)
        height = min(max(math.sqrt(max(0.0, 12.0 * var_y)), min_height), 1.0)
        area = max(width * height, min_width * min_height)
        density = mass / area
        proposals.append(
            BasinProposal(
                basin_id=basin_id,
                mass=mass,
                density=float(density),
                score=mass,
                rect=RectStats(
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    height=height,
                    var_x=var_x,
                    var_y=var_y,
                ),
                peak_yx=peak,
            )
        )

    proposals.sort(key=lambda item: (item.score, item.density), reverse=True)
    return proposals[:top_k], labels


def rect_box(center_x: float, center_y: float, width: float, height: float) -> tuple[float, float, float, float]:
    """Convert center/size rectangle stats into an in-bounds normalized box."""
    half_w = width / 2.0
    half_h = height / 2.0
    x1 = max(0.0, center_x - half_w)
    y1 = max(0.0, center_y - half_h)
    x2 = min(1.0, center_x + half_w)
    y2 = min(1.0, center_y + half_h)

    cur_w = x2 - x1
    cur_h = y2 - y1
    if cur_w < width:
        if x1 <= 0.0:
            x2 = min(1.0, width)
        elif x2 >= 1.0:
            x1 = max(0.0, 1.0 - width)
    if cur_h < height:
        if y1 <= 0.0:
            y2 = min(1.0, height)
        elif y2 >= 1.0:
            y1 = max(0.0, 1.0 - height)
    return x1, y1, x2, y2
