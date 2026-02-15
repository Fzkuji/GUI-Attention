"""Shared constants for multi-precision foveated attention."""

# ---------------------------------------------------------------------------
# Multi-precision levels (from coarse to fine)
# ---------------------------------------------------------------------------
PRECISION_LEVELS = [250_000, 1_003_520, 4_000_000, 14_000_000]
LEVEL_NAMES = ["low", "original", "high", "ultra_high"]
STOP_LEVELS = {2, 3}  # attending to these levels terminates the foveation loop


def precision_for_level(level: int) -> int:
    """Return max_pixels for a given precision level (0-based)."""
    return PRECISION_LEVELS[min(level, len(PRECISION_LEVELS) - 1)]


# Backward compat aliases used by old code (to be removed later)
PRECISION_LOW = PRECISION_LEVELS[0]
PRECISION_HIGH = PRECISION_LEVELS[2]


def precision_for_round(round_idx: int) -> int:
    """Deprecated: use precision_for_level instead."""
    return PRECISION_LOW if round_idx == 0 else PRECISION_HIGH


# ---------------------------------------------------------------------------
# Placeholder suffix appended after each round's image+instruction
# ---------------------------------------------------------------------------
PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)
