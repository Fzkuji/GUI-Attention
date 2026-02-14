"""Shared constants for training and evaluation."""

PRECISION_LOW = 1_003_520   # ~56x56 x 320 patches
PRECISION_HIGH = 5_760_000  # ~56x56 x ~1836 patches

PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)


def precision_for_round(round_idx: int) -> int:
    """Return max_pixels for a given 0-based round index.
    Round 0 = low res (periphery), Round 1+ = high res (fovea)."""
    return PRECISION_LOW if round_idx == 0 else PRECISION_HIGH
