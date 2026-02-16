"""Shared constants for multi-precision foveated attention."""

from gui_aima.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    chat_template as CHAT_TEMPLATE,
    grounding_system_message as GROUNDING_SYSTEM_MESSAGE,
)

# ---------------------------------------------------------------------------
# Multi-precision levels (from coarse to fine)
# ---------------------------------------------------------------------------
PRECISION_LEVELS = [250_000, 1_003_520, 4_000_000, 14_000_000]
LEVEL_NAMES = ["low", "original", "high", "ultra_high"]
STOP_LEVELS = {2, 3}  # attending to these levels terminates the foveation loop


def precision_for_level(level: int) -> int:
    """Return max_pixels for a given precision level (0-based)."""
    return PRECISION_LEVELS[min(level, len(PRECISION_LEVELS) - 1)]


# ---------------------------------------------------------------------------
# Placeholder suffix appended after each round's image+instruction
# ---------------------------------------------------------------------------
PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)
