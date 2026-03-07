"""Inference-time runtime utilities for GUI-Attention."""

from gui_attention.runtime.foveation import SaccadeLoop
from gui_attention.runtime.inference import run_saccade_inference
from gui_attention.runtime.reasoning import (
    ActionSpanStoppingCriteria,
    CLICK_ACTION_SPAN,
    LOOK_ACTION_SPAN,
    ParsedReasoningAction,
    decode_reasoning_content,
    get_reasoning_guide,
    parse_reasoning_action,
    sample_sft_reasoning_content,
    wrap_assistant_content,
)

__all__ = [
    "SaccadeLoop",
    "run_saccade_inference",
    "ActionSpanStoppingCriteria",
    "CLICK_ACTION_SPAN",
    "LOOK_ACTION_SPAN",
    "ParsedReasoningAction",
    "decode_reasoning_content",
    "get_reasoning_guide",
    "parse_reasoning_action",
    "sample_sft_reasoning_content",
    "wrap_assistant_content",
]
