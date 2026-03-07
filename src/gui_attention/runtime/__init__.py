"""Runtime utilities for GUI-Attention.

Keep package imports lazy to avoid circular dependencies such as:

  inputs.builder -> runtime.reasoning -> runtime.__init__ -> runtime.inference -> inputs.builder
"""

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


def __getattr__(name):
    if name == "SaccadeLoop":
        from gui_attention.runtime.foveation import SaccadeLoop

        return SaccadeLoop
    if name == "run_saccade_inference":
        from gui_attention.runtime.inference import run_saccade_inference

        return run_saccade_inference
    if name in {
        "ActionSpanStoppingCriteria",
        "CLICK_ACTION_SPAN",
        "LOOK_ACTION_SPAN",
        "ParsedReasoningAction",
        "decode_reasoning_content",
        "get_reasoning_guide",
        "parse_reasoning_action",
        "sample_sft_reasoning_content",
        "wrap_assistant_content",
    }:
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

        return {
            "ActionSpanStoppingCriteria": ActionSpanStoppingCriteria,
            "CLICK_ACTION_SPAN": CLICK_ACTION_SPAN,
            "LOOK_ACTION_SPAN": LOOK_ACTION_SPAN,
            "ParsedReasoningAction": ParsedReasoningAction,
            "decode_reasoning_content": decode_reasoning_content,
            "get_reasoning_guide": get_reasoning_guide,
            "parse_reasoning_action": parse_reasoning_action,
            "sample_sft_reasoning_content": sample_sft_reasoning_content,
            "wrap_assistant_content": wrap_assistant_content,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
