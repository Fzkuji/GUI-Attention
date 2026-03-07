"""GUI-Attention: Saccade Foveation for Efficient GUI Grounding."""

__version__ = "0.3.0"

__all__ = [
    "MultiRoundInputBuilder",
    "Qwen25VLWithDualHead",
    "build_model",
    "run_saccade_inference",
]


def __getattr__(name):
    if name == "MultiRoundInputBuilder":
        from gui_attention.inputs import MultiRoundInputBuilder

        return MultiRoundInputBuilder
    if name in {"Qwen25VLWithDualHead", "build_model"}:
        from gui_attention.modeling import Qwen25VLWithDualHead, build_model

        return {
            "Qwen25VLWithDualHead": Qwen25VLWithDualHead,
            "build_model": build_model,
        }[name]
    if name == "run_saccade_inference":
        from gui_attention.runtime import run_saccade_inference

        return run_saccade_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
