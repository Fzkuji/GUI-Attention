"""Shared helpers for free-form reasoning with structured action spans."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import StoppingCriteria

from gui_attention.constants import (
    DEFAULT_CLICK_PAD_TOKEN,
    DEFAULT_LOOK_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
)

LOOK_ACTION_SPAN = (
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_LOOK_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
)
CLICK_ACTION_SPAN = (
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_CLICK_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
)

REASONING_ASSISTANT_PREFIX = "<|im_start|>assistant\n"

ROUND0_REASONING_GUIDE = (
    "\nRespond as the assistant with a brief explanation, then end the response with exactly one action span:\n"
    f"{LOOK_ACTION_SPAN}\n"
    "This is the initial low-resolution overview, so you must inspect a high-resolution region next."
)

CROP_REASONING_GUIDE = (
    "\nRespond as the assistant with a brief explanation, then end the response with exactly one action span:\n"
    f"{LOOK_ACTION_SPAN} or {CLICK_ACTION_SPAN}\n"
    "Use the look action when you need another close-up. Use the click action only when the target is clear enough to click now."
)


@dataclass
class ParsedReasoningAction:
    raw_content: str
    used_content: str
    thought: str
    action: str
    format_ok: bool
    parse_failed: bool


def get_reasoning_guide(*, allow_click: bool) -> str:
    return CROP_REASONING_GUIDE if allow_click else ROUND0_REASONING_GUIDE


def decode_reasoning_content(tokenizer, generated_token_ids: Sequence[int]) -> str:
    decoded = tokenizer.decode(
        list(generated_token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded = decoded.replace("<|im_end|>", "").strip()
    return decoded.strip()


def normalize_assistant_content(content: str) -> str:
    text = content.strip()
    if text.startswith("<|im_start|>assistant"):
        text = text[len("<|im_start|>assistant"):].lstrip("\n")
    text = text.replace("<|im_end|>", "").strip()
    return text


def wrap_assistant_content(content: str) -> str:
    text = normalize_assistant_content(content)
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def canonical_assistant_content(thought: str, action: str) -> str:
    thought = " ".join((thought or "").split()).strip()
    if not thought:
        thought = (
            "I should inspect another region more carefully."
            if action == "look"
            else "I have enough evidence to click the target now."
        )
    action_span = LOOK_ACTION_SPAN if action == "look" else CLICK_ACTION_SPAN
    return f"{thought}{action_span}"


def _find_action_span(text: str):
    spans = []
    for action, span in (("look", LOOK_ACTION_SPAN), ("click", CLICK_ACTION_SPAN)):
        start = 0
        while True:
            idx = text.find(span, start)
            if idx < 0:
                break
            spans.append((idx, action, span))
            start = idx + len(span)
    spans.sort(key=lambda x: x[0])
    return spans


def parse_reasoning_action(content: str, *, allow_click: bool) -> ParsedReasoningAction:
    text = normalize_assistant_content(content)
    spans = _find_action_span(text)
    total_hits = len(spans)
    parse_failed = total_hits != 1
    action = "look"
    action_span = LOOK_ACTION_SPAN
    action_pos = -1
    if total_hits == 1:
        action_pos, action, action_span = spans[0]

    thought_block = text[:action_pos] if action_pos >= 0 else text
    thought_block = thought_block.rstrip()
    thought = thought_block.strip()

    format_ok = (
        total_hits == 1
        and text.rstrip().endswith(action_span)
        and bool(thought)
    )

    if action == "click" and not allow_click:
        parse_failed = True
        format_ok = False
        action = "look"
        action_span = LOOK_ACTION_SPAN

    if total_hits != 1:
        action = "look"

    used_content = text if total_hits == 1 and not parse_failed else canonical_assistant_content(thought, action)
    return ParsedReasoningAction(
        raw_content=text,
        used_content=used_content,
        thought=thought,
        action=action,
        format_ok=format_ok and not parse_failed,
        parse_failed=parse_failed or not bool(thought),
    )


class ActionSpanStoppingCriteria(StoppingCriteria):
    """Stop after the model emits a full action span."""

    def __init__(
        self,
        *,
        prompt_len: int,
        pointer_end_token_id: int,
        allowed_action_token_ids: Iterable[int],
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.pointer_end_token_id = pointer_end_token_id
        self.allowed_action_token_ids = {
            int(t) for t in allowed_action_token_ids if t is not None
        }

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[0] != 1:
            return False
        generated = input_ids[0, self.prompt_len:].tolist()
        if not generated:
            return False
        if generated[-1] != self.pointer_end_token_id:
            return False
        return any(t in self.allowed_action_token_ids for t in generated)
