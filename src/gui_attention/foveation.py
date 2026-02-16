"""Saccade foveation loop.

Simulates human eye: peripheral vision (low-res full image) + foveal vision
(high-res crop at one focus point). The focus point can move (saccade).

Algorithm:
    Round 0: [low-res full image] → action head → attention → select focus
    Round 1+: [low-res full (masked)] + [high-res crop] → action head
              → argmax in high-res → click (done)
              → argmax in low-res → saccade (move focus, next round)
    Stop: argmax in high-res, or max_rounds reached.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SaccadeState:
    """Tracks the state of one saccade trajectory."""
    history: List[Tuple[float, float, str]] = field(default_factory=list)
    stopped: bool = False
    final_coords: Optional[Tuple[float, float]] = None

    def record(self, x: float, y: float, source: str):
        """Record a fixation point. source is 'low' or 'high'."""
        self.history.append((x, y, source))


class SaccadeLoop:
    """Manages the saccade decision logic."""

    def __init__(self, max_rounds: int = 3, crop_ratio: float = 0.3):
        self.max_rounds = max_rounds
        self.crop_ratio = crop_ratio

    def new_state(self) -> SaccadeState:
        return SaccadeState()

    def decide_round0(self, state: SaccadeState, global_x: float, global_y: float) -> dict:
        """After round 0 (low-res only), always crop around the attended point.

        Returns:
            dict with action="crop", coords=(x, y).
        """
        state.record(global_x, global_y, "low")
        state.final_coords = (global_x, global_y)
        return {"action": "crop", "coords": (global_x, global_y)}

    def decide_saccade(self, state: SaccadeState, attended_source: str,
                       global_x: float, global_y: float) -> dict:
        """After round 1+, decide whether to click or saccade.

        Args:
            attended_source: 'high' if argmax was in the high-res crop,
                             'low' if argmax was in the (unmasked) low-res patches.
            global_x, global_y: attended point in original image normalised coords.

        Returns:
            dict with action="stop" (click) or action="saccade" (move focus).
        """
        state.record(global_x, global_y, attended_source)
        state.final_coords = (global_x, global_y)

        if attended_source == "high":
            state.stopped = True
            return {"action": "stop", "coords": (global_x, global_y)}
        else:
            # Saccade: move focus to new location
            return {"action": "saccade", "coords": (global_x, global_y)}

    def should_continue(self, state: SaccadeState, current_round: int) -> bool:
        if state.stopped:
            return False
        if current_round >= self.max_rounds:
            return False
        return True
