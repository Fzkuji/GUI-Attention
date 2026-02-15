"""Multi-precision foveation loop.

Progressively zooms into the image by tracking which precision level
the model attends to, and adding higher-resolution crops as needed.

Algorithm:
    1. Start with full image at Level 0 (low precision).
    2. Extract pointer attention over ALL visual tokens (across all images).
    3. Find max-attended token → identify image (level) and spatial location.
    4. If attended level >= STOP threshold → final prediction.
    5. Otherwise, crop around attended location, add at next precision level.
       If the same area was already visited at this level, skip a level.
    6. Repeat until stop or max rounds.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from gui_attention.constants import STOP_LEVELS, PRECISION_LEVELS


@dataclass
class FoveationState:
    """Tracks the state of one foveation trajectory."""
    # Each entry: (global_x, global_y, level) for images added to context
    history: List[Tuple[float, float, int]] = field(default_factory=list)
    # Visited points: rounded coords → highest level seen
    visited: dict = field(default_factory=dict)
    stopped: bool = False
    final_coords: Optional[Tuple[float, float]] = None

    def _round_key(self, x: float, y: float, tolerance: float = 0.05) -> Tuple[int, int]:
        """Quantise coordinates for 'already visited' matching."""
        return (round(x / tolerance), round(y / tolerance))


class FoveationLoop:
    """Manages the progressive zoom decision logic."""

    def __init__(self, max_rounds: int = 5, crop_ratio: float = 0.3,
                 stop_levels: Optional[set] = None,
                 max_level: int = 3):
        self.max_rounds = max_rounds
        self.crop_ratio = crop_ratio
        self.stop_levels = stop_levels or STOP_LEVELS
        self.max_level = max_level

    def new_state(self) -> FoveationState:
        return FoveationState()

    def decide(self, state: FoveationState, attended_level: int,
               global_x: float, global_y: float) -> dict:
        """Given where the model attended, decide the next action.

        Args:
            state: current foveation state.
            attended_level: precision level of the image containing the max-attended token.
            global_x, global_y: attended point in original image normalised coords.

        Returns:
            dict with:
                action: "stop" | "crop"
                level: (for "crop") precision level for the new crop
                coords: (global_x, global_y) of the attended point
        """
        state.history.append((global_x, global_y, attended_level))

        # Stop condition 1: attended to high/ultra-high precision
        if attended_level in self.stop_levels:
            state.stopped = True
            state.final_coords = (global_x, global_y)
            return {"action": "stop", "coords": (global_x, global_y)}

        # Determine next level
        next_level = attended_level + 1

        # Skip logic: if this area was already visited at this level, jump +2
        key = state._round_key(global_x, global_y)
        prev_level = state.visited.get(key)
        if prev_level is not None and prev_level >= attended_level:
            next_level = attended_level + 2

        # Clamp to max level
        next_level = min(next_level, self.max_level)

        # Stop condition 2: next level would exceed max
        if next_level > self.max_level:
            state.stopped = True
            state.final_coords = (global_x, global_y)
            return {"action": "stop", "coords": (global_x, global_y)}

        # Stop condition 3: next level is itself a stop level
        if next_level in self.stop_levels:
            # Still add the crop, but this will be the last round
            pass

        state.visited[key] = next_level
        state.final_coords = (global_x, global_y)

        return {
            "action": "crop",
            "level": next_level,
            "coords": (global_x, global_y),
        }

    def should_continue(self, state: FoveationState, current_round: int) -> bool:
        """Check if we should run another round."""
        if state.stopped:
            return False
        if current_round >= self.max_rounds:
            return False
        return True
