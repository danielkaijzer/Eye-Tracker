"""Per-frame pupil-acceptance gates.

Two gates run in series in the App loop:
1. ConfidenceGate — threshold the detector's confidence score.
2. JumpGate — reject frames whose center jumps too far from a running median
   (catches catastrophic ellipse-fit flips; tuned to allow saccades).
"""
from collections import deque
from typing import Optional, Tuple

import numpy as np


class ConfidenceGate:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def accept(self, confidence: float) -> bool:
        return confidence >= self.threshold

    def describe_reject(self, confidence: float) -> str:
        return f"conf={confidence:.2f} < CONF_THRESH={self.threshold}"


class JumpGate:
    """Reject pupil centers that jump too far from the running median of the
    last `buffer_size` accepted centers. The first frame is always accepted."""

    def __init__(self, threshold_px: float, buffer_size: int):
        self.threshold_px = threshold_px
        self._buffer: deque = deque(maxlen=buffer_size)

    def reset(self) -> None:
        self._buffer.clear()

    def accept(self, center_xy: Tuple[float, float]) -> Optional[np.ndarray]:
        """Return the accepted center as an ndarray, or None if rejected."""
        raw = np.array(center_xy, dtype=float)
        if len(self._buffer) == 0:
            self._buffer.append(raw)
            return raw
        median = np.median(np.array(self._buffer), axis=0)
        if np.linalg.norm(raw - median) > self.threshold_px:
            return None
        self._buffer.append(raw)
        return raw

    def jump_px(self, center_xy: Tuple[float, float]) -> float:
        """Distance from the current center to the running median; NaN if buffer empty."""
        if len(self._buffer) == 0:
            return float("nan")
        median = np.median(np.array(self._buffer), axis=0)
        return float(np.linalg.norm(np.array(center_xy, dtype=float) - median))

    def describe_reject(self, center_xy: Tuple[float, float], confidence: float) -> str:
        return (f"jump={self.jump_px(center_xy):.0f}px > PUPIL_JUMP_THRESH="
                f"{self.threshold_px:.0f} (conf={confidence:.2f})")
