"""Trailing moving-average smoothing for predicted gaze points.

Was the `screen_buffer` deque in the legacy module; the buffer name is kept
as `BUFFER_SIZE` in config for continuity even though it now smooths in
scene-cam coords, not screen coords.
"""
from collections import deque
from typing import Tuple

import numpy as np


XY = Tuple[float, float]


class MovingAverageSmoother:
    def __init__(self, window: int):
        self._buf: deque = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def add(self, point: XY) -> XY:
        self._buf.append(point)
        avg_u = float(np.mean([p[0] for p in self._buf]))
        avg_v = float(np.mean([p[1] for p in self._buf]))
        return avg_u, avg_v
