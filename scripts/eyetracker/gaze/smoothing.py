"""Trailing moving-average smoothing for predicted gaze points.
"""
from collections import deque
from typing import Tuple

import numpy as np


XY = Tuple[float, float]


class MovingAverageSmoother:
    def __init__(self, window: int):
        """
        Params:
            In __main__.py we use GAZE_BUFFER_SIZE for `window` here
        """
        self._buf: deque = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def add(self, point: XY) -> XY:
        self._buf.append(point)
        avg_u = float(np.mean([p[0] for p in self._buf]))
        avg_v = float(np.mean([p[1] for p in self._buf]))
        return avg_u, avg_v
