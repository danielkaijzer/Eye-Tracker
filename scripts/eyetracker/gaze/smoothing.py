"""1€ filter (Casiez et al., 2012) for predicted gaze points.

Adaptive low-pass: cutoff rises with signal speed, so fixations get heavy
smoothing and saccades pass through with little lag. Tuning: lower
`min_cutoff` for steadier fixations; raise `beta` to track fast motion
more aggressively.
"""
import math
import time
from typing import Optional, Tuple


XY = Tuple[float, float]


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _Channel:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0

    def filter(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / dt
        a_d = _alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class OneEuroSmoother:
    def __init__(self,
                 min_cutoff: float = 1.0,
                 beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self._u = _Channel(min_cutoff, beta, d_cutoff)
        self._v = _Channel(min_cutoff, beta, d_cutoff)
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._u.reset()
        self._v.reset()
        self._t_prev = None

    def add(self, point: XY) -> XY:
        t = time.monotonic()
        if self._t_prev is None:
            dt = 1.0 / 30.0
        else:
            dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t
        return self._u.filter(point[0], dt), self._v.filter(point[1], dt)
