"""Target-pattern strategies for the calibration routine.

The routine asks the pattern for a list of (x_screen, y_screen) targets given
the active screen size. Add a new pattern by subclassing TargetPattern.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


XY = Tuple[int, int]


class TargetPattern(ABC):
    @abstractmethod
    def generate(self, screen_width: int, screen_height: int) -> List[XY]:
        """Return on-screen target positions in display order."""


class GridPattern(TargetPattern):
    """Row-major grid of targets, evenly spaced inside a margin from the edges.
    Default 4 cols × 3 rows = 12 targets. Margin is per-edge in pixels."""

    def __init__(self, rows: int = 3, cols: int = 4, margin: int = 220):
        self.rows = rows
        self.cols = cols
        self.margin = margin

    def generate(self, screen_width: int, screen_height: int) -> List[XY]:
        pts: List[XY] = []
        usable_w = screen_width - 2 * self.margin
        usable_h = screen_height - 2 * self.margin
        for r in range(self.rows):
            for c in range(self.cols):
                x = int(self.margin + c * usable_w / (self.cols - 1))
                y = int(self.margin + r * usable_h / (self.rows - 1))
                pts.append((x, y))
        return pts
