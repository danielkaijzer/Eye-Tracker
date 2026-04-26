"""TargetMapper interface — screen <-> scene-cam pixel transform.

Used at calibration time to label each on-screen target with its scene-cam
pixel coordinates, and (eventually) at inference time to project the predicted
scene-cam gaze point back to a screen pixel.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


XY = Tuple[float, float]


class TargetMapper(ABC):
    @abstractmethod
    def screen_to_scene(self, xy_screen: XY, scene_frame: np.ndarray) -> Optional[XY]:
        """Project a screen-pixel coordinate into scene-cam pixels using the
        current scene frame as reference. Returns None if the mapping cannot
        be established (e.g. fiducials not visible)."""

    @abstractmethod
    def scene_to_screen(self, xy_scene: XY, scene_frame: np.ndarray) -> Optional[XY]:
        """Inverse of screen_to_scene. Returns None on failure."""

    @abstractmethod
    def is_ready(self, scene_frame: np.ndarray) -> bool:
        """True if the mapping can be computed for this frame."""
