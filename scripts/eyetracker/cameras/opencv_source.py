"""OpenCV-backed CameraSource."""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from scripts.eyetracker.cameras.base import CameraSource


@dataclass
class CameraSettings:
    """Per-camera capture settings, applied at open() time."""
    request_width: Optional[int] = None
    request_height: Optional[int] = None
    request_fps: Optional[int] = None
    exposure: Optional[float] = None
    flip_vertical: bool = False


class OpenCVCamera(CameraSource):
    def __init__(self, index: int, settings: Optional[CameraSettings] = None):
        self.index = index
        self.settings = settings or CameraSettings()
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return False
        s = self.settings
        if s.request_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.request_width)
        if s.request_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.request_height)
        if s.request_fps is not None:
            cap.set(cv2.CAP_PROP_FPS, s.request_fps)
        if s.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, s.exposure)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        if self.settings.flip_vertical:
            frame = cv2.flip(frame, 0)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
