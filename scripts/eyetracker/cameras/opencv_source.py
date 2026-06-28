"""OpenCV-backed CameraSource.

Capture goes through OpenCV. Exposure does NOT: OpenCV's macOS/AVFoundation
backend can't drive UVC exposure (set() no-ops, get() returns 0), so when a
`uvc_id` is given we route exposure through uvc-util instead — it works
alongside the live capture stream. See UvcExposureController and the
project_macos_uvc_exposure memory.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from scripts.eyetracker.cameras.base import CameraSource
from scripts.eyetracker.cameras.uvc_util import (
    UvcExposureController,
    find_uvc_util,
)


@dataclass
class CameraSettings:
    """Per-camera capture settings, applied at open() time."""
    request_width: Optional[int] = None
    request_height: Optional[int] = None
    request_fps: Optional[int] = None
    flip_vertical: bool = False
    # Manual exposure control (macOS, via uvc-util). uvc_id is the USB
    # "vendor:product" (e.g. "0x0bda:0xd565") used to select the camera;
    # exposure_control names the UVC control to drive as the lever; exposure is
    # its initial value (None leaves the device default). Leave uvc_id None for
    # cameras the app doesn't drive exposure on.
    uvc_id: Optional[str] = None
    exposure_control: str = "exposure-time-abs"
    exposure: Optional[int] = None


class OpenCVCamera(CameraSource):
    def __init__(self, index: int, settings: Optional[CameraSettings] = None):
        self.index = index
        self.settings = settings or CameraSettings()
        self._cap: Optional[cv2.VideoCapture] = None
        # uvc-util exposure controller, set in open() when a uvc_id is given
        # and the binary + device are available.
        self._uvc: Optional[UvcExposureController] = None

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return False
        self._cap = cap
        s = self.settings
        if s.request_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.request_width)
        if s.request_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.request_height)
        if s.request_fps is not None:
            cap.set(cv2.CAP_PROP_FPS, s.request_fps)
        if s.uvc_id is not None:
            self._init_uvc_exposure(s)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    # ---- exposure control (uvc-util) ----------------------------------------

    def _init_uvc_exposure(self, s: CameraSettings) -> None:
        controller = UvcExposureController(s.uvc_id, find_uvc_util(),
                                           control=s.exposure_control)
        if controller.probe():
            controller.apply_initial(s.exposure)
            self._uvc = controller
        else:
            print(f"[exposure] cam {self.index}: no exposure control "
                  f"(uvc-util missing or device {s.uvc_id} not found). Build "
                  "uvc-util and put it on PATH or set UVC_UTIL_PATH.")

    def nudge_exposure(self, direction: int, step_fraction: float) -> bool:
        if self._uvc is None:
            return False
        return self._uvc.nudge_exposure(direction, step_fraction)

    def exposure_status(self) -> Optional[str]:
        return self._uvc.status_str() if self._uvc is not None else None
