"""OpenCV-backed CameraSource.

Capture goes through OpenCV. Exposure does NOT: OpenCV's macOS/AVFoundation
backend can't drive UVC exposure (set() no-ops, get() returns 0), so when a
`uvc_id` is given we route exposure through uvc-util instead — it works
alongside the live capture stream. See UvcExposureController and the
project_macos_uvc_exposure memory.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

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
    # Exposure control (macOS, via uvc-util). uvc_id is the USB "vendor:product"
    # (e.g. "0x0c45:0x6366") used to select the camera. auto_exposure sets the
    # initial mode (False = manual, the AEC-off state); gain is the manual
    # brightness lever (None leaves the device default).
    uvc_id: Optional[str] = None
    auto_exposure: Optional[bool] = None
    gain: Optional[int] = None


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
        controller = UvcExposureController(s.uvc_id, find_uvc_util())
        if controller.probe():
            controller.apply_initial(s.auto_exposure, s.gain)
            self._uvc = controller
        else:
            print(f"[exposure] cam {self.index}: no exposure control "
                  f"(uvc-util missing or device {s.uvc_id} not found). Build "
                  "uvc-util and put it on PATH or set UVC_UTIL_PATH.")

    def set_auto_exposure(self, auto: bool) -> bool:
        return self._uvc.set_auto(auto) if self._uvc is not None else False

    def set_exposure(self, exposure: float) -> bool:
        # The wired manual lever on these modules is gain (exposure-time is a
        # cosmetic no-op), so the manual "exposure" value maps to gain.
        return self._uvc.set_gain(int(exposure)) if self._uvc is not None else False

    def get_exposure_settings(self) -> Tuple[Optional[bool], Optional[float]]:
        if self._uvc is None:
            return (None, None)
        auto, gain = self._uvc.state()
        return (auto, float(gain) if gain is not None else None)

    def exposure_status(self) -> Optional[str]:
        return self._uvc.status_str() if self._uvc is not None else None
