"""OpenCV-backed CameraSource.

Capture goes through OpenCV. Exposure does NOT: OpenCV's macOS/AVFoundation
backend can't drive UVC exposure (set() no-ops, get() returns 0), so when a
`uvc_id` is given we route exposure through uvc-util instead — it works
alongside the live capture stream. See UvcExposureController and the
project_macos_uvc_exposure memory. On Linux, capture uses the V4L2 backend with
MJPG (uncompressed YUYV can't carry 1080p at usable fps over USB 2), and
exposure goes through v4l2-ctl (cameras/v4l2.py).

Frames are pulled by a background thread that keeps only the newest one.
Without it, a camera faster than the App loop (eye cam ~100 fps vs a ~30 Hz
loop) fills the V4L2 buffer queue and read() hands back the OLDEST queued
frame — measured ~130-275 ms stale on the Jetson. AVFoundation drops stale
frames itself, which is why this never showed on macOS.

Each frame carries a capture timestamp (CameraSource.last_timestamp). On Linux
it's the V4L2 buffer timestamp (CLOCK_MONOTONIC), which is the camera's own
hardware timestamp when uvcvideo hwtimestamps=1 (see v4l2.py); elsewhere it's
time.monotonic() when the grabber received the frame.
"""
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from scripts.eyetracker.cameras.base import CameraSource
from scripts.eyetracker.cameras.uvc_util import (
    UvcExposureController,
    find_uvc_util,
)
from scripts.eyetracker.cameras.v4l2 import (
    V4l2ExposureController,
    find_v4l2_ctl,
    uvc_hw_timestamps_enabled,
)

IS_LINUX = sys.platform.startswith("linux")


@dataclass
class CameraSettings:
    """Per-camera capture settings, applied at open() time."""
    request_width: Optional[int] = None
    request_height: Optional[int] = None
    request_fps: Optional[int] = None
    flip_vertical: bool = False
    # Manual exposure control (uvc-util on macOS, v4l2-ctl on Linux). uvc_id is the USB
    # "vendor:product" (e.g. "0x0bda:0xd565") used to select the camera;
    # exposure_control names the UVC control to drive as the lever; exposure is
    # its initial value (None leaves the device default). Leave uvc_id None for
    # cameras the app doesn't drive exposure on. On Linux the device is
    # addressed by its /dev/video index instead, so only exposure is used.
    uvc_id: Optional[str] = None
    exposure_control: str = "exposure-time-abs"
    exposure: Optional[int] = None


class OpenCVCamera(CameraSource):
    def __init__(self, index: int, settings: Optional[CameraSettings] = None):
        self.index = index
        self.settings = settings or CameraSettings()
        self._cap: Optional[cv2.VideoCapture] = None
        # Latest-frame handoff from the grabber thread. _seq bumps per grab
        # (successful or not) so read() never returns the same frame twice.
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: Optional[float] = None
        self._seq = 0
        self._read_seq = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Exposure controller (uvc-util / v4l2-ctl), set in open() when a
        # uvc_id is given and the binary + device are available.
        self._uvc = None

    def open(self) -> bool:
        if IS_LINUX:
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return False
        self._cap = cap
        s = self.settings
        if IS_LINUX:
            # Must precede the size request: V4L2 negotiates size per format.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
        if IS_LINUX and not uvc_hw_timestamps_enabled():
            print(f"[timestamps] cam {self.index}: uvcvideo hwtimestamps is off — "
                  "frame times are host arrival (~2 ms jitter), not the camera "
                  "clock. See cameras/v4l2.py:uvc_hw_timestamps_enabled.")
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True,
                                        name=f"cam{self.index}-grab")
        self._thread.start()
        return True

    def _grab_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                ts = None
            elif IS_LINUX:
                # V4L2 buffer timestamp (ms, CLOCK_MONOTONIC) of this frame.
                # Read on this thread, right after the grab it belongs to.
                ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                ts = time.monotonic()
            with self._cond:
                self._frame = frame if ret else None
                self._frame_ts = ts
                self._seq += 1
                self._cond.notify_all()
            if not ret:
                time.sleep(0.01)  # unplugged/dead cam: don't spin a core

    def read(self, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        """Newest frame not yet returned; blocks until one arrives. None on
        a failed grab or if nothing arrives within timeout_s."""
        if self._cap is None:
            return None
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._read_seq,
                                       timeout=timeout_s):
                return None
            self._read_seq = self._seq
            frame = self._frame
            self.last_timestamp = self._frame_ts
        if frame is None:
            return None
        if self.settings.flip_vertical:
            frame = cv2.flip(frame, 0)
        return frame

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---- exposure control (uvc-util / v4l2-ctl) -----------------------------

    def _init_uvc_exposure(self, s: CameraSettings) -> None:
        if IS_LINUX:
            controller = V4l2ExposureController(self.index, find_v4l2_ctl())
            hint = ("v4l2-ctl missing or no exposure control on "
                    f"/dev/video{self.index}. Install v4l-utils.")
        else:
            controller = UvcExposureController(s.uvc_id, find_uvc_util(),
                                               control=s.exposure_control)
            hint = (f"uvc-util missing or device {s.uvc_id} not found. Build "
                    "uvc-util and put it on PATH or set UVC_UTIL_PATH.")
        if controller.probe():
            controller.apply_initial(s.exposure)
            self._uvc = controller
        else:
            print(f"[exposure] cam {self.index}: no exposure control ({hint})")

    def nudge_exposure(self, direction: int, step_fraction: float) -> bool:
        if self._uvc is None:
            return False
        return self._uvc.nudge_exposure(direction, step_fraction)

    def exposure_status(self) -> Optional[str]:
        return self._uvc.status_str() if self._uvc is not None else None
