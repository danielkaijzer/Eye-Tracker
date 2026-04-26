"""Pupil Labs `Detector2D` + `pye3d.Detector3D` adapter."""
import time
from typing import Optional, Tuple

import numpy as np

from scripts.eyetracker.pupil.base import PupilDetector, PupilSample


class PupilLabsDetector(PupilDetector):
    """Lazy-init wrapper around `pupil_detectors.Detector2D` and
    `pye3d.detector_3d.Detector3D`. The first ~30-60s build the 3D eye-sphere
    model; after that the 2D ellipse is refined by the model's constraint and
    becomes robust to blinks/eyelid occlusion."""

    def __init__(self, focal_length_px: float,
                 resolution: Tuple[int, int] = (640, 480),
                 long_term_blocking: bool = True):
        self.focal_length_px = focal_length_px
        self.resolution = resolution
        self.long_term_blocking = long_term_blocking
        self._det2d = None
        self._det3d = None
        self._camera = None

    def _ensure_initialized(self):
        if self._det3d is not None:
            return
        from pupil_detectors import Detector2D
        from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
        self._camera = CameraModel(focal_length=self.focal_length_px,
                                   resolution=self.resolution)
        self._det2d = Detector2D()
        mode = DetectorMode.blocking if self.long_term_blocking else DetectorMode.async_
        self._det3d = Detector3D(camera=self._camera, long_term_mode=mode)

    def detect(self, gray_frame: np.ndarray) -> Optional[PupilSample]:
        self._ensure_initialized()
        result_2d = self._det2d.detect(gray_frame)
        result_2d["timestamp"] = time.perf_counter()
        result_3d = self._det3d.update_and_detect(result_2d, gray_frame)
        conf = float(result_3d["confidence"])
        if conf <= 0.0:
            return None
        ell = result_3d["ellipse"]
        cx, cy = ell["center"]
        center_int = (int(round(cx)), int(round(cy)))
        ellipse_tuple = ((cx, cy), ell["axes"], ell["angle"])
        return PupilSample(center=center_int, ellipse=ellipse_tuple, confidence=conf)

    def reset(self) -> None:
        """Discard the 3D eye-sphere model so the next detect() rebuilds it.
        Use after re-seating the headset."""
        self._det3d = None
        self._camera = None
