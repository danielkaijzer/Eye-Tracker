"""PupilDetector interface — pluggable pupil-localization stage.

Implementations include the Pupil Labs `Detector2D` + `pye3d.Detector3D` stack
(see `pupil_labs.py`) and could include a hand-rolled detector or any other
ellipse-fitter implementing the same contract.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PupilSample:
    """One pupil-detection result.

    center is rounded ints suitable for cv2 drawing; ellipse is the
    ((cx, cy), (axes_minor, axes_major), angle_deg) tuple cv2.ellipse expects.
    """
    center: Tuple[int, int]
    ellipse: Tuple
    confidence: float


class PupilDetector(ABC):
    @abstractmethod
    def detect(self, gray_frame: np.ndarray) -> Optional[PupilSample]:
        """Return a PupilSample if a pupil was detected, else None.
        Conventionally None means "no detection at all"; a low-confidence
        detection should still be returned so callers can apply their own gate."""

    def reset(self) -> None:
        """Discard any persistent model state (e.g. an eye-sphere model
        that needs to reconverge after the headset is re-seated). Default no-op."""
