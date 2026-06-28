"""CameraSource interface — frame providers for the App loop."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CameraSource(ABC):
    """Frame provider. After open() succeeds, .width/.height reflect the
    camera's actual frame size (which may differ from any requested size)."""

    width: int = 0
    height: int = 0

    @abstractmethod
    def open(self) -> bool:
        """Probe the source and apply any per-source settings.
        Return True on success."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if the source is exhausted/failed."""

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources. Safe to call when not open."""

    # ---- runtime exposure control (optional) --------------------------------
    # Live-controllable cameras override these; sources that can't (e.g. a
    # recorded-video source) keep the no-op defaults so the App can call them
    # unconditionally without knowing the concrete type.

    def nudge_exposure(self, direction: int, step_fraction: float) -> bool:
        """Step the manual exposure by a fraction of its range (direction +1/-1).
        Fractional so it feels consistent regardless of the underlying control's
        scale. Return True if applied."""
        return False

    def exposure_status(self) -> Optional[str]:
        """Short human label for the on-screen readout (e.g. "exp 400"), or None
        if the source has no exposure control."""
        return None
