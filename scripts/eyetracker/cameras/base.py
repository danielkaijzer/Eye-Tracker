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
