"""GazeMapper interface — pluggable pupil-pixel -> scene-cam-pixel mapper.

Today's only implementation is `PolynomialGazeMapper` (2nd-degree bivariate
polynomial). Future candidates: thin-plate spline, weighted least-squares.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


XY = Tuple[float, float]


@dataclass
class FitReport:
    """Diagnostic info from a single .fit() call. Used to print / warn."""
    n_points: int
    loo_avg_err: float    # leave-one-out reprojection error, mean (px)
    loo_max_err: float    # leave-one-out reprojection error, max (px)


class GazeMapper(ABC):
    @abstractmethod
    def fit(self, pupil_pts: np.ndarray, scene_pts: np.ndarray) -> FitReport:
        """Fit the mapping from pupil-pixel to scene-cam-pixel coordinates.
        pupil_pts: (N, 2) array; scene_pts: (N, 2) array."""

    @abstractmethod
    def predict(self, pupil_xy: XY) -> XY:
        """Return the scene-cam pixel for the given pupil pixel.
        Caller is responsible for clipping / smoothing."""

    @abstractmethod
    def is_fitted(self) -> bool:
        """True iff fit() has been called or load_state_dict() restored coefficients."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Serialize fit state (e.g. coefficients) to a plain dict.
        Persistence layer turns this into npz / json / whatever."""

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore fit state from a dict produced by state_dict()."""
