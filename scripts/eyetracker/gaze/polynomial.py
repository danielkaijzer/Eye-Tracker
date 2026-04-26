"""2nd-degree bivariate polynomial gaze mapper.

Feature row per sample: [1, gx, gy, gx^2, gy^2, gx*gy].
Two independent least-squares fits — one for scene_x, one for scene_y.
LOO error = leave-one-out reprojection error in scene-cam pixels.
"""
import math
from typing import Optional

import numpy as np

from scripts.eyetracker.gaze.base import FitReport, GazeMapper, XY


def _build_features(gx: float, gy: float) -> np.ndarray:
    return np.array([1.0, gx, gy, gx * gx, gy * gy, gx * gy])


class PolynomialGazeMapper(GazeMapper):
    def __init__(self):
        self.coeffs_x: Optional[np.ndarray] = None
        self.coeffs_y: Optional[np.ndarray] = None

    def fit(self, pupil_pts: np.ndarray, scene_pts: np.ndarray) -> FitReport:
        pupil_pts = np.asarray(pupil_pts)
        scene_pts = np.asarray(scene_pts)
        n = len(pupil_pts)
        if n < 6:
            raise ValueError(f"Need at least 6 calibration points, have {n}.")
        if len(scene_pts) != n:
            raise ValueError(
                f"pupil_pts ({n}) and scene_pts ({len(scene_pts)}) length mismatch."
            )

        A = np.zeros((n, 6))
        bx = np.zeros(n)
        by = np.zeros(n)
        for i, (pp, sp) in enumerate(zip(pupil_pts, scene_pts)):
            A[i] = _build_features(pp[0], pp[1])
            bx[i] = sp[0]
            by[i] = sp[1]

        cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        self.coeffs_x = cx
        self.coeffs_y = cy

        errors = []
        for i in range(n):
            A_loo = np.delete(A, i, axis=0)
            bx_loo = np.delete(bx, i)
            by_loo = np.delete(by, i)
            cx_loo, _, _, _ = np.linalg.lstsq(A_loo, bx_loo, rcond=None)
            cy_loo, _, _, _ = np.linalg.lstsq(A_loo, by_loo, rcond=None)
            pred_x = A[i] @ cx_loo
            pred_y = A[i] @ cy_loo
            errors.append(math.sqrt((pred_x - bx[i]) ** 2 + (pred_y - by[i]) ** 2))

        return FitReport(n_points=n,
                         loo_avg_err=float(np.mean(errors)),
                         loo_max_err=float(np.max(errors)))

    def predict(self, pupil_xy: XY) -> XY:
        if not self.is_fitted():
            raise RuntimeError("PolynomialGazeMapper.predict() called before fit().")
        feat = _build_features(pupil_xy[0], pupil_xy[1])
        return float(feat @ self.coeffs_x), float(feat @ self.coeffs_y)

    def is_fitted(self) -> bool:
        return self.coeffs_x is not None and self.coeffs_y is not None

    def state_dict(self) -> dict:
        if not self.is_fitted():
            return {}
        return {
            "poly_coeffs_x": self.coeffs_x,
            "poly_coeffs_y": self.coeffs_y,
        }

    def load_state_dict(self, state: dict) -> None:
        self.coeffs_x = np.asarray(state["poly_coeffs_x"])
        self.coeffs_y = np.asarray(state["poly_coeffs_y"])
