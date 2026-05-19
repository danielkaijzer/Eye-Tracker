"""Bivariate polynomial gaze mapper.

Supports degree 2 (6 coeffs: [1, x, y, x^2, y^2, xy]) and
degree 3 (10 coeffs: degree-2 + [x^3, y^3, x^2 y, x y^2]).

Two independent least-squares fits — one for scene_x, one for scene_y.
LOO error = leave-one-out reprojection error in scene-cam pixels;
per-point LOO errors are returned for two-pass recapture logic.
"""
import math
from typing import Optional

import numpy as np

from scripts.eyetracker.gaze.base import FitReport, GazeMapper, XY


_SUPPORTED_DEGREES = (2, 3)
_FEATURE_COUNT = {2: 6, 3: 10}


def _build_features(gx: float, gy: float, degree: int) -> np.ndarray:
    if degree == 2:
        return np.array([1.0, gx, gy, gx * gx, gy * gy, gx * gy])
    if degree == 3:
        return np.array([
            1.0, gx, gy,
            gx * gx, gy * gy, gx * gy,
            gx * gx * gx, gy * gy * gy,
            gx * gx * gy, gx * gy * gy,
        ])
    raise ValueError(f"Unsupported polynomial degree: {degree}")


class PolynomialGazeMapper(GazeMapper):
    def __init__(self, degree: int = 2):
        if degree not in _SUPPORTED_DEGREES:
            raise ValueError(f"degree must be one of {_SUPPORTED_DEGREES}, got {degree}")
        self.degree = degree
        self.coeffs_x: Optional[np.ndarray] = None
        self.coeffs_y: Optional[np.ndarray] = None

    def set_degree(self, degree: int) -> None:
        """Change the active degree. Invalidates any previous fit."""
        if degree not in _SUPPORTED_DEGREES:
            raise ValueError(f"degree must be one of {_SUPPORTED_DEGREES}, got {degree}")
        if degree != self.degree:
            self.degree = degree
            self.coeffs_x = None
            self.coeffs_y = None

    def fit(self, pupil_pts: np.ndarray, scene_pts: np.ndarray) -> FitReport:
        pupil_pts = np.asarray(pupil_pts)
        scene_pts = np.asarray(scene_pts)
        n = len(pupil_pts)
        k = _FEATURE_COUNT[self.degree]
        if n < k:
            raise ValueError(
                f"Need at least {k} calibration points for degree {self.degree}, have {n}."
            )
        if len(scene_pts) != n:
            raise ValueError(
                f"pupil_pts ({n}) and scene_pts ({len(scene_pts)}) length mismatch."
            )

        A = np.zeros((n, k))
        bx = np.zeros(n)
        by = np.zeros(n)
        for i, (pp, sp) in enumerate(zip(pupil_pts, scene_pts)):
            A[i] = _build_features(pp[0], pp[1], self.degree)
            bx[i] = sp[0]
            by[i] = sp[1]

        cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        self.coeffs_x = cx
        self.coeffs_y = cy

        errors = np.zeros(n)
        for i in range(n):
            A_loo = np.delete(A, i, axis=0)
            bx_loo = np.delete(bx, i)
            by_loo = np.delete(by, i)
            cx_loo, _, _, _ = np.linalg.lstsq(A_loo, bx_loo, rcond=None)
            cy_loo, _, _, _ = np.linalg.lstsq(A_loo, by_loo, rcond=None)
            pred_x = A[i] @ cx_loo
            pred_y = A[i] @ cy_loo
            errors[i] = math.sqrt((pred_x - bx[i]) ** 2 + (pred_y - by[i]) ** 2)

        return FitReport(n_points=n,
                         loo_avg_err=float(np.mean(errors)),
                         loo_max_err=float(np.max(errors)),
                         per_point_errs=errors)

    def predict(self, pupil_xy: XY) -> XY:
        if not self.is_fitted():
            raise RuntimeError("PolynomialGazeMapper.predict() called before fit().")
        feat = _build_features(pupil_xy[0], pupil_xy[1], self.degree)
        return float(feat @ self.coeffs_x), float(feat @ self.coeffs_y)

    def is_fitted(self) -> bool:
        return self.coeffs_x is not None and self.coeffs_y is not None

    def state_dict(self) -> dict:
        if not self.is_fitted():
            return {}
        return {
            "poly_coeffs_x": self.coeffs_x,
            "poly_coeffs_y": self.coeffs_y,
            "poly_degree": self.degree,
        }

    def load_state_dict(self, state: dict) -> None:
        cx = np.asarray(state["poly_coeffs_x"])
        cy = np.asarray(state["poly_coeffs_y"])
        # Prefer explicit degree, fall back to inferring from coeff length
        # so old npz files (no poly_degree field) still load correctly.
        degree = state.get("poly_degree")
        if degree is None:
            degree = next((d for d, k in _FEATURE_COUNT.items() if k == len(cx)), None)
            if degree is None:
                raise ValueError(
                    f"Cannot infer polynomial degree from {len(cx)} coefficients."
                )
        self.degree = int(degree)
        self.coeffs_x = cx
        self.coeffs_y = cy
