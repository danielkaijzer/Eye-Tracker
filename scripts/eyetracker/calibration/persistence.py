"""Calibration persistence: JSON model + per-session metadata + labels.csv.

Three on-disk artifacts, each in the format that fits its job:

- `calibration.json` — the live polynomial model restored at runtime. Just the
  coefficients + the few fields the runtime needs. Human-readable, no pickle.
- `session_<ts>/metadata.json` — the per-session dataset record: camera
  intrinsics/extrinsics, hardware/subject provenance, sizes, aruco config, and a
  fit summary. Self-contained so a session is interpretable in isolation.
- `session_<ts>/labels.csv` — one row per accepted calibration sample (the raw
  ground-truth pairs); appended during capture so a crash mid-session keeps
  whatever was already written.

The raw sample arrays are NOT duplicated into calibration.json — they live in
the session's labels.csv (and `metadata.json` records `source_session`).
"""
import csv
import datetime
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from scripts.eyetracker.calibration.paths import (
    calibration_path,
    dataset_root,
    scene_intrinsics_path,
)
from scripts.eyetracker.config import (
    ARUCO_DICT_NAME,
    ARUCO_IDS,
    ARUCO_MARKER_PX,
    ARUCO_QUIET_ZONE_PX,
    EYE_CAM_FOV_DEG,
    EYE_CAM_RESOLUTION,
)
from scripts.eyetracker.gaze.base import FitReport, GazeMapper


LABELS_CSV_HEADER = [
    "image_path", "fixation_id", "x_screen", "y_screen",
    "pupil_x", "pupil_y", "confidence", "timestamp",
    "scene_target_x", "scene_target_y",
]


@dataclass
class CalibrationSnapshot:
    """Everything captured during one calibration session, ready to persist."""
    pupil_vectors: np.ndarray         # (N, 2)
    scene_points: np.ndarray          # (N, 2) — labels in scene-cam pixels
    screen_points: np.ndarray         # (N, 2) — on-screen target positions
    aruco_screen_centers: np.ndarray  # (4, 2) or (0, 2) if no ArUco context
    scene_size: Optional[Tuple[int, int]]
    screen_size: Optional[Tuple[int, int]]


@dataclass
class LoadedCalibration:
    age_hrs: float
    scene_size: Optional[Tuple[int, int]]
    screen_size: Optional[Tuple[int, int]]


def _to_list(arr) -> Optional[list]:
    """numpy array (or None) -> JSON-serializable nested list (or None)."""
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def save_calibration(snapshot: CalibrationSnapshot, mapper: GazeMapper,
                     source_session: Optional[str] = None) -> None:
    """Write the live model to `calibration.json`. `source_session` names the
    session dir whose labels.csv holds the samples this fit came from."""
    state = mapper.state_dict()
    data = {
        "coord_space": "scene",
        "timestamp": time.time(),
        "source_session": source_session,
        "scene_size": list(snapshot.scene_size) if snapshot.scene_size else None,
        "screen_size": list(snapshot.screen_size) if snapshot.screen_size else None,
        "model": {
            "type": "polynomial",
            "degree": int(state["poly_degree"]),
            "coeffs_x": _to_list(state["poly_coeffs_x"]),
            "coeffs_y": _to_list(state["poly_coeffs_y"]),
        },
    }
    path = calibration_path()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Calibration model saved to {path}")


def load_calibration(mapper: GazeMapper) -> Optional[LoadedCalibration]:
    """Restore mapper coefficients from `calibration.json` and return session
    metadata. Returns None if nothing is saved or if the file's coord_space
    does not match this build (we never want to mix screen/scene fits)."""
    path = calibration_path()
    if not os.path.exists(path):
        print("No saved calibration found.")
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("coord_space") != "scene":
        print("  ERROR: saved calibration is not in scene-cam coord space. "
              "Recalibrate with 'c'.")
        return None
    model = data["model"]
    mapper.load_state_dict({
        "poly_coeffs_x": np.asarray(model["coeffs_x"]),
        "poly_coeffs_y": np.asarray(model["coeffs_y"]),
        "poly_degree": model.get("degree"),
    })
    scene_size = tuple(data["scene_size"]) if data.get("scene_size") else None
    screen_size = tuple(data["screen_size"]) if data.get("screen_size") else None
    age_hrs = (time.time() - float(data["timestamp"])) / 3600
    print(f"Calibration loaded (age: {age_hrs:.1f}h, scene={scene_size}).")
    if age_hrs > 24:
        print("  WARNING: Calibration is >24h old. Consider recalibrating.")
    return LoadedCalibration(age_hrs=age_hrs, scene_size=scene_size,
                             screen_size=screen_size)


def _scene_intrinsics_snapshot() -> Optional[dict]:
    """Inline copy of the current scene-cam intrinsics, or None if not yet
    calibrated. Denormalized into each session so it stays interpretable even
    if the central intrinsics file later changes."""
    path = scene_intrinsics_path()
    if not os.path.exists(path):
        return None
    with open(path) as f:
        intr = json.load(f)
    return {
        "K": intr.get("K"),
        "dist": intr.get("dist"),
        "reproj_rms": intr.get("reproj_rms"),
    }


def write_session_metadata(session_dir: str,
                           snapshot: CalibrationSnapshot,
                           report: FitReport,
                           degree: int) -> None:
    """Emit `metadata.json` for a finished session: the self-contained dataset
    record. Hardware/subject fields that the pipeline does not yet produce
    (eye intrinsics, extrinsics, subject id, kappa, versions) are written as
    null placeholders — the extrinsics jig and richer capture fill them later."""
    sw, sh = snapshot.scene_size if snapshot.scene_size else (None, None)
    pw, ph = snapshot.screen_size if snapshot.screen_size else (None, None)
    eye_w, eye_h = EYE_CAM_RESOLUTION
    aruco_dict_id = int(cv2.aruco.DICT_4X4_50) if hasattr(cv2, "aruco") else -1
    meta = {
        "session_id": os.path.basename(os.path.normpath(session_dir)),
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "timestamp": time.time(),
        "phase": "calibration",
        "subject_id": None,
        "glasses": None,
        "headset_model_version": None,
        "kappa_deg": None,
        "software": {"pupil_detector": None, "pye3d": None, "app_git_sha": None},
        "screen": {"width": pw, "height": ph},
        "scene_cam": {"width": sw, "height": sh, "fps": None, "identifier": None},
        "eye_cam": {"width": eye_w, "height": eye_h, "fps": None,
                    "fov_deg": EYE_CAM_FOV_DEG, "identifier": None},
        "aruco": {
            "dict_name": ARUCO_DICT_NAME,
            "dict_id": aruco_dict_id,
            "marker_px": ARUCO_MARKER_PX,
            "quiet_zone_px": ARUCO_QUIET_ZONE_PX,
            "ids": list(ARUCO_IDS),
            "screen_centers": _to_list(snapshot.aruco_screen_centers),
        },
        "rig_calibration_id": None,
        "intrinsics": {"eye": None, "scene": _scene_intrinsics_snapshot()},
        "extrinsics": None,
        "fit": {
            "degree": int(degree),
            "n_points": int(report.n_points),
            "loo_avg_px": float(report.loo_avg_err),
            "loo_max_px": float(report.loo_max_err),
        },
    }
    path = os.path.join(session_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Session metadata saved to {path}")


def begin_session(root: Optional[str] = None) -> Tuple[str, str]:
    """Create a `session_<timestamp>/` directory under `root` (default
    `dataset_root()`), write the labels.csv header, and return
    (session_dir, labels_csv_path)."""
    base = root if root is not None else dataset_root()
    session_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir = os.path.join(base, session_name)
    os.makedirs(session_dir, exist_ok=True)
    labels_path = os.path.join(session_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        csv.writer(f).writerow(LABELS_CSV_HEADER)
    return session_dir, labels_path


def append_label_rows(labels_path: str, rows: list) -> None:
    with open(labels_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)
