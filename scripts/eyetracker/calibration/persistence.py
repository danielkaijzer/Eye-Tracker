"""Calibration .npz save/load + per-session dir + labels.csv header.

The .npz schema is intentionally compatible with what the legacy single-file
script wrote:

  poly_coeffs_x, poly_coeffs_y, vectors, scene_points, screen_points,
  coord_space="scene", scene_width, scene_height, screen_width, screen_height,
  aruco_dict_id, aruco_dict_name, aruco_marker_px, aruco_quiet_zone_px,
  aruco_screen_centers, timestamp

Old history files used `all_points` (screen-space). We treat any history file
without `all_scene_points` as fresh so the two coord spaces don't get mixed.
"""
import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from scripts.eyetracker.calibration.paths import (
    calibration_path,
    dataset_root,
    history_path,
)
from scripts.eyetracker.config import (
    ARUCO_DICT_NAME,
    ARUCO_IDS,
    ARUCO_MARKER_PX,
    ARUCO_QUIET_ZONE_PX,
)
from scripts.eyetracker.gaze.base import GazeMapper


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


def save_calibration(snapshot: CalibrationSnapshot, mapper: GazeMapper) -> None:
    """Write the mapper state + session metadata to `calibration_pupil.npz`,
    and append the (vectors, scene_points) pair to the history file."""
    path = calibration_path()
    aruco_dict_id = int(cv2.aruco.DICT_4X4_50) if hasattr(cv2, "aruco") else -1
    state = mapper.state_dict()
    sw, sh = snapshot.scene_size if snapshot.scene_size is not None else (None, None)
    pw, ph = snapshot.screen_size if snapshot.screen_size is not None else (None, None)
    np.savez(path,
             poly_coeffs_x=state["poly_coeffs_x"],
             poly_coeffs_y=state["poly_coeffs_y"],
             vectors=snapshot.pupil_vectors,
             scene_points=snapshot.scene_points,
             screen_points=snapshot.screen_points,
             coord_space="scene",
             scene_width=sw,
             scene_height=sh,
             screen_width=pw,
             screen_height=ph,
             aruco_dict_id=aruco_dict_id,
             aruco_dict_name=ARUCO_DICT_NAME,
             aruco_marker_px=ARUCO_MARKER_PX,
             aruco_quiet_zone_px=ARUCO_QUIET_ZONE_PX,
             aruco_screen_centers=snapshot.aruco_screen_centers,
             timestamp=time.time())
    print(f"  Calibration saved to {path}")
    _append_history(snapshot.pupil_vectors, snapshot.scene_points)


def _append_history(vectors: np.ndarray, scene_points: np.ndarray) -> None:
    hist_path = history_path()
    if os.path.exists(hist_path):
        old = np.load(hist_path, allow_pickle=True)
        if "all_scene_points" in old.files:
            old_vectors = list(old["all_vectors"])
            old_scene_points = list(old["all_scene_points"])
        else:
            old_vectors = []
            old_scene_points = []
    else:
        old_vectors = []
        old_scene_points = []
    old_vectors.append(vectors)
    old_scene_points.append(scene_points)
    np.savez(hist_path,
             all_vectors=np.array(old_vectors, dtype=object),
             all_scene_points=np.array(old_scene_points, dtype=object))
    total_pts = sum(len(v) for v in old_vectors)
    print(f"  History: {len(old_vectors)} sessions, {total_pts} total points.")


def load_calibration(mapper: GazeMapper) -> Optional[LoadedCalibration]:
    """Restore mapper coefficients from disk and return session metadata.
    Returns None if no calibration is saved or if the file's coord_space
    does not match this build (we never want to mix screen/scene fits)."""
    path = calibration_path()
    if not os.path.exists(path):
        print("No saved calibration found.")
        return None
    data = np.load(path, allow_pickle=True)
    if "coord_space" not in data.files or str(data["coord_space"]) != "scene":
        print("  ERROR: saved calibration is not in scene-cam coord space. "
              "Recalibrate with 'c'.")
        return None
    mapper.load_state_dict({
        "poly_coeffs_x": data["poly_coeffs_x"],
        "poly_coeffs_y": data["poly_coeffs_y"],
    })
    screen_size = None
    scene_size = None
    if "screen_width" in data.files and "screen_height" in data.files:
        screen_size = (int(data["screen_width"]), int(data["screen_height"]))
    if "scene_width" in data.files and "scene_height" in data.files:
        scene_size = (int(data["scene_width"]), int(data["scene_height"]))
    age_hrs = (time.time() - float(data["timestamp"])) / 3600
    print(f"Calibration loaded (age: {age_hrs:.1f}h, scene={scene_size}).")
    if age_hrs > 24:
        print("  WARNING: Calibration is >24h old. Consider recalibrating.")
    return LoadedCalibration(age_hrs=age_hrs, scene_size=scene_size,
                             screen_size=screen_size)


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
