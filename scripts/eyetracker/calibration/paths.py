"""Filesystem locations for calibration artifacts.

The live mapper model lives in `calibration.json` at the package root, next to
the scene-cam intrinsics (`scene_intrinsics.json`). Per-session data — eye/scene
frames, `labels.csv`, and `metadata.json` — lives under
`<repo>/data/calibration/session_<timestamp>/`. Camera-rig calibrations produced
by the extrinsics jig live under `<repo>/rig_calibrations/`.
"""
import os


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))


def calibration_path() -> str:
    """The live polynomial model restored at runtime."""
    return os.path.join(_PACKAGE_DIR, "calibration.json")


def scene_intrinsics_path() -> str:
    """Scene-cam intrinsics (K, distortion) produced by the intrinsics tool."""
    return os.path.join(_PACKAGE_DIR, "scene_intrinsics.json")


def dataset_root() -> str:
    """Parent dir holding every captured `session_<timestamp>/`."""
    return os.path.join(_REPO_ROOT, "data", "calibration")


def rig_calibrations_root() -> str:
    """Parent dir for camera-rig calibrations (intrinsics + extrinsics) from
    the extrinsics jig. One `<rig_id>.json` per jig run."""
    return os.path.join(_REPO_ROOT, "rig_calibrations")
