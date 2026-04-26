"""Filesystem locations for calibration artifacts.

The mapper coefficients + last-session metadata live in `calibration_pupil.npz`
at the package root. Per-session image dumps + labels.csv live under
`<repo>/data/calibration/session_<timestamp>/`.
"""
import os


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))


def calibration_path() -> str:
    return os.path.join(_PACKAGE_DIR, "calibration_pupil.npz")


def history_path() -> str:
    return os.path.join(_PACKAGE_DIR, "calibration_pupil_history.npz")


def dataset_root() -> str:
    return os.path.join(_REPO_ROOT, "data", "calibration")
