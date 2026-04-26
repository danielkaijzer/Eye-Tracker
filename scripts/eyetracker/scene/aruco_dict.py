"""ArUco dictionary access + marker detection + marker-image generation.

Pure cv2; no Tk. The Tk-side rendering (PhotoImage caching, canvas draws) lives
in `display/tk_overlay.py` so this module stays usable from the App loop.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from scripts.eyetracker.config import ARUCO_DICT_NAME


_aruco_detector = None
_aruco_dict = None


def _require_aruco():
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is not available in this OpenCV build. "
            "Install opencv-contrib-python to enable ArUco support."
        )


def get_aruco_dict():
    """Return the cv2.aruco predefined dictionary named in config.ARUCO_DICT_NAME."""
    global _aruco_dict
    if _aruco_dict is not None:
        return _aruco_dict
    _require_aruco()
    if not hasattr(cv2.aruco, ARUCO_DICT_NAME):
        raise ImportError(
            f"cv2.aruco does not expose dictionary {ARUCO_DICT_NAME!r}. "
            "Check the spelling in config.ARUCO_DICT_NAME."
        )
    _aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    return _aruco_dict


def _get_detector():
    """Modern cv2.aruco.ArucoDetector if available, else None (caller falls back
    to the legacy free-function path)."""
    global _aruco_detector
    if _aruco_detector is not None:
        return _aruco_detector
    _require_aruco()
    try:
        params = cv2.aruco.DetectorParameters()
        _aruco_detector = cv2.aruco.ArucoDetector(get_aruco_dict(), params)
    except AttributeError:
        _aruco_detector = None
    return _aruco_detector


def detect_markers(scene_bgr: np.ndarray) -> Tuple[Optional[Tuple], Optional[np.ndarray]]:
    """Return (corners, ids) using the modern ArucoDetector if available,
    else the legacy detectMarkers call. corners/ids follow the OpenCV shape."""
    detector = _get_detector()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(scene_bgr)
        return corners, ids
    dictionary = get_aruco_dict()
    try:
        params = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(scene_bgr, dictionary, parameters=params)
    return corners, ids


def generate_marker_png(marker_id: int, size_px: int) -> bytes:
    """Render a marker to a PNG byte string. Useful for the Tk overlay
    (which wraps it in a tk.PhotoImage)."""
    _require_aruco()
    img = cv2.aruco.generateImageMarker(get_aruco_dict(), marker_id, size_px)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode ArUco marker {marker_id} to PNG.")
    return bytes(buf.tobytes())
