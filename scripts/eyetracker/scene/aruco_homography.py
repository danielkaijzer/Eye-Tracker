"""ArUco-based screen<->scene homography (the current TargetMapper impl).

Four corner markers (IDs 0=TL, 1=TR, 2=BR, 3=BL) sit inside white quiet zones
at each corner of the calibration screen. We detect them in the scene-cam
frame and solve for the screen->scene homography via cv2.findHomography.

Stateful pieces:
- screen size (must be set before screen_anchor_points / quiet_zone_origins
  return non-empty results)
- last marker count (cached for HUD display by the Tk overlay)
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from scripts.eyetracker.config import ARUCO_IDS, ARUCO_QUIET_ZONE_PX
from scripts.eyetracker.scene.aruco_dict import detect_markers
from scripts.eyetracker.scene.target_mapper import TargetMapper, XY


class ArucoHomography(TargetMapper):
    def __init__(self,
                 marker_ids: Tuple[int, ...] = ARUCO_IDS,
                 quiet_zone_px: int = ARUCO_QUIET_ZONE_PX):
        self.marker_ids = marker_ids
        self.quiet_zone_px = quiet_zone_px
        self.screen_width: Optional[int] = None
        self.screen_height: Optional[int] = None
        self.last_marker_count = 0

    def set_screen_size(self, width: int, height: int) -> None:
        self.screen_width = width
        self.screen_height = height

    def screen_anchor_points(self) -> Dict[int, XY]:
        """Dict {marker_id: (cx, cy)} of on-screen marker centers, or {} if
        screen size hasn't been set yet."""
        if self.screen_width is None or self.screen_height is None:
            return {}
        q = self.quiet_zone_px
        half = q / 2.0
        sw, sh = self.screen_width, self.screen_height
        return {
            self.marker_ids[0]: (half, half),                  # TL
            self.marker_ids[1]: (sw - half, half),             # TR
            self.marker_ids[2]: (sw - half, sh - half),        # BR
            self.marker_ids[3]: (half, sh - half),             # BL
        }

    def quiet_zone_origins(self) -> Dict[int, Tuple[int, int]]:
        """Dict {marker_id: (x_nw, y_nw)} of quiet-zone NW corners hugging
        each screen corner."""
        if self.screen_width is None or self.screen_height is None:
            return {}
        q = self.quiet_zone_px
        sw, sh = self.screen_width, self.screen_height
        return {
            self.marker_ids[0]: (0, 0),
            self.marker_ids[1]: (sw - q, 0),
            self.marker_ids[2]: (sw - q, sh - q),
            self.marker_ids[3]: (0, sh - q),
        }

    def update_marker_count(self, scene_bgr: Optional[np.ndarray]) -> int:
        """Detect markers and cache how many of self.marker_ids were found."""
        if scene_bgr is None:
            self.last_marker_count = 0
            return 0
        _, ids = detect_markers(scene_bgr)
        if ids is None:
            self.last_marker_count = 0
            return 0
        found = {int(i) for i in ids.flatten()} & set(self.marker_ids)
        self.last_marker_count = len(found)
        return self.last_marker_count

    def compute_homography(self, scene_bgr: Optional[np.ndarray]
                           ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Solve for the 3x3 screen->scene homography H and the mean
        reprojection error in scene-cam pixels. (None, None) on failure."""
        if scene_bgr is None:
            return None, None
        corners, ids = detect_markers(scene_bgr)
        if ids is None or len(ids) < 4:
            return None, None
        id_to_center: Dict[int, np.ndarray] = {}
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in self.marker_ids:
                continue
            id_to_center[mid] = c.reshape(-1, 2).mean(axis=0)
        if any(i not in id_to_center for i in self.marker_ids):
            return None, None

        anchors = self.screen_anchor_points()
        if not anchors:
            return None, None

        screen_pts = np.array([anchors[i] for i in self.marker_ids], dtype=np.float32)
        scene_pts = np.array([id_to_center[i] for i in self.marker_ids], dtype=np.float32)

        H, _ = cv2.findHomography(screen_pts, scene_pts, method=0)
        if H is None:
            return None, None

        homog = np.hstack([screen_pts, np.ones((4, 1), dtype=np.float32)])
        projected = (H @ homog.T).T
        ws = projected[:, 2:3]
        if np.any(np.abs(ws) < 1e-9):
            return None, None
        projected_xy = projected[:, :2] / ws
        errs = np.linalg.norm(projected_xy - scene_pts, axis=1)
        return H, float(np.mean(errs))

    # ---- TargetMapper interface ----

    def is_ready(self, scene_frame: np.ndarray) -> bool:
        H, _ = self.compute_homography(scene_frame)
        return H is not None

    def screen_to_scene(self, xy_screen: XY, scene_frame: np.ndarray) -> Optional[XY]:
        H, _ = self.compute_homography(scene_frame)
        if H is None:
            return None
        v = np.array([xy_screen[0], xy_screen[1], 1.0], dtype=float)
        out = H @ v
        if abs(out[2]) < 1e-9:
            return None
        return (float(out[0] / out[2]), float(out[1] / out[2]))

    def scene_to_screen(self, xy_scene: XY, scene_frame: np.ndarray) -> Optional[XY]:
        H, _ = self.compute_homography(scene_frame)
        if H is None:
            return None
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        v = np.array([xy_scene[0], xy_scene[1], 1.0], dtype=float)
        out = H_inv @ v
        if abs(out[2]) < 1e-9:
            return None
        return (float(out[0] / out[2]), float(out[1] / out[2]))
