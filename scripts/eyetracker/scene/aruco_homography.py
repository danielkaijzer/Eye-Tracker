"""ArUco-based screen<->scene homography (the current TargetMapper impl).

Four corner markers (IDs 0=TL, 1=TR, 2=BR, 3=BL) sit inside white quiet zones
at each corner of the calibration screen. We detect them in the scene-cam
frame and solve for the screen->scene homography via cv2.findHomography.

Stateful pieces:
- screen size (must be set before screen_anchor_points / quiet_zone_origins
  return non-empty results)
- per-frame detection cache written by process_frame(): last marker count +
  found IDs (for the Tk overlay HUD) and the raw corners/ids (so
  cached_homography() can solve without re-running detection — detection on
  a 1080p frame is the expensive step and used to run twice per frame during
  collection)
"""
from typing import Dict, FrozenSet, Optional, Tuple

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
        self.last_found_ids: FrozenSet[int] = frozenset()
        # Raw detection from the most recent process_frame() call.
        self._cached_corners: Optional[Tuple] = None
        self._cached_ids: Optional[np.ndarray] = None

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

    def process_frame(self, scene_bgr: Optional[np.ndarray]) -> int:
        """Run marker detection ONCE for this scene frame and cache the result.

        Call once per new scene frame (the App loop does). Updates
        last_marker_count / last_found_ids for the overlay HUD and stores the
        raw corners/ids so cached_homography() can solve without re-detecting.
        Returns the marker count."""
        if scene_bgr is None:
            corners, ids = None, None
        else:
            corners, ids = detect_markers(scene_bgr)
        self._cached_corners = corners
        self._cached_ids = ids
        if ids is None:
            self.last_found_ids = frozenset()
        else:
            self.last_found_ids = (frozenset(int(i) for i in ids.flatten())
                                   & frozenset(self.marker_ids))
        self.last_marker_count = len(self.last_found_ids)
        return self.last_marker_count

    def draw_cached_detections(self, frame_bgr: np.ndarray) -> None:
        """Outline the markers found by the last process_frame() call onto
        frame_bgr in place (pass the same frame the cache came from) — used
        by the calibration overlay's troubleshooting preview."""
        if self._cached_ids is None or len(self._cached_ids) == 0:
            return
        cv2.aruco.drawDetectedMarkers(frame_bgr, self._cached_corners,
                                      self._cached_ids)

    def cached_homography(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Solve the homography from the detection cached by the most recent
        process_frame() call — no re-detection. Only valid for that same frame;
        callers holding a different frame must use compute_homography()."""
        return self._homography_from(self._cached_corners, self._cached_ids)

    def compute_homography(self, scene_bgr: Optional[np.ndarray]
                           ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Detect markers in scene_bgr and solve for the 3x3 screen->scene
        homography H and the mean reprojection error in scene-cam pixels.
        (None, None) on failure. Prefer cached_homography() when
        process_frame() already ran on this exact frame."""
        if scene_bgr is None:
            return None, None
        corners, ids = detect_markers(scene_bgr)
        return self._homography_from(corners, ids)

    def _homography_from(self, corners, ids
                         ) -> Tuple[Optional[np.ndarray], Optional[float]]:
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
