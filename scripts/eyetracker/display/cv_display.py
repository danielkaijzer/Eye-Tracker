"""OpenCV-window-based Display."""
from typing import Optional, Tuple

import cv2
import numpy as np

from scripts.eyetracker.config import DISPLAY_HEIGHT, DISPLAY_WIDTH
from scripts.eyetracker.display.base import Display, XY


EYE_WINDOW = "Eye Camera"
SCENE_WINDOW = "External Camera (Gaze)"


class CvDisplay(Display):
    """cv2.imshow-based runtime display. Eye + scene windows are positioned
    side-by-side; scene is downscaled to the configured display size."""

    def __init__(self,
                 with_scene: bool = True,
                 display_size: Tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT)):
        self.with_scene = with_scene
        self.display_w, self.display_h = display_size
        self._opened = False

    def open(self) -> None:
        cv2.namedWindow(EYE_WINDOW)
        cv2.moveWindow(EYE_WINDOW, 50, 50)
        if self.with_scene:
            cv2.namedWindow(SCENE_WINDOW)
            cv2.moveWindow(SCENE_WINDOW, 720, 50)
        self._opened = True

    def close(self) -> None:
        if self._opened:
            cv2.destroyAllWindows()
            self._opened = False

    def show_eye(self, frame: np.ndarray) -> None:
        cv2.imshow(EYE_WINDOW, frame)

    def show_scene(self, frame: np.ndarray, gaze_xy: Optional[XY]) -> None:
        """Show the scene-cam frame downscaled to the display window.
        gaze_xy, if provided, is in *scene-cam* pixel coords; we scale it
        here using the frame's own dimensions."""
        if not self.with_scene:
            return
        scene_h, scene_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.display_w, self.display_h))
        if gaze_xy is not None and scene_w > 0 and scene_h > 0:
            disp_x = int(gaze_xy[0] * self.display_w / scene_w)
            disp_y = int(gaze_xy[1] * self.display_h / scene_h)
            cv2.circle(resized, (disp_x, disp_y), 8, (0, 255, 0), -1)
        cv2.imshow(SCENE_WINDOW, resized)

    def poll_key(self) -> Tuple[Optional[str], int]:
        raw = cv2.waitKey(1) & 0xFF
        ch = chr(raw) if raw != 255 else None
        return ch, raw

    def wait_for_pause(self) -> None:
        cv2.waitKey(0)
