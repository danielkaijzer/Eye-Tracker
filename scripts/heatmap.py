import cv2
import numpy as np
import time
import os

class Heatmap:
    BLUR_KERNEL = 51 #width of each gaussian blob (larger = smoother and blurrier)
    OVERLAY_ALPHA = 0.6 #alpha for blending the heatmap against the background
    MIN_POINTS_TO_DISPLAY = 5 #minimum points before the heatmap shows color
    WINDOW_NAME = "Heatmap"


    def __init__(self, width=640, height=480, output_dir=None):
        self.w = width
        self.h = height
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))

        #float32 accumulator to avoid overflow on long sessions
        self._accum = np.zeros((height, width), dtype=np.float32)
        self._point_count = 0
        self._visible = True
        self._session_start = time.time()

        #background drawn behind the heatmap
        self._background = np.full((height, width, 3), 30, dtype=np.uint8)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, width, height)

    #--------------------------------------------------------------------------------------
    #Public API
    def add_point(self, x: int, y: int, weight: float = 1.0):
        """records a single gaze point, called after calibration every frame"""
        x = int(np.clip(x, 0, self.w - 1))
        y = int(np.clip(y, 0, self.h - 1))
        self._accum[y, x] = weight
        self._point_count += 1


    def render(self):
        """recompute and show the heatmap window, called once per frame"""
        if not self._visible:
            return
        frame = self._build_frame()
        self._draw_stats(frame)
        cv2.imshow(self.WINDOW_NAME, frame)


    def save(self, filename: str = None):
        """saves the current heatmap to a png"""
        if filename is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            filename = f"heatmap_{ts}.png"
        path = os.path.join(self.output_dir, filename)
        frame = self._build_frame()
        self._draw_stats(frame)
        cv2.imwrite(path, frame)
        print(f"[heatmap] saved to {path}  ({self._point_count} points)")
        return path


    def toggle(self):
        """shows or hides the live window"""
        self._visible = not self._visible
        if not self._visible:
            cv2.destroyWindow(self.WINDOW_NAME)
        else:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, width=self.w, height=self.h)


    def set_background(self, bgr_image: np.ndarray):
        """swaps the background behind the heatmap (resized to match heatmap automatically)"""
        self._background = cv2.resize(bgr_image, (self.w, self.h))


    def destroy(self):
        """destroys the heatmap window"""
        cv2.destroyWindow(self.WINDOW_NAME)

    #---------------------------------------------------------------------------------
    #HELPERS

    def _build_frame(self) -> np.ndarray:
        """gaussian-blurs the accumulator, maps it to a color heatmap, then overlays it on the background"""
        if self._point_count < Heatmap.MIN_POINTS_TO_DISPLAY:
            return self._background.copy()

        #blurs for the smooth blobs
        blurred = cv2.GaussianBlur(self._accum, (self.BLUR_KERNEL, self.BLUR_KERNEL), 0)

        #normalizes to 0-255
        norm = cv2.normalize(blurred, 0, 255, cv2.NORM_MINMAX)
        heat_gray = norm.astype(np.uint8)

        #apply JET colormap, so blue=low and red=high
        heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)

        #masks the empty areas so it only shows colors where there's data
        mask = (heat_gray > 5).astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=2)

        #
