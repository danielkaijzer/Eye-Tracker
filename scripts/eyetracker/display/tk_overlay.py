"""Tk-based fullscreen calibration overlay.

Owns the Tk root + canvas, an internal key queue (populated by Tk key bindings),
and a cache of `tk.PhotoImage` instances for the four ArUco corner markers.
Reads from a CalibrationRoutine to repaint each frame; never mutates routine
state directly.
"""
import base64
import tkinter as tk
from typing import List, Optional, Tuple

from scripts.eyetracker.config import (
    ARUCO_MARKER_PX,
    ARUCO_QUIET_ZONE_PX,
    CALIB_SAMPLES,
)
from scripts.eyetracker.display.base import CalibrationOverlay
from scripts.eyetracker.scene.aruco_dict import generate_marker_png
from scripts.eyetracker.scene.aruco_homography import ArucoHomography


class TkCalibrationOverlay(CalibrationOverlay):
    def __init__(self, target_mapper: ArucoHomography):
        self.target_mapper = target_mapper
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._key_queue: List[str] = []
        self._photo_images: list = []
        self.screen_width = 0
        self.screen_height = 0

    # ---- lifecycle ----

    def open(self) -> Tuple[int, int]:
        self._key_queue = []
        self._photo_images = []
        root = tk.Tk()
        root.configure(bg="black")
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.config(cursor="none")
        root.update_idletasks()
        self.screen_width = root.winfo_width()
        self.screen_height = root.winfo_height()

        canvas = tk.Canvas(root, width=self.screen_width,
                           height=self.screen_height,
                           bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        for ch in ("c", "s", "q"):
            root.bind(f"<KeyPress-{ch}>",
                      lambda _e, _ch=ch: self._on_key(_ch))
        root.focus_force()
        root.update()

        self._root = root
        self._canvas = canvas
        return self.screen_width, self.screen_height

    def is_open(self) -> bool:
        return self._root is not None

    def close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except tk.TclError:
                pass
        self._root = None
        self._canvas = None
        self._photo_images = []

    def pump(self) -> None:
        if self._root is None:
            return
        try:
            self._root.update()
        except tk.TclError:
            self.close()

    def poll_key(self) -> Optional[str]:
        return self._key_queue.pop(0) if self._key_queue else None

    # ---- rendering ----

    def render(self, routine) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        canvas.delete("all")

        if getattr(routine, "awaiting_pose", False):
            self._render_pose_break(routine)
            return

        active_idx = routine.current_idx

        for i, pt in enumerate(routine.targets):
            x, y = pt
            if i in routine.skipped_indices:
                r = 8
                canvas.create_line(x - r, y - r, x + r, y + r,
                                   fill="#808080", width=3)
                canvas.create_line(x - r, y + r, x + r, y - r,
                                   fill="#808080", width=3)
            elif i == active_idx:
                r = 20
                canvas.create_oval(x - r, y - r, x + r, y + r,
                                   fill="red", outline="")
                if routine.is_collecting:
                    rr = 30
                    canvas.create_oval(x - rr, y - rr, x + rr, y + rr,
                                       outline="#ffa500", width=3)
            elif i < active_idx:
                r = 8
                canvas.create_oval(x - r, y - r, x + r, y + r,
                                   fill="#00b400", outline="")
            else:
                r = 6
                canvas.create_oval(x - r, y - r, x + r, y + r,
                                   fill="#505050", outline="")

        self._draw_aruco_corners()

        status = f"Point {active_idx + 1}/{routine.total_points}"
        if getattr(routine, "num_poses", 1) > 1:
            status = (f"Pose {routine.current_pose}/{routine.num_poses}  "
                      + status)
        if routine.is_collecting:
            status += (f" - collecting "
                       f"[{routine.collecting_sample_count}/{CALIB_SAMPLES}]")
        else:
            status += " - press 'c' to capture, 's' to skip, 'q' to quit"
        canvas.create_text(self.screen_width // 2, 40, text=status,
                           fill="white", anchor="n", font=("Courier", 20))

        marker_count = self.target_mapper.last_marker_count
        color = "#00ff00" if marker_count == 4 else "#ff6060"
        canvas.create_text(self.screen_width // 2, self.screen_height - 40,
                           text=f"aruco: {marker_count}/4 markers visible",
                           fill=color, anchor="s", font=("Courier", 16))

    def _render_pose_break(self, routine) -> None:
        """Between-pose screen: keep the ArUco corners up (so the user can
        check marker visibility while repositioning) and show the next-pose
        instruction."""
        canvas = self._canvas
        self._draw_aruco_corners()
        cx = self.screen_width // 2
        cy = self.screen_height // 2
        canvas.create_text(
            cx, cy - 40,
            text=f"Pose {routine.current_pose}/{routine.num_poses} done",
            fill="white", anchor="s", font=("Courier", 28))
        canvas.create_text(
            cx, cy,
            text=f"Next: {routine.next_pose_guidance()}",
            fill="#ffd000", anchor="center", font=("Courier", 22),
            width=int(self.screen_width * 0.7))
        canvas.create_text(
            cx, cy + 70,
            text="Reposition, then press 'c' to continue ('q' to quit)",
            fill="white", anchor="n", font=("Courier", 18))

        marker_count = self.target_mapper.last_marker_count
        color = "#00ff00" if marker_count == 4 else "#ff6060"
        canvas.create_text(
            cx, self.screen_height - 40,
            text=f"aruco: {marker_count}/4 markers visible",
            fill=color, anchor="s", font=("Courier", 16))

    # ---- internals ----

    def _on_key(self, ch: str) -> None:
        self._key_queue.append(ch)

    def _ensure_aruco_photo_images(self):
        if len(self._photo_images) == 4:
            return self._photo_images
        from scripts.eyetracker.config import ARUCO_IDS
        imgs = []
        for marker_id in ARUCO_IDS:
            png_bytes = generate_marker_png(marker_id, ARUCO_MARKER_PX)
            b64 = base64.b64encode(png_bytes)
            # Bind to this session's root explicitly. Without master=, Tk uses
            # the stale tkinter._default_root (a prior, destroyed root that
            # close() didn't clear), so on the 2nd calibration the image is
            # created in a dead interpreter -> "image pyimageN doesn't exist".
            imgs.append(tk.PhotoImage(master=self._root, data=b64))
        self._photo_images = imgs
        return imgs

    def _draw_aruco_corners(self) -> None:
        canvas = self._canvas
        if canvas is None:
            return
        photos = self._ensure_aruco_photo_images()
        origins = self.target_mapper.quiet_zone_origins()
        if not origins:
            return
        from scripts.eyetracker.config import ARUCO_IDS
        q = ARUCO_QUIET_ZONE_PX
        inset = (q - ARUCO_MARKER_PX) // 2
        for i, marker_id in enumerate(ARUCO_IDS):
            ox, oy = origins[marker_id]
            canvas.create_rectangle(ox, oy, ox + q, oy + q,
                                    fill="white", outline="")
            canvas.create_image(ox + inset, oy + inset, anchor="nw",
                                image=photos[i])
