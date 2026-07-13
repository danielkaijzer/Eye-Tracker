"""Tk-based fullscreen calibration overlay.

Owns the Tk root + canvas, an internal key queue (populated by Tk key bindings),
and a cache of `tk.PhotoImage` instances for the four ArUco corner markers.
Reads from a CalibrationRoutine to repaint each frame; never mutates routine
state directly.
"""
import base64
import tkinter as tk
from typing import List, Optional, Tuple

import numpy as np

from scripts.eyetracker.config import (
    ARUCO_MARKER_PX,
    ARUCO_QUIET_ZONE_PX,
    ARUCO_WHITE_LEVEL,
    ARUCO_WHITE_MIN,
    ARUCO_WHITE_STEP,
    CALIB_SAMPLES,
)
from scripts.eyetracker.display.base import CalibrationOverlay
from scripts.eyetracker.scene.aruco_dict import generate_marker_png
from scripts.eyetracker.scene.aruco_homography import ArucoHomography


# Human-readable corner names, index-matched to config.ARUCO_IDS (0=TL, 1=TR,
# 2=BR, 3=BL) — used to say WHICH marker is missing in the bottom HUD.
_CORNER_NAMES = ("TL", "TR", "BR", "BL")


class TkCalibrationOverlay(CalibrationOverlay):
    def __init__(self, target_mapper: ArucoHomography):
        self.target_mapper = target_mapper
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._key_queue: List[str] = []
        self._photo_images: list = []
        self.screen_width = 0
        self.screen_height = 0
        # Per-frame state stamped by the App before render() (the App owns
        # the camera + pupil pipeline; the overlay just draws).
        self.exposure_status = ""
        self.pupil_ok = True
        # Displayed white level of markers + quiet zones; '-'/'=' nudge it
        # when full white blooms on the scene cam. Session-only (resets to
        # the config default on restart).
        self.marker_white_level = ARUCO_WHITE_LEVEL
        # Camera preview ('p'): frames stamped by the App while enabled.
        # _preview_photos holds this frame's PhotoImages — Tk only keeps a
        # weak handle, so dropping the Python reference blanks the canvas.
        self.preview_enabled = False
        self.preview_scene: Optional[np.ndarray] = None
        self.preview_eye: Optional[np.ndarray] = None
        self._preview_photos: list = []

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

        # The fullscreen overlay owns keyboard focus during calibration, so
        # every key the App should react to must be bound here or it is
        # silently swallowed. Brackets/braces are the scene-exposure nudges
        # (App._handle_key); Escape aborts the in-progress fixation.
        for keysym, ch in (("c", "c"), ("s", "s"), ("q", "q"),
                           ("bracketleft", "["), ("bracketright", "]"),
                           ("braceleft", "{"), ("braceright", "}"),
                           ("minus", "-"), ("equal", "="), ("p", "p")):
            root.bind(f"<KeyPress-{keysym}>",
                      lambda _e, _ch=ch: self._on_key(_ch))
        root.bind("<KeyPress-Escape>", lambda _e: self._on_key("esc"))
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
        self._preview_photos = []
        self.preview_enabled = False
        self.preview_scene = None
        self.preview_eye = None

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
                self._draw_active_indicators(x, y, routine)
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
                       f"[{routine.collecting_sample_count}/{CALIB_SAMPLES}] "
                       "(Esc = abort)")
        else:
            status += (" - press 'c' to capture, 's' to skip, "
                       "'p' = preview, 'q' to quit")
        canvas.create_text(self.screen_width // 2, 40, text=status,
                           fill="white", anchor="n", font=("Courier", 20))
        if routine.notice:
            canvas.create_text(self.screen_width // 2, 80, text=routine.notice,
                               fill="#ffb000", anchor="n", font=("Courier", 16))

        self._draw_bottom_hud()
        self._draw_preview()

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

        self._draw_bottom_hud()
        self._draw_preview()

    def _draw_preview(self) -> None:
        """Center-screen camera previews ('p' hotkey): the App stamps
        annotated, pre-downscaled BGR frames; this just PPM-encodes and
        draws them side by side. Deliberately covers the middle of the
        screen — it's a between-attempts troubleshooting view, not something
        to leave up while capturing."""
        if not self.preview_enabled:
            return
        panels = [(label, frame) for label, frame in
                  (("scene (red = clipped)", self.preview_scene),
                   ("eye", self.preview_eye))
                  if frame is not None]
        if not panels:
            return
        canvas = self._canvas
        self._preview_photos = []
        gap = 24
        pad = 10
        total_w = (sum(frame.shape[1] for _, frame in panels)
                   + gap * (len(panels) - 1))
        x = (self.screen_width - total_w) // 2
        center_y = self.screen_height // 2
        for label, frame in panels:
            h, w = frame.shape[:2]
            top = center_y - h // 2
            photo = self._photo_from_bgr(frame)
            self._preview_photos.append(photo)
            canvas.create_rectangle(x - pad, top - pad - 20,
                                    x + w + pad, top + h + pad,
                                    fill="#181818", outline="#606060")
            canvas.create_text(x, top - 4, text=label,
                               fill="#c0c0c0", anchor="sw",
                               font=("Courier", 13))
            canvas.create_image(x, top, anchor="nw", image=photo)
            x += w + gap

    def _photo_from_bgr(self, frame_bgr: np.ndarray) -> tk.PhotoImage:
        """BGR array -> tk.PhotoImage via raw PPM bytes (P6) — no PNG
        compression, cheap enough to do per frame."""
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        header = f"P6 {w} {h} 255 ".encode()
        return tk.PhotoImage(master=self._root, data=header + rgb.tobytes(),
                             format="PPM")

    def _draw_active_indicators(self, x: int, y: int, routine) -> None:
        """Peripheral status around the active dot — quiet when healthy, loud
        when something is wrong — so the user can hold fixation and still see
        why capture isn't progressing (peripheral vision resolves the
        appearance of a thick ring far better than a color change).

        - collecting: dark track ring + orange progress arc (samples/target),
          clockwise from 12 o'clock
        - pupil not detected this frame: solid red inner ring
        - ArUco marker missing: red arc on the outer ring, in the screen
          quadrant of that marker (TL/TR/BR/BL) — points at the problem corner
        """
        canvas = self._canvas
        if routine.is_collecting:
            track_r = 28
            canvas.create_oval(x - track_r, y - track_r,
                               x + track_r, y + track_r,
                               outline="#404040", width=4)
            frac = min(routine.collecting_sample_count / CALIB_SAMPLES, 1.0)
            if frac > 0:
                # Negative extent sweeps clockwise; 359.9 because Tk treats
                # a full ±360 arc as extent 0 and draws nothing.
                canvas.create_arc(x - track_r, y - track_r,
                                  x + track_r, y + track_r,
                                  start=90, extent=-359.9 * frac,
                                  style=tk.ARC, outline="#ffa500", width=4)

        if not self.pupil_ok:
            pupil_r = 38
            canvas.create_oval(x - pupil_r, y - pupil_r,
                               x + pupil_r, y + pupil_r,
                               outline="#ff2020", width=5)

        from scripts.eyetracker.config import ARUCO_IDS
        found = self.target_mapper.last_found_ids
        marker_r = 48
        # Tk arc angles: 0 = 3 o'clock, counterclockwise. Quadrant start
        # angles index-matched to ARUCO_IDS order (TL, TR, BR, BL).
        quadrant_starts = (90, 0, 270, 180)
        for marker_id, start in zip(ARUCO_IDS, quadrant_starts):
            if marker_id not in found:
                canvas.create_arc(x - marker_r, y - marker_r,
                                  x + marker_r, y + marker_r,
                                  start=start, extent=90,
                                  style=tk.ARC, outline="#ff2020", width=5)

    def _draw_bottom_hud(self) -> None:
        """Bottom-of-screen diagnostics, stacked upward from the bottom:
        marker visibility (naming any missing corner), a pupil-lost warning,
        and the scene-exposure state with its nudge keys."""
        canvas = self._canvas
        center_x = self.screen_width // 2
        marker_count = self.target_mapper.last_marker_count
        text = f"aruco: {marker_count}/4 markers visible"
        if marker_count < 4:
            from scripts.eyetracker.config import ARUCO_IDS
            found = self.target_mapper.last_found_ids
            missing = [name for marker_id, name in zip(ARUCO_IDS, _CORNER_NAMES)
                       if marker_id not in found]
            text += f" - missing: {', '.join(missing)}"
        color = "#00ff00" if marker_count == 4 else "#ff6060"
        canvas.create_text(center_x, self.screen_height - 40,
                           text=text,
                           fill=color, anchor="s", font=("Courier", 16))
        if not self.pupil_ok:
            canvas.create_text(center_x, self.screen_height - 66,
                               text="pupil: not detected",
                               fill="#ff6060", anchor="s", font=("Courier", 16))
        if self.exposure_status:
            canvas.create_text(
                center_x, self.screen_height - 92,
                text=f"{self.exposure_status}  ('['/']' fine, '{{'/'}}' coarse)",
                fill="#b0b0b0", anchor="s", font=("Courier", 14))
        canvas.create_text(
            center_x, self.screen_height - 116,
            text=(f"marker white: {self.marker_white_level}/255  "
                  "('-'/'=' adjust)"),
            fill="#b0b0b0", anchor="s", font=("Courier", 14))

    # ---- internals ----

    def _on_key(self, ch: str) -> None:
        self._key_queue.append(ch)

    def toggle_preview(self) -> None:
        """Show/hide the center-screen camera previews ('p' hotkey)."""
        self.preview_enabled = not self.preview_enabled
        if not self.preview_enabled:
            self.preview_scene = None
            self.preview_eye = None
            self._preview_photos = []

    def nudge_marker_white(self, direction: int) -> None:
        """Adjust the displayed white level of the markers + quiet zones by
        one ARUCO_WHITE_STEP ('-'/'=' hotkeys). Invalidates the marker
        PhotoImage cache so the next draw rebuilds at the new level."""
        new_level = self.marker_white_level + direction * ARUCO_WHITE_STEP
        new_level = max(ARUCO_WHITE_MIN, min(255, new_level))
        if new_level == self.marker_white_level:
            return
        self.marker_white_level = new_level
        self._photo_images = []
        print(f"[markers] white level = {self.marker_white_level}/255")

    def _ensure_aruco_photo_images(self):
        if len(self._photo_images) == 4:
            return self._photo_images
        from scripts.eyetracker.config import ARUCO_IDS
        imgs = []
        for marker_id in ARUCO_IDS:
            png_bytes = generate_marker_png(marker_id, ARUCO_MARKER_PX,
                                            self.marker_white_level)
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
        # Quiet zone must be the SAME grey as the marker's white cells — a
        # dimmed marker inside a full-white zone kills detection contrast.
        level = self.marker_white_level
        quiet_zone_fill = f"#{level:02x}{level:02x}{level:02x}"
        for i, marker_id in enumerate(ARUCO_IDS):
            ox, oy = origins[marker_id]
            canvas.create_rectangle(ox, oy, ox + q, oy + q,
                                    fill=quiet_zone_fill, outline="")
            canvas.create_image(ox + inset, oy + inset, anchor="nw",
                                image=photos[i])
