"""Display + CalibrationOverlay interfaces.

A Display owns the runtime windows (eye + scene) and the keyboard input
loop. A CalibrationOverlay owns the fullscreen target/marker UI shown only
during calibration. Both are pluggable so the cv2/Tk pair can be replaced
later (e.g. with a web display) without changing the App loop.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


XY = Tuple[int, int]


class Display(ABC):
    @abstractmethod
    def open(self) -> None:
        """Set up windows."""

    @abstractmethod
    def close(self) -> None:
        """Tear down windows."""

    @abstractmethod
    def show_eye(self, frame: np.ndarray) -> None:
        """Display the (already-annotated) eye-camera frame."""

    @abstractmethod
    def show_scene(self, frame: np.ndarray, gaze_xy: Optional[XY]) -> None:
        """Display the scene-camera frame, optionally with a gaze dot
        in display coordinates (caller already scaled from scene-cam coords)."""

    @abstractmethod
    def poll_key(self) -> Tuple[Optional[str], int]:
        """Return (key_char or None, raw_key_code). raw_key_code is exposed
        so the App can detect special keys like SPACE without ambiguity."""

    @abstractmethod
    def wait_for_pause(self) -> None:
        """Block until any key is pressed (used for the spacebar pause/resume)."""


class CalibrationOverlay(ABC):
    # Per-frame state stamped by the App before render(); implementations may
    # ignore them. exposure_status is a one-line scene-exposure summary;
    # pupil_ok is whether the pupil pipeline produced an accepted center this
    # frame (post confidence + jump gates).
    exposure_status: str = ""
    pupil_ok: bool = True
    # In-overlay camera preview ('p' hotkey). preview_enabled is owned by the
    # overlay (toggle_preview); while it is True the App stamps the two frames
    # (BGR arrays, already annotated + downscaled) before each render().
    preview_enabled: bool = False
    preview_scene: Optional[np.ndarray] = None
    preview_eye: Optional[np.ndarray] = None

    @abstractmethod
    def open(self) -> Tuple[int, int]:
        """Open the fullscreen overlay; return (screen_width, screen_height)."""

    @abstractmethod
    def is_open(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def render(self, routine) -> None:
        """Repaint the overlay from the routine's current state."""

    @abstractmethod
    def pump(self) -> None:
        """Tick the overlay's event loop (e.g. tk root.update())."""

    @abstractmethod
    def poll_key(self) -> Optional[str]:
        """Return the next queued key char, or None."""

    def nudge_marker_white(self, direction: int) -> None:
        """Optional: adjust the displayed ArUco white level by one step
        (+1 brighter / -1 darker). Default: no-op."""

    def toggle_preview(self) -> None:
        """Optional: toggle the in-overlay camera preview. Default: no-op."""
