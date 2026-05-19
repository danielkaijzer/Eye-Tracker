"""App — composition root and main run loop.

Construct an `App` with concrete instances of every replaceable axis
(camera sources, pupil detector, gates, gaze mapper + smoother, target
mapper, calibration routine, calibration overlay, runtime display) and
call `.run()`. The wiring of which concrete classes to use lives in
`__main__.py`, not here.
"""
import time
from typing import Optional

import cv2
import numpy as np

from scripts.eyetracker.calibration.persistence import (
    load_calibration as load_calibration_state,
)
from scripts.eyetracker.calibration.routine import CalibrationRoutine
from scripts.eyetracker.cameras.base import CameraSource
from scripts.eyetracker.cameras.utils import crop_to_aspect_ratio
from scripts.eyetracker.display.base import CalibrationOverlay, Display
from scripts.eyetracker.gaze.base import GazeMapper
from scripts.eyetracker.gaze.smoothing import OneEuroSmoother
from scripts.eyetracker.pupil.base import PupilDetector
from scripts.eyetracker.pupil.gating import ConfidenceGate, JumpGate
from scripts.eyetracker.scene.aruco_homography import ArucoHomography


_GATE_LOG_THROTTLE_S = 1.0


class App:
    def __init__(self,
                 *,
                 eye_cam: CameraSource,
                 scene_cam: Optional[CameraSource],
                 pupil: PupilDetector,
                 conf_gate: ConfidenceGate,
                 jump_gate: JumpGate,
                 mapper: GazeMapper,
                 smoother: OneEuroSmoother,
                 target_mapper: ArucoHomography,
                 routine: CalibrationRoutine,
                 overlay: CalibrationOverlay,
                 display: Display):
        self.eye_cam = eye_cam
        self.scene_cam = scene_cam
        self.pupil = pupil
        self.conf_gate = conf_gate
        self.jump_gate = jump_gate
        self.mapper = mapper
        self.smoother = smoother
        self.target_mapper = target_mapper
        self.routine = routine
        self.overlay = overlay
        self.display = display

        # Per-frame mutable state, all owned by the App instance.
        self.last_pupil_center: Optional[np.ndarray] = None
        self.last_confidence: float = 0.0
        self.last_eye_frame: Optional[np.ndarray] = None
        self.last_scene_frame: Optional[np.ndarray] = None
        self._last_gate_log_ts: float = 0.0

    # ---- entry point --------------------------------------------------------

    def run(self) -> None:
        if not self.eye_cam.open():
            print(f"Error: could not open eye camera ({self.eye_cam}).")
            return
        if self.scene_cam is not None and not self.scene_cam.open():
            self.scene_cam = None
        if self.scene_cam is not None:
            print(f"Scene cam: {self.scene_cam.width}x{self.scene_cam.height}")
            self.routine.scene_size = (self.scene_cam.width, self.scene_cam.height)

        self.routine.jump_gate = self.jump_gate
        self.display.open()

        print("Controls: 'c' = calibrate, 'l' = load calibration, "
              "'r' = reset pupil 3D model, 'q' = quit, space = pause")

        try:
            self._loop()
        finally:
            self.overlay.close()
            self.eye_cam.release()
            if self.scene_cam is not None:
                self.scene_cam.release()
            self.display.close()

    def _loop(self) -> None:
        while True:
            eye_frame = self.eye_cam.read()
            if eye_frame is None:
                break

            self._process_eye_frame(eye_frame)

            was_calibration_active = self.routine.is_active
            if self.routine.is_collecting:
                self.routine.tick(
                    pupil_center=self.last_pupil_center,
                    eye_frame=self.last_eye_frame,
                    scene_frame=self.last_scene_frame,
                    confidence=self.last_confidence,
                )
            if was_calibration_active and not self.routine.is_active:
                self.overlay.close()
                # Reset the gaze smoother so the new fit starts clean.
                self.smoother.reset()

            if self.scene_cam is not None:
                ext_frame = self.scene_cam.read()
                if ext_frame is not None:
                    self.last_scene_frame = ext_frame.copy()
                    self.target_mapper.update_marker_count(self.last_scene_frame)

                    gaze_xy = None
                    if self.mapper.is_fitted() and not self.routine.is_active:
                        gaze_xy = self._predict_gaze_scene_xy()
                    self.display.show_scene(ext_frame, gaze_xy)

            if self.routine.is_active and self.overlay.is_open():
                self.overlay.render(self.routine)
                self.overlay.pump()

            tk_key = self.overlay.poll_key()
            cv_key, raw = self.display.poll_key()
            key = tk_key or cv_key
            if not self._handle_key(key, raw):
                break

    # ---- per-stage helpers --------------------------------------------------

    def _process_eye_frame(self, frame: np.ndarray) -> None:
        """Crop, run pupil detector, gate, draw, send to display."""
        frame = crop_to_aspect_ratio(frame)
        self.last_eye_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sample = self.pupil.detect(gray)
        confidence = sample.confidence if sample is not None else 0.0
        self.last_confidence = confidence

        accepted = None
        reject_reason = None
        if sample is None:
            reject_reason = "no detection"
        elif not self.conf_gate.accept(confidence):
            reject_reason = self.conf_gate.describe_reject(confidence)
        else:
            accepted = self.jump_gate.accept(sample.center)
            if accepted is None:
                reject_reason = self.jump_gate.describe_reject(
                    sample.center, confidence)

        if accepted is not None:
            self.last_pupil_center = accepted
            cv2.circle(frame, sample.center, 4, (0, 255, 0), -1)
            if sample.ellipse is not None:
                cv2.ellipse(frame, sample.ellipse, (20, 255, 255), 2)
            text = (f"Pupil: ({sample.center[0]}, {sample.center[1]})  "
                    f"conf={confidence:.2f}")
            cv2.putText(frame, text, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            self.last_pupil_center = None
            cv2.circle(frame, (15, 15), 8, (0, 0, 255), -1)
            cv2.putText(frame, "SKIP", (28, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if reject_reason is not None and reject_reason != "no detection":
                now = time.time()
                if now - self._last_gate_log_ts >= _GATE_LOG_THROTTLE_S:
                    print(f"[gate] reject: {reject_reason}")
                    self._last_gate_log_ts = now

        self.display.show_eye(frame)

    def _predict_gaze_scene_xy(self) -> Optional[tuple]:
        """Return (x, y) in scene-cam pixels, smoothed and clipped, or None
        if no current pupil sample."""
        if self.last_pupil_center is None or self.scene_cam is None:
            return None
        pred = self.mapper.predict(
            (float(self.last_pupil_center[0]), float(self.last_pupil_center[1]))
        )
        avg_u, avg_v = self.smoother.add(pred)
        sw = self.scene_cam.width
        sh = self.scene_cam.height
        return (int(np.clip(avg_u, 0, sw - 1)),
                int(np.clip(avg_v, 0, sh - 1)))

    # ---- key dispatch -------------------------------------------------------

    def _handle_key(self, key: Optional[str], raw: int) -> bool:
        """Returns True to continue the loop, False to quit."""
        if key == 'q':
            return False
        if raw == ord(' '):
            self.display.wait_for_pause()
        elif key == 'l':
            self._handle_load()
        elif key == 'c':
            if not self.routine.is_active:
                self._start_calibration()
            elif not self.routine.is_collecting:
                self.routine.begin_capture()
        elif key == 's':
            if self.routine.is_active:
                self.routine.skip()
        elif key == 'r':
            self.pupil.reset()
            print("Pupil 3D model reset — give it ~30s to reconverge.")
        return True

    def _handle_load(self) -> None:
        load_calibration_state(self.mapper)
        self.smoother.reset()

    def _start_calibration(self) -> None:
        sw, sh = self.overlay.open()
        self.target_mapper.set_screen_size(sw, sh)
        if self.scene_cam is not None:
            self.routine.scene_size = (self.scene_cam.width, self.scene_cam.height)
        self.routine.start(sw, sh)
