"""Entry point: `python -m scripts.eyetracker`.

This is the only place that decides which concrete implementation of each
ABC to use. Swap a class here (e.g. PolynomialGazeMapper -> TpsGazeMapper)
and nothing else needs to change.
"""
import argparse

import cv2

from scripts.eyetracker.app import App
from scripts.eyetracker.calibration.collector import SampleCollector
from scripts.eyetracker.calibration.routine import CalibrationRoutine
from scripts.eyetracker.calibration.targets import GridPattern
from scripts.eyetracker.cameras.discovery import detect_cameras
from scripts.eyetracker.cameras.opencv_source import CameraSettings, OpenCVCamera
from scripts.eyetracker.config import (
    BUFFER_SIZE,
    CALIB_INLIERS,
    CALIB_SAMPLES,
    CALIB_SCENE_STD_THRESH,
    CALIB_STD_THRESH,
    CALIB_WARMUP,
    CONF_THRESH,
    EYE_CAM_FOCAL_LENGTH_PX,
    HIGH_FPS_MODE,
    PUPIL_BUFFER_SIZE,
    PUPIL_JUMP_THRESH,
    SCENE_REQUEST_HEIGHT,
    SCENE_REQUEST_WIDTH,
)
from scripts.eyetracker.display.cv_display import CvDisplay
from scripts.eyetracker.display.selection_gui import SelectionGui
from scripts.eyetracker.display.tk_overlay import TkCalibrationOverlay
from scripts.eyetracker.display.web_display import WebDisplay
from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper
from scripts.eyetracker.gaze.smoothing import MovingAverageSmoother
from scripts.eyetracker.pupil.gating import ConfidenceGate, JumpGate
from scripts.eyetracker.pupil.pupil_labs import PupilLabsDetector
from scripts.eyetracker.scene.aruco_homography import ArucoHomography


def _eye_cam_settings() -> CameraSettings:
    if HIGH_FPS_MODE:
        return CameraSettings(request_width=320, request_height=240,
                              request_fps=120, exposure=-5, flip_vertical=True)
    return CameraSettings(exposure=-5, flip_vertical=True)


def _scene_cam_settings() -> CameraSettings:
    return CameraSettings(request_width=SCENE_REQUEST_WIDTH,
                          request_height=SCENE_REQUEST_HEIGHT)


def _build_app(eye_index: int, web: bool = False) -> App:
    eye_cam = OpenCVCamera(eye_index, _eye_cam_settings())
    scene_index = 1 if eye_index == 0 else 0
    scene_cam = OpenCVCamera(scene_index, _scene_cam_settings())

    target_mapper = ArucoHomography()
    mapper = PolynomialGazeMapper()
    routine = CalibrationRoutine(
        pattern=GridPattern(rows=3, cols=4, margin=220),
        collector=SampleCollector(
            samples=CALIB_SAMPLES,
            inliers=CALIB_INLIERS,
            pupil_std_thresh=CALIB_STD_THRESH,
            scene_std_thresh=CALIB_SCENE_STD_THRESH,
            warmup=CALIB_WARMUP,
        ),
        target_mapper=target_mapper,
        mapper=mapper,
    )
    return App(
        eye_cam=eye_cam,
        scene_cam=scene_cam,
        pupil=PupilLabsDetector(focal_length_px=EYE_CAM_FOCAL_LENGTH_PX),
        conf_gate=ConfidenceGate(threshold=CONF_THRESH),
        jump_gate=JumpGate(threshold_px=PUPIL_JUMP_THRESH,
                           buffer_size=PUPIL_BUFFER_SIZE),
        mapper=mapper,
        smoother=MovingAverageSmoother(window=BUFFER_SIZE),
        target_mapper=target_mapper,
        routine=routine,
        overlay=TkCalibrationOverlay(target_mapper=target_mapper),
        display=WebDisplay() if web else CvDisplay(with_scene=True),
    )


def _run_video(path: str) -> None:
    """Standalone path for replaying a recorded eye-cam video.
    No scene cam, no calibration UI — just pupil detection."""
    pupil = PupilLabsDetector(focal_length_px=EYE_CAM_FOCAL_LENGTH_PX)
    conf_gate = ConfidenceGate(threshold=CONF_THRESH)
    jump_gate = JumpGate(threshold_px=PUPIL_JUMP_THRESH,
                         buffer_size=PUPIL_BUFFER_SIZE)
    display = CvDisplay(with_scene=False)
    display.open()

    from scripts.eyetracker.cameras.utils import crop_to_aspect_ratio
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_to_aspect_ratio(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sample = pupil.detect(gray)
            if sample is not None and conf_gate.accept(sample.confidence):
                accepted = jump_gate.accept(sample.center)
                if accepted is not None:
                    cv2.circle(frame, sample.center, 4, (0, 255, 0), -1)
                    if sample.ellipse is not None:
                        cv2.ellipse(frame, sample.ellipse, (20, 255, 255), 2)
            display.show_eye(frame)
            key, raw = display.poll_key()
            if key == 'q':
                break
            if raw == ord(' '):
                display.wait_for_pause()
    finally:
        cap.release()
        display.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="scripts.eyetracker")
    parser.add_argument(
        "--web", action="store_true",
        help="Stream annotated frames over HTTP (MJPEG) instead of "
             "opening cv2.imshow windows. See display/web_display.py.",
    )
    args = parser.parse_args()

    cameras = detect_cameras()
    result = SelectionGui().pick(cameras)
    if result is None:
        return
    kind, val = result
    if kind == "camera":
        _build_app(int(val), web=args.web).run()
    elif kind == "video":
        _run_video(str(val))


if __name__ == "__main__":
    main()
