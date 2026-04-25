"""Pupil-only eye tracker.

Pipeline:
- Pupil Labs `pupil_detectors.Detector2D` (dark-pupil, 2014 paper) finds the
  pupil center in the eye-cam frame, then `pye3d.Detector3D` refines that
  center using a persistent 3D eye-sphere model. First ~30-60s of use build
  the model; after that the 2D pupil output is temporally smoothed and
  robust to blinks/eyelid occlusion. Press 'r' to reset the model if the
  headset is re-seated mid-session.
- A 2nd-degree bivariate polynomial maps pupil pixel directly to scene-cam
  pixel. No screen-pixel rescale at inference time — the mapping is
  head-pose invariant for the gaze-on-scene use case.
- Calibration shows 12 red dots fullscreen on the monitor. ArUco markers
  at the four screen corners give a screen<->scene-cam homography
  (`detect_aruco_homography`); each fixation's training label is the known
  on-screen dot position projected through that homography into scene-cam
  pixels. Training in scene-cam space is what makes the fit head-invariant.
- `screen_pixel_for_scene_gaze` is stubbed for future inference-time
  screen-pixel mapping.

A hand-rolled reference port lives in `eyetracker_pupil.py`, along with
`docs/pupil_detector_port_notes.md` — that file is for future port work
and is not imported from here.

"""

# TODO: Break up the logic of this code into separate scripts. Move towards a modular, plugin architecture.

# TODO: the 2nd-degree polynomial extrapolates poorly outside the
# calibration grid — corner gaze (especially bottom-right) drifts. Future
# options:
#   1. Add an outer 4th row/column of calibration points at a wider margin
#      so the corners are interpolated, not extrapolated. Current grid is
#      3x4 with margin 120.
#   2. Replace the polynomial with a thin-plate spline (TPS): interpolates
#      exactly at training points and behaves more smoothly at edges.
#   3. Weight edge/corner points higher in the least-squares fit.
import base64
import cv2
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import time
from collections import deque

from scripts.eyetracker.config import (
    ARUCO_DICT_NAME,
    ARUCO_IDS,
    ARUCO_MARKER_PX,
    ARUCO_QUIET_ZONE_PX,
    BUFFER_SIZE,
    CALIB_INLIERS,
    CALIB_SAMPLES,
    CALIB_SCENE_STD_THRESH,
    CALIB_STD_THRESH,
    CALIB_WARMUP,
    CONF_THRESH,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    EYE_CAM_FOCAL_LENGTH_PX,
    EYE_CAM_FOV_DEG,
    EYE_CAM_FOV_IS_DIAGONAL,
    EYE_CAM_RESOLUTION,
    HIGH_FPS_MODE,
    PUPIL_BUFFER_SIZE,
    PUPIL_JUMP_THRESH,
    SCENE_REQUEST_HEIGHT,
    SCENE_REQUEST_WIDTH,
)
from scripts.eyetracker.cameras.discovery import detect_cameras
from scripts.eyetracker.cameras.opencv_source import CameraSettings, OpenCVCamera
from scripts.eyetracker.cameras.utils import crop_to_aspect_ratio
from scripts.eyetracker.pupil.gating import ConfidenceGate, JumpGate
from scripts.eyetracker.pupil.pupil_labs import PupilLabsDetector
from scripts.eyetracker.calibration.persistence import (
    CalibrationSnapshot,
    append_label_rows,
    begin_session,
    load_calibration as _load_calibration,
    save_calibration as _save_calibration_to_disk,
)
from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper
from scripts.eyetracker.gaze.smoothing import MovingAverageSmoother
from scripts.eyetracker.scene.aruco_dict import generate_marker_png
from scripts.eyetracker.scene.aruco_homography import ArucoHomography

_gaze_mapper = PolynomialGazeMapper()
_gaze_smoother = MovingAverageSmoother(window=BUFFER_SIZE)

calib_points_screen = []
calib_vectors_eye = []
calib_skipped_indices = []
calib_state = 0
calib_total_points = 12

calib_collecting = False
calib_collect_frames = []
calib_collect_scene = []
calib_points_scene = []
_calib_warmup_remaining = 0

last_pupil_center = None
last_confidence = 0.0

_pupil_detector: PupilLabsDetector | None = None
_jump_gate: JumpGate | None = None
_conf_gate: ConfidenceGate | None = None

calibrated = False

scene_cam_width = None
scene_cam_height = None

circle_x = 0
circle_y = 0

screen_width = None
screen_height = None

calib_session_dir = None
calib_labels_path = None
calib_pending_rows = []
calib_pending_image_paths = []

calib_tk_root = None
calib_tk_canvas = None
_tk_key_queue = []

last_eye_frame = None
last_scene_frame = None
_last_no_scene_log_ts = 0.0
_last_aruco_log_ts = 0.0

_aruco_photo_images = []
_aruco_mapper = ArucoHomography()


def _ensure_aruco_photo_images():
    """Generate and cache the four marker images as tk.PhotoImage instances.
    Must be called after a tk root exists. Stays here (not in scene/) because
    tk.PhotoImage requires a live Tk root; will move into display/ at step 9."""
    global _aruco_photo_images
    if len(_aruco_photo_images) == 4:
        return _aruco_photo_images
    imgs = []
    for marker_id in ARUCO_IDS:
        png_bytes = generate_marker_png(marker_id, ARUCO_MARKER_PX)
        b64 = base64.b64encode(png_bytes)
        imgs.append(tk.PhotoImage(data=b64))
    _aruco_photo_images = imgs
    return _aruco_photo_images


def _draw_aruco_corners(canvas):
    if screen_width is None or screen_height is None:
        return
    photos = _ensure_aruco_photo_images()
    origins = _aruco_mapper.quiet_zone_origins()
    q = ARUCO_QUIET_ZONE_PX
    inset = (q - ARUCO_MARKER_PX) // 2
    for i, marker_id in enumerate(ARUCO_IDS):
        ox, oy = origins[marker_id]
        canvas.create_rectangle(ox, oy, ox + q, oy + q, fill="white", outline="")
        canvas.create_image(ox + inset, oy + inset, anchor="nw", image=photos[i])


def reset_pye3d_model():
    """Discard the 3D eye model so the next detect call re-initializes it.
    Use after re-seating the headset."""
    if _pupil_detector is not None:
        _pupil_detector.reset()
    print("Pupil 3D model reset — give it ~30s to reconverge.")


_last_gate_log_ts = 0.0


def process_frame(frame):
    """Detect pupil, gate on confidence + outlier rejection, update last_pupil_center."""
    global last_pupil_center, last_confidence, _last_gate_log_ts, last_eye_frame

    frame = crop_to_aspect_ratio(frame)
    last_eye_frame = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sample = _pupil_detector.detect(gray_frame)
    confidence = sample.confidence if sample is not None else 0.0
    last_confidence = confidence

    accepted = None
    reject_reason = None
    if sample is None:
        reject_reason = "no detection"
    elif not _conf_gate.accept(confidence):
        reject_reason = _conf_gate.describe_reject(confidence)
    else:
        accepted = _jump_gate.accept(sample.center)
        if accepted is None:
            reject_reason = _jump_gate.describe_reject(sample.center, confidence)

    if accepted is not None:
        last_pupil_center = accepted
        cv2.circle(frame, sample.center, 4, (0, 255, 0), -1)
        if sample.ellipse is not None:
            cv2.ellipse(frame, sample.ellipse, (20, 255, 255), 2)
        text = f"Pupil: ({sample.center[0]}, {sample.center[1]})  conf={confidence:.2f}"
        cv2.putText(frame, text, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        last_pupil_center = None
        cv2.circle(frame, (15, 15), 8, (0, 0, 255), -1)
        cv2.putText(frame, "SKIP", (28, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if reject_reason is not None and reject_reason != "no detection":
            now = time.time()
            if now - _last_gate_log_ts >= 1.0:
                print(f"[gate] reject: {reject_reason}")
                _last_gate_log_ts = now

    cv2.imshow("Eye Camera", frame)


def update_gaze_circle_from_current_gaze():
    """Map current pupil position to scene-camera coords via fitted polynomial.

    Polynomial outputs gaze directly in native scene-cam pixels; no rescale.
    Drawing scales from scene-cam space to display space separately.
    """
    global circle_x, circle_y
    if not calibrated or last_pupil_center is None:
        return
    if not _gaze_mapper.is_fitted():
        return
    if scene_cam_width is None or scene_cam_height is None:
        return

    pred = _gaze_mapper.predict((last_pupil_center[0], last_pupil_center[1]))
    avg_u, avg_v = _gaze_smoother.add(pred)

    circle_x = int(np.clip(avg_u, 0, scene_cam_width - 1))
    circle_y = int(np.clip(avg_v, 0, scene_cam_height - 1))


def compute_polynomial_calibration():
    """Fit 2nd-degree polynomial from pupil positions to scene-cam coords."""
    n = len(calib_vectors_eye)
    if len(calib_points_scene) != n:
        print(f"  ERROR: calib_points_scene ({len(calib_points_scene)}) and "
              f"calib_vectors_eye ({n}) length mismatch. Aborting fit.")
        return False

    try:
        report = _gaze_mapper.fit(np.array(calib_vectors_eye),
                                  np.array(calib_points_scene))
    except ValueError as e:
        print(str(e))
        return False

    skipped = len(calib_skipped_indices)
    print(f"Polynomial calibration fitted ({report.n_points} points, {skipped} skipped).")
    print(f"  LOO error: avg={report.loo_avg_err:.1f}px, max={report.loo_max_err:.1f}px")
    err_threshold = 0.04 * (scene_cam_width or 640)
    if report.loo_avg_err > err_threshold:
        print(f"  WARNING: High error (>{err_threshold:.0f}px at this scene-cam resolution) — consider recalibrating.")

    _save_calibration()
    return True


def _save_calibration():
    aruco_centers_map = _aruco_mapper.screen_anchor_points()
    aruco_screen_centers = np.array(
        [aruco_centers_map[i] for i in ARUCO_IDS], dtype=float
    ) if aruco_centers_map else np.zeros((0, 2), dtype=float)
    snapshot = CalibrationSnapshot(
        pupil_vectors=np.array(calib_vectors_eye),
        scene_points=np.array(calib_points_scene),
        screen_points=np.array(calib_points_screen),
        aruco_screen_centers=aruco_screen_centers,
        scene_size=(scene_cam_width, scene_cam_height)
            if scene_cam_width is not None and scene_cam_height is not None else None,
        screen_size=(screen_width, screen_height)
            if screen_width is not None and screen_height is not None else None,
    )
    _save_calibration_to_disk(snapshot, _gaze_mapper)


def load_calibration():
    global calibrated
    global screen_width, screen_height
    global scene_cam_width, scene_cam_height
    result = _load_calibration(_gaze_mapper)
    if result is None:
        return False
    if result.screen_size is not None:
        screen_width, screen_height = result.screen_size
    if result.scene_size is not None:
        scene_cam_width, scene_cam_height = result.scene_size
    calibrated = True
    return True


def _on_tk_key(ch):
    _tk_key_queue.append(ch)


def start_calibration():
    global calib_state, calib_points_screen, calib_vectors_eye
    global calib_skipped_indices
    global calib_collecting, calib_collect_frames, calib_total_points
    global calib_collect_scene, calib_points_scene
    global screen_width, screen_height
    global calib_session_dir, calib_labels_path
    global calib_pending_rows, calib_pending_image_paths
    global calib_tk_root, calib_tk_canvas, _tk_key_queue

    global _aruco_photo_images

    calib_state = 1
    if _jump_gate is not None:
        _jump_gate.reset()
    calib_vectors_eye = []
    calib_skipped_indices = []
    calib_collecting = False
    calib_collect_frames = []
    calib_collect_scene = []
    calib_points_scene = []
    calib_pending_rows = []
    calib_pending_image_paths = []
    _tk_key_queue = []
    _aruco_photo_images = []

    calib_tk_root = tk.Tk()
    calib_tk_root.configure(bg="black")
    calib_tk_root.attributes("-fullscreen", True)
    calib_tk_root.attributes("-topmost", True)
    calib_tk_root.config(cursor="none")
    calib_tk_root.update_idletasks()
    screen_width = calib_tk_root.winfo_width()
    screen_height = calib_tk_root.winfo_height()
    _aruco_mapper.set_screen_size(screen_width, screen_height)

    calib_tk_canvas = tk.Canvas(
        calib_tk_root,
        width=screen_width, height=screen_height,
        bg="black", highlightthickness=0,
    )
    calib_tk_canvas.pack(fill="both", expand=True)

    calib_tk_root.bind("<KeyPress-c>", lambda _: _on_tk_key("c"))
    calib_tk_root.bind("<KeyPress-s>", lambda _: _on_tk_key("s"))
    calib_tk_root.bind("<KeyPress-q>", lambda _: _on_tk_key("q"))
    calib_tk_root.focus_force()
    calib_tk_root.update()

    margin_x = 220
    margin_y = 220
    cols, rows = 4, 3
    calib_points_screen = []
    for r in range(rows):
        for col in range(cols):
            x = int(margin_x + col * (screen_width - 2 * margin_x) / (cols - 1))
            y = int(margin_y + r * (screen_height - 2 * margin_y) / (rows - 1))
            calib_points_screen.append((x, y))
    calib_total_points = len(calib_points_screen)

    calib_session_dir, calib_labels_path = begin_session()

    print(f"Calibration started ({calib_total_points} points) on {screen_width}x{screen_height} screen.")
    print(f"  Dataset: {calib_session_dir}")
    print("  Look at the RED dot and press 'c'.")


def render_calibration_overlay():
    """Fullscreen tk overlay with calibration targets in screen-pixel coords."""
    if calib_tk_canvas is None:
        return
    calib_tk_canvas.delete("all")

    active_idx = calib_state - 1
    for i, pt in enumerate(calib_points_screen):
        x, y = pt
        if i in calib_skipped_indices:
            r = 8
            calib_tk_canvas.create_line(x - r, y - r, x + r, y + r,
                                        fill="#808080", width=3)
            calib_tk_canvas.create_line(x - r, y + r, x + r, y - r,
                                        fill="#808080", width=3)
        elif i == active_idx:
            r = 20
            calib_tk_canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="red", outline="")
            if calib_collecting:
                rr = 30
                calib_tk_canvas.create_oval(x - rr, y - rr, x + rr, y + rr,
                                            outline="#ffa500", width=3)
        elif i < active_idx:
            r = 8
            calib_tk_canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="#00b400", outline="")
        else:
            r = 6
            calib_tk_canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="#505050", outline="")

    _draw_aruco_corners(calib_tk_canvas)

    status = f"Point {calib_state}/{calib_total_points}"
    if calib_collecting:
        status += f" - collecting [{len(calib_collect_frames)}/{CALIB_SAMPLES}]"
    else:
        status += " - press 'c' to capture, 's' to skip, 'q' to quit"
    calib_tk_canvas.create_text(screen_width // 2, 40, text=status, fill="white",
                                anchor="n", font=("Courier", 20))

    marker_count = _aruco_mapper.last_marker_count
    aruco_color = "#00ff00" if marker_count == 4 else "#ff6060"
    calib_tk_canvas.create_text(screen_width // 2, screen_height - 40,
                                text=f"aruco: {marker_count}/4 markers visible",
                                fill=aruco_color, anchor="s",
                                font=("Courier", 16))


def begin_capture():
    global calib_collecting, calib_collect_frames, calib_collect_scene
    global _calib_warmup_remaining
    calib_collecting = True
    calib_collect_frames = []
    calib_collect_scene = []
    _calib_warmup_remaining = CALIB_WARMUP


def skip_current_point():
    global calib_collecting, calib_collect_frames, calib_collect_scene
    global calib_state, calibrated
    global calib_pending_rows, calib_pending_image_paths

    if calib_state <= 0 or calib_state > calib_total_points:
        return

    fixation_idx = calib_state - 1
    if calib_collecting:
        for p in calib_pending_image_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        calib_pending_rows = []
        calib_pending_image_paths = []
        calib_collecting = False
        calib_collect_frames = []
        calib_collect_scene = []

    calib_skipped_indices.append(fixation_idx)
    print(f"Skipped point {calib_state}/{calib_total_points}.")

    if calib_state >= calib_total_points:
        n = len(calib_vectors_eye)
        skipped = len(calib_skipped_indices)
        if n < 6:
            print(f"Not enough non-skipped points to fit "
                  f"({n} captured, {skipped} skipped). Need at least 6.")
        elif compute_polynomial_calibration():
            calibrated = True
            print("Calibration Complete!")
        calib_state = 0
        _teardown_calibration_overlay()
    else:
        calib_state += 1


def tick_capture():
    """Collect samples for the current target. Returns True when one point is done."""
    global calib_collecting, calib_collect_frames, calib_collect_scene
    global calib_state, calibrated
    global calib_pending_rows, calib_pending_image_paths
    global _calib_warmup_remaining, _last_no_scene_log_ts, _last_aruco_log_ts
    if not calib_collecting:
        return False
    if last_pupil_center is None or last_eye_frame is None:
        return False
    if _calib_warmup_remaining > 0:
        _calib_warmup_remaining -= 1
        return False

    if last_scene_frame is None:
        now = time.time()
        if now - _last_no_scene_log_ts >= 1.0:
            print("  No scene camera — cannot calibrate in scene-cam mode.")
            _last_no_scene_log_ts = now
        return False

    H, _ = _aruco_mapper.compute_homography(last_scene_frame)
    if H is None:
        now = time.time()
        if now - _last_aruco_log_ts >= 1.0:
            print("  [aruco] not all 4 markers visible")
            _last_aruco_log_ts = now
        return False

    fixation_idx = calib_state - 1
    tx, ty = calib_points_screen[fixation_idx]
    proj = H @ np.array([tx, ty, 1.0], dtype=float)
    if abs(proj[2]) < 1e-9:
        now = time.time()
        if now - _last_aruco_log_ts >= 1.0:
            print("  [aruco] degenerate homography projection")
            _last_aruco_log_ts = now
        return False
    target_u = float(proj[0] / proj[2])
    target_v = float(proj[1] / proj[2])

    sample_idx = len(calib_collect_frames)
    img_name = f"fix{fixation_idx:02d}_sample{sample_idx:02d}.png"
    img_path = os.path.join(calib_session_dir, img_name)
    cv2.imwrite(img_path, last_eye_frame)
    scene_img_name = f"fix{fixation_idx:02d}_sample{sample_idx:02d}_scene.png"
    scene_img_path = os.path.join(calib_session_dir, scene_img_name)
    cv2.imwrite(scene_img_path, last_scene_frame)

    px = float(last_pupil_center[0])
    py = float(last_pupil_center[1])
    calib_pending_rows.append([
        img_name, fixation_idx, tx, ty,
        f"{px:.3f}", f"{py:.3f}", f"{last_confidence:.4f}", f"{time.time():.3f}",
        f"{target_u:.3f}", f"{target_v:.3f}",
    ])
    calib_pending_image_paths.append(img_path)
    calib_pending_image_paths.append(scene_img_path)

    calib_collect_frames.append(last_pupil_center.copy())
    calib_collect_scene.append(np.array([target_u, target_v], dtype=float))
    if len(calib_collect_frames) < CALIB_SAMPLES:
        return False

    samples = np.array(calib_collect_frames)
    raw_std = float(np.max(np.std(samples, axis=0)))

    median = np.median(samples, axis=0)
    deviations = np.linalg.norm(samples - median, axis=1)
    keep_idx = np.argsort(deviations)[:CALIB_INLIERS]
    inliers = samples[keep_idx]
    max_std = float(np.max(np.std(inliers, axis=0)))

    if max_std > CALIB_STD_THRESH:
        print(f"  High variance (inlier std={max_std:.2f}px, raw std={raw_std:.2f}px). "
              f"Retrying — hold still and press 'c'.")
        for p in calib_pending_image_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        calib_pending_rows = []
        calib_pending_image_paths = []
        calib_collecting = False
        calib_collect_frames = []
        calib_collect_scene = []
        return False

    scene_samples = np.array(calib_collect_scene)
    scene_inliers = scene_samples[keep_idx]
    scene_max_std = float(np.max(np.std(scene_inliers, axis=0)))
    if scene_max_std > CALIB_SCENE_STD_THRESH:
        print(f"  High label variance (std={scene_max_std:.2f}px). "
              f"Head may have moved. Retrying — hold still and press 'c'.")
        for p in calib_pending_image_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        calib_pending_rows = []
        calib_pending_image_paths = []
        calib_collecting = False
        calib_collect_frames = []
        calib_collect_scene = []
        return False

    median_vec = np.median(inliers, axis=0)
    calib_vectors_eye.append(median_vec)
    scene_median = np.median(scene_inliers, axis=0)
    calib_points_scene.append(scene_median)

    try:
        H_chk, reproj_err = _aruco_mapper.compute_homography(last_scene_frame)
    except Exception as e:
        H_chk, reproj_err = None, None
        print(f"  ArUco detection error for fixation {fixation_idx}: {e}")
    if H_chk is not None:
        v = np.array([tx, ty, 1.0], dtype=float)
        proj_chk = H_chk @ v
        if abs(proj_chk[2]) > 1e-9:
            u = float(proj_chk[0] / proj_chk[2])
            vp = float(proj_chk[1] / proj_chk[2])
            err = math.sqrt((u - scene_median[0])**2 + (vp - scene_median[1])**2)
            print(f"  ArUco check: fixation {fixation_idx} predicted ({u:.1f},{vp:.1f}) "
                  f"vs label-median ({scene_median[0]:.1f},{scene_median[1]:.1f}), "
                  f"err={err:.1f}px (reproj={reproj_err:.1f}px)")
    else:
        print(f"  ArUco: not all 4 markers visible for fixation {fixation_idx}")

    append_label_rows(calib_labels_path, calib_pending_rows)
    calib_pending_rows = []
    calib_pending_image_paths = []
    calib_collecting = False
    calib_collect_frames = []
    calib_collect_scene = []

    if calib_state >= calib_total_points:
        n = len(calib_vectors_eye)
        skipped = len(calib_skipped_indices)
        if n < 6:
            print(f"Not enough non-skipped points to fit "
                  f"({n} captured, {skipped} skipped). Need at least 6.")
        elif compute_polynomial_calibration():
            calibrated = True
            print("Calibration Complete!")
        calib_state = 0
        _teardown_calibration_overlay()
    else:
        calib_state += 1
        print(f"  Captured {len(calib_vectors_eye)}/{calib_total_points}. Look at next dot, press 'c'.")
    return True


def _teardown_calibration_overlay():
    global calib_tk_root, calib_tk_canvas
    if calib_tk_root is not None:
        try:
            calib_tk_root.destroy()
        except tk.TclError:
            pass
    calib_tk_root = None
    calib_tk_canvas = None


def _build_eye_cam_settings() -> CameraSettings:
    if HIGH_FPS_MODE:
        return CameraSettings(request_width=320, request_height=240,
                              request_fps=120, exposure=-5, flip_vertical=True)
    return CameraSettings(exposure=-5, flip_vertical=True)


def _build_scene_cam_settings() -> CameraSettings:
    return CameraSettings(request_width=SCENE_REQUEST_WIDTH,
                          request_height=SCENE_REQUEST_HEIGHT)


def process_camera():
    global selected_camera, circle_x, circle_y, calibrated, last_scene_frame
    global scene_cam_width, scene_cam_height
    global _pupil_detector, _jump_gate, _conf_gate

    try:
        cam_index = int(selected_camera.get())
    except ValueError:
        print("No valid camera selected.")
        return

    eye_cam = OpenCVCamera(cam_index, _build_eye_cam_settings())
    if not eye_cam.open():
        print(f"Error: Could not open eye camera at index {cam_index}.")
        return

    _pupil_detector = PupilLabsDetector(focal_length_px=EYE_CAM_FOCAL_LENGTH_PX)
    _jump_gate = JumpGate(threshold_px=PUPIL_JUMP_THRESH, buffer_size=PUPIL_BUFFER_SIZE)
    _conf_gate = ConfidenceGate(threshold=CONF_THRESH)

    external_index = 1 if cam_index == 0 else 0
    scene_cam = OpenCVCamera(external_index, _build_scene_cam_settings())
    if scene_cam.open():
        scene_cam_width = scene_cam.width
        scene_cam_height = scene_cam.height
        print(f"Scene cam: requested {SCENE_REQUEST_WIDTH}x{SCENE_REQUEST_HEIGHT}, "
              f"got {scene_cam_width}x{scene_cam_height}")
    else:
        scene_cam = None

    circle_x = (scene_cam_width or DISPLAY_WIDTH) // 2
    circle_y = (scene_cam_height or DISPLAY_HEIGHT) // 2
    calibrated = False

    cv2.namedWindow("Eye Camera")
    cv2.moveWindow("Eye Camera", 50, 50)

    if scene_cam is not None:
        cv2.namedWindow("External Camera (Gaze)")
        cv2.moveWindow("External Camera (Gaze)", 720, 50)

    print("Controls: 'c' = calibrate, 'l' = load calibration, 'r' = reset pupil 3D model, 'q' = quit, space = pause")

    while True:
        eye_frame = eye_cam.read()
        if eye_frame is None:
            break

        process_frame(eye_frame)

        if calib_collecting:
            tick_capture()

        if scene_cam is not None:
            ext_frame = scene_cam.read()
            if ext_frame is not None:
                last_scene_frame = ext_frame.copy()
                _aruco_mapper.update_marker_count(last_scene_frame)
                ext_frame_resized = cv2.resize(ext_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                if calibrated and calib_state == 0:
                    update_gaze_circle_from_current_gaze()
                    disp_x = int(circle_x * DISPLAY_WIDTH / (scene_cam_width or DISPLAY_WIDTH))
                    disp_y = int(circle_y * DISPLAY_HEIGHT / (scene_cam_height or DISPLAY_HEIGHT))
                    cv2.circle(ext_frame_resized, (disp_x, disp_y), 8, (0, 255, 0), -1)

                cv2.imshow("External Camera (Gaze)", ext_frame_resized)

        if calib_state > 0 and calib_tk_root is not None:
            render_calibration_overlay()
            try:
                calib_tk_root.update()
            except tk.TclError:
                _teardown_calibration_overlay()

        tk_key = _tk_key_queue.pop(0) if _tk_key_queue else None
        cv_key_raw = cv2.waitKey(1) & 0xFF
        cv_key = chr(cv_key_raw) if cv_key_raw != 255 else None
        key = tk_key or cv_key

        if key == 'q':
            break
        elif cv_key_raw == ord(' '):
            cv2.waitKey(0)
        elif key == 'l':
            load_calibration()
        elif key == 'c':
            if calib_state == 0:
                start_calibration()
            elif not calib_collecting:
                begin_capture()
        elif key == 's':
            if calib_state > 0:
                skip_current_point()
        elif key == 'r':
            reset_pye3d_model()

    _teardown_calibration_overlay()
    eye_cam.release()
    if scene_cam is not None:
        scene_cam.release()
    cv2.destroyAllWindows()


def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


def selection_gui():
    global selected_camera
    cameras = detect_cameras()

    root = tk.Tk()
    root.title("Select Input Source")

    root.eval('tk::PlaceWindow . center')
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)

    tk.Label(root, text="Pupil-only Eye Tracker", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select Camera:").pack(pady=5)

    selected_camera = tk.StringVar()
    selected_camera.set(str(cameras[0]) if cameras else "No cameras found")

    camera_dropdown = ttk.Combobox(root, textvariable=selected_camera,
                                   values=[str(cam) for cam in cameras])
    camera_dropdown.pack(pady=5)

    tk.Button(root, text="Start Camera",
              command=lambda: [root.destroy(), process_camera()]).pack(pady=5)
    tk.Button(root, text="Browse Video",
              command=lambda: [root.destroy(), process_video()]).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    selection_gui()
