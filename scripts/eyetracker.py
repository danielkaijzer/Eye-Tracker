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
import csv
import cv2
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import time
from collections import deque

BUFFER_SIZE = 5
screen_buffer = deque(maxlen=BUFFER_SIZE)

calib_points_screen = []
calib_vectors_eye = []
calib_skipped_indices = []
calib_state = 0
calib_total_points = 12

calib_collecting = False
calib_collect_frames = []
calib_collect_scene = []
calib_points_scene = []
CALIB_SAMPLES = 15
CALIB_INLIERS = 10
CALIB_STD_THRESH = 12.0
CALIB_SCENE_STD_THRESH = 10.0
CALIB_WARMUP = 5
_calib_warmup_remaining = 0

poly_coeffs_x = None
poly_coeffs_y = None

HIGH_FPS_MODE = False

last_pupil_center = None
last_confidence = 0.0

_pupil_detector_2d = None
_pupil_detector_3d = None
_pupil_camera = None

# Eye cam: 0.3 MP (640x480) with 80° lens. If the spec turns out to be
# diagonal FOV rather than horizontal, set EYE_CAM_FOV_IS_DIAGONAL = True.
EYE_CAM_RESOLUTION = (640, 480)
EYE_CAM_FOV_DEG = 80.0
EYE_CAM_FOV_IS_DIAGONAL = True


def _compute_eye_focal_length_px():
    w, h = EYE_CAM_RESOLUTION
    ref = math.sqrt(w * w + h * h) if EYE_CAM_FOV_IS_DIAGONAL else w
    return (ref / 2.0) / math.tan(math.radians(EYE_CAM_FOV_DEG / 2.0))


EYE_CAM_FOCAL_LENGTH_PX = _compute_eye_focal_length_px()

# Library confidence = min(0.99, support/circ) * (support/total_edges)^2.
# Clean pupils typically score > 0.8; 0.25 is permissive.
CONF_THRESH = 0.25

# Outlier rejection on pupil center. A fast saccade to an extreme-gaze
# angle moves the pupil ~200-300 px in a single 30fps frame, so the
# threshold needs to accommodate that — only catches catastrophic flips.
PUPIL_BUFFER_SIZE = 7
pupil_buffer = deque(maxlen=PUPIL_BUFFER_SIZE)
PUPIL_JUMP_THRESH = 400.0

calibrated = False

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

SCENE_REQUEST_WIDTH = 1920
SCENE_REQUEST_HEIGHT = 1080

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
_last_aruco_marker_count = 0

ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_MARKER_PX = 160
ARUCO_QUIET_ZONE_PX = 200
ARUCO_IDS = (0, 1, 2, 3)
_aruco_photo_images = []
_aruco_detector = None


def detect_cameras(max_cams=10):
    available_cameras = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))


def _get_aruco_dict():
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is not available in this OpenCV build. "
            "Install opencv-contrib-python to enable ArUco support."
        )
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _get_aruco_detector():
    global _aruco_detector
    if _aruco_detector is not None:
        return _aruco_detector
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is not available in this OpenCV build. "
            "Install opencv-contrib-python to enable ArUco support."
        )
    try:
        dictionary = _get_aruco_dict()
        params = cv2.aruco.DetectorParameters()
        _aruco_detector = cv2.aruco.ArucoDetector(dictionary, params)
    except AttributeError:
        _aruco_detector = None
    return _aruco_detector


def _detect_aruco_markers(scene_bgr):
    """Return (corners, ids) using the modern ArucoDetector if available,
    else the legacy detectMarkers call. corners/ids follow the OpenCV shape."""
    detector = _get_aruco_detector()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(scene_bgr)
        return corners, ids
    dictionary = _get_aruco_dict()
    try:
        params = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(scene_bgr, dictionary, parameters=params)
    return corners, ids


def _update_aruco_count(scene_bgr):
    global _last_aruco_marker_count
    if scene_bgr is None:
        _last_aruco_marker_count = 0
        return
    _, ids = _detect_aruco_markers(scene_bgr)
    if ids is None:
        _last_aruco_marker_count = 0
        return
    found = {int(i) for i in ids.flatten()} & set(ARUCO_IDS)
    _last_aruco_marker_count = len(found)


def _aruco_screen_centers():
    """Return dict {marker_id: (cx, cy)} of on-screen marker centers.
    IDs: 0=TL, 1=TR, 2=BR, 3=BL. Quiet zone hugs the corner; marker is centered
    inside the ARUCO_QUIET_ZONE_PX x ARUCO_QUIET_ZONE_PX white pad."""
    if screen_width is None or screen_height is None:
        return {}
    q = ARUCO_QUIET_ZONE_PX
    half = q / 2.0
    positions = {
        0: (half, half),
        1: (screen_width - half, half),
        2: (screen_width - half, screen_height - half),
        3: (half, screen_height - half),
    }
    return positions


def _aruco_quiet_zone_origins():
    """Return dict {marker_id: (x_nw, y_nw)} of NW corners of the quiet-zone
    rectangle (which hugs the screen corners)."""
    if screen_width is None or screen_height is None:
        return {}
    q = ARUCO_QUIET_ZONE_PX
    return {
        0: (0, 0),
        1: (screen_width - q, 0),
        2: (screen_width - q, screen_height - q),
        3: (0, screen_height - q),
    }


def _ensure_aruco_photo_images():
    """Generate and cache the four marker images as tk.PhotoImage instances.
    Must be called after a tk root exists."""
    global _aruco_photo_images
    if len(_aruco_photo_images) == 4:
        return _aruco_photo_images
    dictionary = _get_aruco_dict()
    imgs = []
    for marker_id in ARUCO_IDS:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, ARUCO_MARKER_PX)
        ok, buf = cv2.imencode(".png", marker)
        if not ok:
            raise RuntimeError(f"Failed to encode ArUco marker {marker_id} to PNG.")
        b64 = base64.b64encode(buf.tobytes())
        imgs.append(tk.PhotoImage(data=b64))
    _aruco_photo_images = imgs
    return _aruco_photo_images


def _draw_aruco_corners(canvas):
    if screen_width is None or screen_height is None:
        return
    photos = _ensure_aruco_photo_images()
    origins = _aruco_quiet_zone_origins()
    q = ARUCO_QUIET_ZONE_PX
    inset = (q - ARUCO_MARKER_PX) // 2
    for i, marker_id in enumerate(ARUCO_IDS):
        ox, oy = origins[marker_id]
        canvas.create_rectangle(ox, oy, ox + q, oy + q, fill="white", outline="")
        canvas.create_image(ox + inset, oy + inset, anchor="nw", image=photos[i])


def detect_aruco_homography(scene_bgr):
    """Detect the four corner ArUco markers in a scene-cam frame and compute
    the screen -> scene-cam homography.

    Returns (H, reprojection_error) where H is a 3x3 homography matrix mapping
    (x_screen, y_screen, 1) -> (x_scene, y_scene, 1), or (None, None) if <4
    markers were found or the homography degenerated.
    """
    if scene_bgr is None:
        return None, None
    corners, ids = _detect_aruco_markers(scene_bgr)
    if ids is None or len(ids) < 4:
        return None, None
    id_to_center = {}
    ids_flat = ids.flatten()
    for c, mid in zip(corners, ids_flat):
        mid = int(mid)
        if mid not in ARUCO_IDS:
            continue
        pts = c.reshape(-1, 2)
        id_to_center[mid] = pts.mean(axis=0)
    if any(i not in id_to_center for i in ARUCO_IDS):
        return None, None

    screen_centers = _aruco_screen_centers()
    if not screen_centers:
        return None, None

    screen_pts = np.array([screen_centers[i] for i in ARUCO_IDS], dtype=np.float32)
    scene_pts = np.array([id_to_center[i] for i in ARUCO_IDS], dtype=np.float32)

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
    reproj_err = float(np.mean(errs))
    return H, reproj_err


# Unused in the runtime loop; wired in once inference-time screen mapping is needed.
def screen_pixel_for_scene_gaze(scene_xy, scene_bgr):
    """If ArUco markers are visible in scene_bgr, map a predicted scene-cam
    gaze point back to screen pixels via inverse homography.

    Returns (x_screen, y_screen) or None.
    """
    H, _ = detect_aruco_homography(scene_bgr)
    if H is None:
        return None
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    v = np.array([scene_xy[0], scene_xy[1], 1.0], dtype=float)
    out = H_inv @ v
    if abs(out[2]) < 1e-9:
        return None
    return (float(out[0] / out[2]), float(out[1] / out[2]))


def _get_pupil_detectors():
    """Lazy-init Detector2D + pye3d Detector3D sharing one CameraModel."""
    global _pupil_detector_2d, _pupil_detector_3d, _pupil_camera
    if _pupil_detector_3d is None:
        from pupil_detectors import Detector2D
        from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
        _pupil_camera = CameraModel(focal_length=EYE_CAM_FOCAL_LENGTH_PX,
                                    resolution=(640, 480))
        _pupil_detector_2d = Detector2D()
        _pupil_detector_3d = Detector3D(camera=_pupil_camera,
                                        long_term_mode=DetectorMode.blocking)
    return _pupil_detector_2d, _pupil_detector_3d


def reset_pye3d_model():
    """Discard the 3D eye model so the next detect call re-initializes it.
    Use after re-seating the headset."""
    global _pupil_detector_3d
    _pupil_detector_3d = None
    print("Pupil 3D model reset — give it ~30s to reconverge.")


def detect_pupil(gray_frame):
    """Returns (center, ellipse, confidence).

    center: (int, int) for cv2.circle
    ellipse: ((cx, cy), (minor, major), angle_deg) for cv2.ellipse
    The 2D ellipse output is from pye3d (Detector2D's ellipse refined by
    the 3D model's constraint).
    """
    det2d, det3d = _get_pupil_detectors()
    result_2d = det2d.detect(gray_frame)
    result_2d["timestamp"] = time.perf_counter()
    result_3d = det3d.update_and_detect(result_2d, gray_frame)
    conf = float(result_3d["confidence"])
    if conf <= 0.0:
        return None, None, 0.0
    ell = result_3d["ellipse"]
    cx, cy = ell["center"]
    center_int = (int(round(cx)), int(round(cy)))
    ellipse_tuple = ((cx, cy), ell["axes"], ell["angle"])
    return center_int, ellipse_tuple, conf


def smooth_pupil_position(raw_center):
    """Reject frames where pupil center jumps too far from running median."""
    raw = np.array(raw_center, dtype=float)

    if len(pupil_buffer) == 0:
        pupil_buffer.append(raw)
        return raw

    median = np.median(np.array(pupil_buffer), axis=0)
    if np.linalg.norm(raw - median) > PUPIL_JUMP_THRESH:
        return None

    pupil_buffer.append(raw)
    return raw


_last_gate_log_ts = 0.0


def process_frame(frame):
    """Detect pupil, gate on confidence + outlier rejection, update last_pupil_center."""
    global last_pupil_center, last_confidence, _last_gate_log_ts, last_eye_frame

    frame = crop_to_aspect_ratio(frame)
    last_eye_frame = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pupil_center, pupil_ellipse, confidence = detect_pupil(gray_frame)
    last_confidence = confidence

    accepted = None
    reject_reason = None
    if pupil_center is None or confidence <= 0:
        reject_reason = "no detection"
    elif confidence < CONF_THRESH:
        reject_reason = f"conf={confidence:.2f} < CONF_THRESH={CONF_THRESH}"
    else:
        accepted = smooth_pupil_position(pupil_center)
        if accepted is None:
            jump_px = float("nan")
            if len(pupil_buffer) > 0:
                median = np.median(np.array(pupil_buffer), axis=0)
                jump_px = float(np.linalg.norm(np.array(pupil_center, dtype=float) - median))
            reject_reason = (f"jump={jump_px:.0f}px > PUPIL_JUMP_THRESH={PUPIL_JUMP_THRESH:.0f} "
                             f"(conf={confidence:.2f})")

    if accepted is not None:
        last_pupil_center = accepted
        cv2.circle(frame, pupil_center, 4, (0, 255, 0), -1)
        if pupil_ellipse is not None:
            cv2.ellipse(frame, pupil_ellipse, (20, 255, 255), 2)
        text = f"Pupil: ({pupil_center[0]}, {pupil_center[1]})  conf={confidence:.2f}"
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
    if poly_coeffs_x is None or poly_coeffs_y is None:
        return
    if scene_cam_width is None or scene_cam_height is None:
        return

    px, py = last_pupil_center[0], last_pupil_center[1]
    feat = _build_poly_features(px, py)

    u_scene = feat @ poly_coeffs_x
    v_scene = feat @ poly_coeffs_y

    screen_buffer.append((u_scene, v_scene))
    avg_u = np.mean([p[0] for p in screen_buffer])
    avg_v = np.mean([p[1] for p in screen_buffer])

    circle_x = int(np.clip(avg_u, 0, scene_cam_width - 1))
    circle_y = int(np.clip(avg_v, 0, scene_cam_height - 1))


def _build_poly_features(gx, gy):
    """2nd-degree polynomial feature row: [1, gx, gy, gx^2, gy^2, gx*gy]"""
    return np.array([1.0, gx, gy, gx*gx, gy*gy, gx*gy])


def compute_polynomial_calibration():
    """Fit 2nd-degree polynomial from pupil positions to scene-cam coords."""
    global poly_coeffs_x, poly_coeffs_y
    n = len(calib_vectors_eye)
    if n < 6:
        print(f"Need at least 6 calibration points, have {n}.")
        return False
    if len(calib_points_scene) != n:
        print(f"  ERROR: calib_points_scene ({len(calib_points_scene)}) and "
              f"calib_vectors_eye ({n}) length mismatch. Aborting fit.")
        return False

    A = np.zeros((n, 6))
    bx = np.zeros(n)
    by = np.zeros(n)
    for i, (v, pt) in enumerate(zip(calib_vectors_eye, calib_points_scene)):
        A[i] = _build_poly_features(v[0], v[1])
        bx[i] = pt[0]
        by[i] = pt[1]

    cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    poly_coeffs_x = cx
    poly_coeffs_y = cy

    errors = []
    for i in range(n):
        A_loo = np.delete(A, i, axis=0)
        bx_loo = np.delete(bx, i)
        by_loo = np.delete(by, i)
        cx_loo, _, _, _ = np.linalg.lstsq(A_loo, bx_loo, rcond=None)
        cy_loo, _, _, _ = np.linalg.lstsq(A_loo, by_loo, rcond=None)
        pred_x = A[i] @ cx_loo
        pred_y = A[i] @ cy_loo
        err = math.sqrt((pred_x - bx[i])**2 + (pred_y - by[i])**2)
        errors.append(err)

    avg_err = np.mean(errors)
    max_err = np.max(errors)
    skipped = len(calib_skipped_indices)
    print(f"Polynomial calibration fitted ({n} points, {skipped} skipped).")
    print(f"  LOO error: avg={avg_err:.1f}px, max={max_err:.1f}px")
    err_threshold = 0.04 * (scene_cam_width or 640)
    if avg_err > err_threshold:
        print(f"  WARNING: High error (>{err_threshold:.0f}px at this scene-cam resolution) — consider recalibrating.")

    _save_calibration()
    return True


def _calibration_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_pupil.npz")


def _history_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_pupil_history.npz")


def _save_calibration():
    global poly_coeffs_x, poly_coeffs_y
    path = _calibration_path()
    vectors = np.array(calib_vectors_eye)
    scene_points = np.array(calib_points_scene)
    screen_points = np.array(calib_points_screen)
    aruco_centers_map = _aruco_screen_centers()
    aruco_screen_centers = np.array(
        [aruco_centers_map[i] for i in ARUCO_IDS], dtype=float
    ) if aruco_centers_map else np.zeros((0, 2), dtype=float)
    aruco_dict_id = int(cv2.aruco.DICT_4X4_50) if hasattr(cv2, "aruco") else -1
    np.savez(path,
             poly_coeffs_x=poly_coeffs_x,
             poly_coeffs_y=poly_coeffs_y,
             vectors=vectors,
             scene_points=scene_points,
             screen_points=screen_points,
             coord_space="scene",
             scene_width=scene_cam_width,
             scene_height=scene_cam_height,
             screen_width=screen_width,
             screen_height=screen_height,
             aruco_dict_id=aruco_dict_id,
             aruco_dict_name=ARUCO_DICT_NAME,
             aruco_marker_px=ARUCO_MARKER_PX,
             aruco_quiet_zone_px=ARUCO_QUIET_ZONE_PX,
             aruco_screen_centers=aruco_screen_centers,
             timestamp=time.time())
    print(f"  Calibration saved to {path}")

    hist_path = _history_path()
    # Old history files use `all_points` (screen-space); treat any file without
    # `all_scene_points` as fresh so we don't mix coord spaces.
    if os.path.exists(hist_path):
        old = np.load(hist_path, allow_pickle=True)
        if 'all_scene_points' in old.files:
            old_vectors = list(old['all_vectors'])
            old_scene_points = list(old['all_scene_points'])
        else:
            old_vectors = []
            old_scene_points = []
    else:
        old_vectors = []
        old_scene_points = []
    old_vectors.append(vectors)
    old_scene_points.append(scene_points)
    np.savez(hist_path,
             all_vectors=np.array(old_vectors, dtype=object),
             all_scene_points=np.array(old_scene_points, dtype=object))
    total_pts = sum(len(v) for v in old_vectors)
    print(f"  History: {len(old_vectors)} sessions, {total_pts} total points.")


def load_calibration():
    global poly_coeffs_x, poly_coeffs_y, calibrated
    global screen_width, screen_height
    global scene_cam_width, scene_cam_height
    path = _calibration_path()
    if not os.path.exists(path):
        print("No saved calibration found.")
        return False
    data = np.load(path, allow_pickle=True)
    if "coord_space" not in data.files or str(data["coord_space"]) != "scene":
        print("  ERROR: saved calibration is not in scene-cam coord space. "
              "Recalibrate with 'c'.")
        return False
    poly_coeffs_x = data['poly_coeffs_x']
    poly_coeffs_y = data['poly_coeffs_y']
    if 'screen_width' in data.files:
        screen_width = int(data['screen_width'])
    if 'screen_height' in data.files:
        screen_height = int(data['screen_height'])
    if 'scene_width' in data.files:
        scene_cam_width = int(data['scene_width'])
        scene_cam_height = int(data['scene_height'])
    scene_w = scene_cam_width
    scene_h = scene_cam_height
    ts = float(data['timestamp'])
    age_hrs = (time.time() - ts) / 3600
    calibrated = True
    print(f"Calibration loaded (age: {age_hrs:.1f}h, scene={scene_w}x{scene_h}).")
    if age_hrs > 24:
        print("  WARNING: Calibration is >24h old. Consider recalibrating.")
    return True


def _dataset_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration",
    )


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
    pupil_buffer.clear()
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

    session_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    calib_session_dir = os.path.join(_dataset_root(), session_name)
    os.makedirs(calib_session_dir, exist_ok=True)
    calib_labels_path = os.path.join(calib_session_dir, "labels.csv")
    with open(calib_labels_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "image_path", "fixation_id", "x_screen", "y_screen",
            "pupil_x", "pupil_y", "confidence", "timestamp",
            "scene_target_x", "scene_target_y",
        ])

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

    aruco_color = "#00ff00" if _last_aruco_marker_count == 4 else "#ff6060"
    calib_tk_canvas.create_text(screen_width // 2, screen_height - 40,
                                text=f"aruco: {_last_aruco_marker_count}/4 markers visible",
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

    H, _ = detect_aruco_homography(last_scene_frame)
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
        H_chk, reproj_err = detect_aruco_homography(last_scene_frame)
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

    with open(calib_labels_path, "a", newline="") as f:
        csv.writer(f).writerows(calib_pending_rows)
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


def process_camera():
    global selected_camera, circle_x, circle_y, calibrated, last_scene_frame
    global scene_cam_width, scene_cam_height

    try:
        cam_index = int(selected_camera.get())
    except ValueError:
        print("No valid camera selected.")
        return

    eye_cap = cv2.VideoCapture(cam_index)
    if not eye_cap.isOpened():
        print(f"Error: Could not open eye camera at index {cam_index}.")
        return

    if HIGH_FPS_MODE:
        eye_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        eye_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        eye_cap.set(cv2.CAP_PROP_FPS, 120)

    eye_cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    external_index = 1 if cam_index == 0 else 0
    external_cap = cv2.VideoCapture(external_index)

    if external_cap.isOpened():
        external_cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCENE_REQUEST_WIDTH)
        external_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCENE_REQUEST_HEIGHT)
        scene_cam_width = int(external_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        scene_cam_height = int(external_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Scene cam: requested {SCENE_REQUEST_WIDTH}x{SCENE_REQUEST_HEIGHT}, "
              f"got {scene_cam_width}x{scene_cam_height}")
    else:
        external_cap = None

    circle_x = (scene_cam_width or DISPLAY_WIDTH) // 2
    circle_y = (scene_cam_height or DISPLAY_HEIGHT) // 2
    calibrated = False

    cv2.namedWindow("Eye Camera")
    cv2.moveWindow("Eye Camera", 50, 50)

    if external_cap is not None:
        cv2.namedWindow("External Camera (Gaze)")
        cv2.moveWindow("External Camera (Gaze)", 720, 50)

    print("Controls: 'c' = calibrate, 'l' = load calibration, 'r' = reset pupil 3D model, 'q' = quit, space = pause")

    while True:
        ret_eye, eye_frame = eye_cap.read()
        if not ret_eye:
            break

        eye_frame = cv2.flip(eye_frame, 0)
        process_frame(eye_frame)

        if calib_collecting:
            tick_capture()

        if external_cap is not None:
            ret_ext, ext_frame = external_cap.read()
            if ret_ext:
                last_scene_frame = ext_frame.copy()
                _update_aruco_count(last_scene_frame)
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
    eye_cap.release()
    if external_cap is not None:
        external_cap.release()
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
