"""Pupil-only eye tracker (no corneal reflection).

Dark-pupil detection -> ellipse fit -> bivariate polynomial map from
pupil center to scene-camera coordinates. Based on Kassner, Patera,
Bulling 2014 (arXiv:1405.0006). v1 reuses the blob -> local-ROI
adaptive-threshold -> fitEllipse detector from the PCCR script; a
Swirski-style edge-based detector is a planned v2 upgrade.
"""
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
calib_vectors_eye = []    # np.array([px, py]) pupil centers, one per target
calib_state = 0
calib_total_points = 12

calib_collecting = False
calib_collect_frames = []
CALIB_SAMPLES = 15
CALIB_STD_THRESH = 3.0

poly_coeffs_x = None
poly_coeffs_y = None

HIGH_FPS_MODE = False

# --- Pupil globals ---
last_pupil_center = None  # np.array([x, y], dtype=float) or None if invalid
last_confidence = 0.0

# Confidence = supporting-edge-length / Ramanujan-2 ellipse circumference.
# Pupil Labs report values up to ~0.97 on clean data; 0.3–0.6 is typical with
# partial occlusion. Tune lower if too many frames are rejected.
CONF_THRESH = 0.35

# Outlier rejection on pupil center — guards against the detector flipping
# to a non-pupil blob (eyelash, eyebrow). Tune after a first calibration run.
PUPIL_BUFFER_SIZE = 7
pupil_buffer = deque(maxlen=PUPIL_BUFFER_SIZE)
PUPIL_JUMP_THRESH = 200.0

# --- Świrski / Pupil-Labs detection pipeline params ---
# Center-surround Haar search: scan image at these half-widths (px).
SWIRSKI_R_MIN = 20
SWIRSKI_R_MAX = 60
SWIRSKI_R_STEP = 4
SWIRSKI_XY_STEP = 4

CANNY_LOW = 40
CANNY_HIGH = 100

# "Dark" pixels: darker than (1st-percentile pixel + DARK_OFFSET). Proxy for
# Pupil Labs' "lowest spike in histogram + user offset".
DARK_OFFSET = 25

# Spectral reflection (glint) cutoff; edges near these pixels are dropped.
BRIGHT_THRESH = 200

# Curvature-continuity split: if consecutive tangent vectors' dot product
# drops below this, split the contour there.
CURVATURE_SPLIT_DOT = 0.3

PUPIL_MIN_MINOR_AXIS = 10
PUPIL_MAX_MAJOR_AXIS_FRAC = 0.5   # of smaller ROI side
PUPIL_MAX_ASPECT = 2.5

# Inlier distance (px) from contour point to candidate ellipse.
ELLIPSE_INLIER_THRESH_PX = 2.5
# Minimum fraction of a sub-contour's points that must be inliers.
ELLIPSE_SUPPORT_FRAC = 0.55

calibrated = False

EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2

circle_x = EXT_CX
circle_y = EXT_CY


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


def _integral_rect_sum(integral, x1, y1, x2, y2):
    """Sum of pixels in rect [y1:y2, x1:x2) using an integral image."""
    return (integral[y2, x2] - integral[y1, x2]
            - integral[y2, x1] + integral[y1, x1])


def _find_pupil_region(gray):
    """Świrski-style Haar center-surround to find initial pupil region.

    Returns (center, half-width) of the strongest dark-center / bright-surround
    response, or (None, 0).
    """
    h, w = gray.shape
    if min(h, w) < 2 * SWIRSKI_R_MIN + 4:
        return None, 0

    integral = cv2.integral(gray).astype(np.int64)

    best_response = -np.inf
    best_center = None
    best_r = 0

    for r in range(SWIRSKI_R_MIN, SWIRSKI_R_MAX + 1, SWIRSKI_R_STEP):
        if 2 * r >= min(h, w):
            break
        xs = np.arange(2 * r, w - 2 * r, SWIRSKI_XY_STEP)
        ys = np.arange(2 * r, h - 2 * r, SWIRSKI_XY_STEP)
        if len(xs) == 0 or len(ys) == 0:
            continue
        X, Y = np.meshgrid(xs, ys)

        inner = _integral_rect_sum(integral, X - r, Y - r, X + r, Y + r)
        outer_total = _integral_rect_sum(integral, X - 2 * r, Y - 2 * r,
                                         X + 2 * r, Y + 2 * r)
        outer = outer_total - inner

        inner_area = (2 * r) ** 2
        outer_area = (4 * r) ** 2 - inner_area
        response = outer.astype(float) / outer_area - inner.astype(float) / inner_area

        idx = np.unravel_index(np.argmax(response), response.shape)
        if response[idx] > best_response:
            best_response = float(response[idx])
            best_center = (int(X[idx]), int(Y[idx]))
            best_r = r

    return best_center, best_r


def _find_dark_threshold(roi):
    """Pixel value below which a pixel counts as 'dark'.

    Paper: offset from lowest spike in intensity histogram. Proxy here:
    1st-percentile pixel + DARK_OFFSET — robust to hot pixels, no peak-finding
    dependency.
    """
    darkest = int(np.percentile(roi, 1))
    return darkest + DARK_OFFSET


def _find_spectral_mask(roi):
    """Dilated mask of near-saturated pixels (IR glints / spectral reflections)."""
    _, mask = cv2.threshold(roi, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(mask, k, iterations=2)


def _filter_edges(edges, roi):
    """Keep edges on dark-region boundaries; drop edges near glints."""
    dark_thresh = _find_dark_threshold(roi)
    dark_mask = ((roi < dark_thresh).astype(np.uint8)) * 255
    # Dilate so edges on the dark/light transition still pass the gate.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.dilate(dark_mask, k, iterations=1)

    bright_mask = _find_spectral_mask(roi)

    filtered = cv2.bitwise_and(edges, dark_mask)
    filtered = cv2.bitwise_and(filtered, cv2.bitwise_not(bright_mask))
    return filtered


def _split_by_curvature(contour):
    """Split a contour at points where the tangent direction changes abruptly."""
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 7:
        return [contour] if n >= 5 else []

    k = 3  # look-ahead for tangent estimation
    tangents = np.zeros((n, 2), dtype=float)
    for i in range(n):
        forward = pts[(i + k) % n] - pts[i]
        f_norm = np.linalg.norm(forward)
        if f_norm > 0:
            tangents[i] = forward / f_norm

    subs = []
    start = 0
    for i in range(1, n):
        dot = float(np.dot(tangents[i], tangents[i - 1]))
        if dot < CURVATURE_SPLIT_DOT:
            if i - start >= 5:
                subs.append(contour[start:i])
            start = i
    if n - start >= 5:
        subs.append(contour[start:])
    return subs


def _fit_candidate_ellipses(sub_contours, roi_shape):
    """Fit an ellipse to each sub-contour; filter by size and aspect ratio."""
    max_axis = int(min(roi_shape[:2]) * PUPIL_MAX_MAJOR_AXIS_FRAC)
    candidates = []
    for sub in sub_contours:
        if len(sub) < 5:
            continue
        try:
            ellipse = cv2.fitEllipse(sub)
        except cv2.error:
            continue
        (_cx, _cy), (minor, major), _ang = ellipse
        if minor < PUPIL_MIN_MINOR_AXIS or major > max_axis or minor <= 0:
            continue
        if major / minor > PUPIL_MAX_ASPECT:
            continue
        candidates.append((ellipse, sub))
    return candidates


def _point_ellipse_distance_approx(points, ellipse):
    """Approximate Euclidean distance from points to an ellipse contour.

    Uses algebraic distance scaled by average radius — accurate within a few
    percent for points close to the curve, which is all we need for inlier
    gating. Exact distance is a quartic root-finding problem; not worth it here.
    """
    (cx, cy), (minor_d, major_d), angle_deg = ellipse
    if minor_d <= 0 or major_d <= 0:
        return np.full(len(points), np.inf)
    a = major_d / 2.0
    b = minor_d / 2.0
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    x_rot = cos_a * dx + sin_a * dy
    y_rot = -sin_a * dx + cos_a * dy
    d_normed = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    return np.abs(d_normed - 1.0) * (a + b) / 2.0


def _find_supporting_contours(ellipse, sub_contours):
    """Sub-contours with at least ELLIPSE_SUPPORT_FRAC of points near the ellipse."""
    supporting = []
    for sub in sub_contours:
        pts = sub.reshape(-1, 2).astype(float)
        dists = _point_ellipse_distance_approx(pts, ellipse)
        inliers = dists < ELLIPSE_INLIER_THRESH_PX
        if len(inliers) > 0 and inliers.mean() >= ELLIPSE_SUPPORT_FRAC:
            supporting.append(sub)
    return supporting


def _ellipse_circumference_ramanujan2(a, b):
    """Ramanujan's 2nd-order approximation of ellipse perimeter."""
    if a + b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))


def _compute_confidence(ellipse, supporting_contours):
    """Supporting-edge-length / ellipse-circumference, clipped to [0, 1]."""
    _, (minor_d, major_d), _ = ellipse
    if minor_d <= 0 or major_d <= 0:
        return 0.0
    circ = _ellipse_circumference_ramanujan2(major_d / 2.0, minor_d / 2.0)
    if circ <= 0:
        return 0.0
    support_length = sum(len(sub) for sub in supporting_contours)
    return min(support_length / circ, 1.0)


def detect_pupil(gray_frame):
    """Świrski / Pupil-Labs pupil detection.

    Pipeline: Haar center-surround → Canny → dark/glint edge filter →
    curvature-split sub-contours → candidate ellipse fits → combinatorial
    support search → Ramanujan-2 confidence.

    Returns (center, ellipse, confidence) in full-frame coordinates.
    """
    region_center, region_r = _find_pupil_region(gray_frame)
    if region_center is None:
        return None, None, 0.0

    h, w = gray_frame.shape
    margin = max(int(region_r * 3), 40)
    cx, cy = region_center
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape) < 30:
        return None, None, 0.0

    edges = cv2.Canny(roi, CANNY_LOW, CANNY_HIGH)
    filtered = _filter_edges(edges, roi)

    contours, _ = cv2.findContours(filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    sub_contours = []
    for cnt in contours:
        sub_contours.extend(_split_by_curvature(cnt))
    if not sub_contours:
        return None, None, 0.0

    candidates = _fit_candidate_ellipses(sub_contours, roi.shape)
    if not candidates:
        return None, None, 0.0

    max_axis = min(roi.shape[:2]) * PUPIL_MAX_MAJOR_AXIS_FRAC
    best_conf = 0.0
    best_ellipse = None
    for cand_ellipse, _ in candidates:
        support = _find_supporting_contours(cand_ellipse, sub_contours)
        if not support:
            continue
        all_pts = np.vstack([s.reshape(-1, 2) for s in support])
        if len(all_pts) < 5:
            continue
        try:
            refit = cv2.fitEllipse(all_pts)
        except cv2.error:
            refit = cand_ellipse
        _, (minor_r, major_r), _ = refit
        if minor_r < PUPIL_MIN_MINOR_AXIS or major_r > max_axis or minor_r <= 0:
            continue
        if major_r / minor_r > PUPIL_MAX_ASPECT:
            continue
        conf = _compute_confidence(refit, support)
        if conf > best_conf:
            best_conf = conf
            best_ellipse = refit

    if best_ellipse is None:
        return None, None, best_conf

    (ex, ey), axes, angle = best_ellipse
    center = (int(ex) + x1, int(ey) + y1)
    full_ellipse = (center, axes, angle)
    return center, full_ellipse, best_conf


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


def process_frame(frame):
    """Detect pupil, gate on confidence + outlier rejection, update last_pupil_center."""
    global last_pupil_center, last_confidence

    frame = crop_to_aspect_ratio(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pupil_center, pupil_ellipse, confidence = detect_pupil(gray_frame)
    last_confidence = confidence

    accepted = None
    if pupil_center is not None and confidence >= CONF_THRESH:
        accepted = smooth_pupil_position(pupil_center)

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

    cv2.imshow("Eye Camera", frame)


def update_gaze_circle_from_current_gaze():
    """Map current pupil position to scene-camera coords via fitted polynomial."""
    global circle_x, circle_y
    if not calibrated or last_pupil_center is None:
        return
    if poly_coeffs_x is None or poly_coeffs_y is None:
        return

    px, py = last_pupil_center[0], last_pupil_center[1]
    feat = _build_poly_features(px, py)

    u = feat @ poly_coeffs_x
    v = feat @ poly_coeffs_y

    screen_buffer.append((u, v))
    avg_u = np.mean([p[0] for p in screen_buffer])
    avg_v = np.mean([p[1] for p in screen_buffer])

    circle_x = int(np.clip(avg_u, 0, EXT_WIDTH - 1))
    circle_y = int(np.clip(avg_v, 0, EXT_HEIGHT - 1))


def _build_poly_features(gx, gy):
    """2nd-degree polynomial feature row: [1, gx, gy, gx^2, gy^2, gx*gy]"""
    return np.array([1.0, gx, gy, gx*gx, gy*gy, gx*gy])


def compute_polynomial_calibration():
    """Fit 2nd-degree polynomial from pupil positions to screen coords."""
    global poly_coeffs_x, poly_coeffs_y
    n = len(calib_vectors_eye)
    if n < 6:
        print(f"Need at least 6 calibration points, have {n}.")
        return False

    A = np.zeros((n, 6))
    bx = np.zeros(n)
    by = np.zeros(n)
    for i, (v, pt) in enumerate(zip(calib_vectors_eye, calib_points_screen)):
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
    print(f"Polynomial calibration fitted ({n} points).")
    print(f"  LOO error: avg={avg_err:.1f}px, max={max_err:.1f}px")
    if avg_err > 40:
        print("  WARNING: High error — consider recalibrating.")

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
    points = np.array(calib_points_screen)
    np.savez(path,
             poly_coeffs_x=poly_coeffs_x,
             poly_coeffs_y=poly_coeffs_y,
             vectors=vectors,
             points=points,
             timestamp=time.time())
    print(f"  Calibration saved to {path}")

    hist_path = _history_path()
    if os.path.exists(hist_path):
        old = np.load(hist_path, allow_pickle=True)
        old_vectors = list(old['all_vectors'])
        old_points = list(old['all_points'])
    else:
        old_vectors = []
        old_points = []
    old_vectors.append(vectors)
    old_points.append(points)
    np.savez(hist_path,
             all_vectors=np.array(old_vectors, dtype=object),
             all_points=np.array(old_points, dtype=object))
    total_pts = sum(len(v) for v in old_vectors)
    print(f"  History: {len(old_vectors)} sessions, {total_pts} total points.")


def load_calibration():
    global poly_coeffs_x, poly_coeffs_y, calibrated
    path = _calibration_path()
    if not os.path.exists(path):
        print("No saved calibration found.")
        return False
    data = np.load(path, allow_pickle=True)
    poly_coeffs_x = data['poly_coeffs_x']
    poly_coeffs_y = data['poly_coeffs_y']
    ts = float(data['timestamp'])
    age_hrs = (time.time() - ts) / 3600
    calibrated = True
    print(f"Calibration loaded (age: {age_hrs:.1f}h).")
    if age_hrs > 24:
        print("  WARNING: Calibration is >24h old. Consider recalibrating.")
    return True


def start_calibration():
    global calib_state, calib_points_screen, calib_vectors_eye
    global calib_collecting, calib_collect_frames, calib_total_points
    calib_state = 1
    pupil_buffer.clear()
    calib_vectors_eye = []
    calib_collecting = False
    calib_collect_frames = []

    margin_x = 40
    margin_y = 40
    cols, rows = 4, 3
    calib_points_screen = []
    for r in range(rows):
        for col in range(cols):
            x = int(margin_x + col * (EXT_WIDTH - 2 * margin_x) / (cols - 1))
            y = int(margin_y + r * (EXT_HEIGHT - 2 * margin_y) / (rows - 1))
            calib_points_screen.append((x, y))
    calib_total_points = len(calib_points_screen)
    print(f"Calibration started ({calib_total_points} points). Look at the RED dot and press 'c'.")


def begin_capture():
    global calib_collecting, calib_collect_frames
    calib_collecting = True
    calib_collect_frames = []


def tick_capture():
    """Collect samples for the current target. Returns True when one point is done."""
    global calib_collecting, calib_collect_frames, calib_state, calibrated
    if not calib_collecting:
        return False
    if last_pupil_center is None:
        return False

    calib_collect_frames.append(last_pupil_center.copy())
    if len(calib_collect_frames) < CALIB_SAMPLES:
        return False

    samples = np.array(calib_collect_frames)
    std_dev = np.std(samples, axis=0)
    max_std = np.max(std_dev)

    if max_std > CALIB_STD_THRESH:
        print(f"  High variance (std={max_std:.2f}px). Retrying — hold still and press 'c'.")
        calib_collecting = False
        calib_collect_frames = []
        return False

    median_vec = np.median(samples, axis=0)
    calib_vectors_eye.append(median_vec)
    calib_collecting = False
    calib_collect_frames = []

    if calib_state >= calib_total_points:
        if compute_polynomial_calibration():
            calibrated = True
            print("Calibration Complete!")
        calib_state = 0
    else:
        calib_state += 1
        print(f"  Captured {len(calib_vectors_eye)}/{calib_total_points}. Look at next dot, press 'c'.")
    return True


def process_camera():
    global selected_camera, circle_x, circle_y, calibrated

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
        external_cap.set(cv2.CAP_PROP_FRAME_WIDTH, EXT_WIDTH)
        external_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, EXT_HEIGHT)
    else:
        external_cap = None

    circle_x, circle_y = EXT_CX, EXT_CY
    calibrated = False

    cv2.namedWindow("Eye Camera")
    cv2.moveWindow("Eye Camera", 50, 600)

    if external_cap is not None:
        cv2.namedWindow("External Camera (Gaze)")
        cv2.moveWindow("External Camera (Gaze)", 720, 50)

    print("Controls: 'c' = calibrate, 'l' = load calibration, 'q' = quit, space = pause")

    while True:
        ret_eye, eye_frame = eye_cap.read()
        if not ret_eye:
            break

        eye_frame = cv2.flip(eye_frame, 0)
        cv2.imshow("Original Eye Frame", eye_frame)
        process_frame(eye_frame)

        if calib_collecting:
            tick_capture()

        if external_cap is not None:
            ret_ext, ext_frame = external_cap.read()
            if ret_ext:
                ext_frame_resized = cv2.resize(ext_frame, (EXT_WIDTH, EXT_HEIGHT))

                if calib_state > 0:
                    target = calib_points_screen[calib_state - 1]
                    cv2.circle(ext_frame_resized, target, 15, (0, 0, 255), -1)
                    for i, pt in enumerate(calib_points_screen):
                        if i < len(calib_vectors_eye):
                            cv2.circle(ext_frame_resized, pt, 5, (0, 200, 0), -1)
                        elif i != calib_state - 1:
                            cv2.circle(ext_frame_resized, pt, 5, (100, 100, 100), -1)

                    status_text = f"Point {calib_state}/{calib_total_points}"
                    if calib_collecting:
                        progress = len(calib_collect_frames)
                        status_text += f" - collecting [{progress}/{CALIB_SAMPLES}]"
                        cv2.circle(ext_frame_resized, target, 20, (0, 165, 255), 3)
                    else:
                        status_text += " - press 'c'"
                    cv2.putText(ext_frame_resized, status_text,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif calibrated:
                    update_gaze_circle_from_current_gaze()
                    cv2.circle(ext_frame_resized, (circle_x, circle_y), 8, (0, 255, 0), -1)

                cv2.imshow("External Camera (Gaze)", ext_frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('l'):
            load_calibration()
        elif key == ord('c'):
            if calib_state == 0:
                start_calibration()
            elif not calib_collecting:
                begin_capture()

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
