import cv2
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time

gaze_buffer = [] # Moving average buffer
screen_buffer = []
BUFFER_SIZE = 5  # Small enough to be responsive, large enough to stop jitter

calib_points_screen = []  # [(x,y), ...] in screen coords
calib_vectors_eye = []    # [np.array([dx, dy]), ...] PCCR vectors
calib_state = 0           # 0: Idle, >0: current target index (1-based)
calib_total_points = 12   # 3x4 grid

# Multi-sample collection state
calib_collecting = False
calib_collect_frames = []
CALIB_SAMPLES = 15        # frames to collect per point
CALIB_STD_THRESH = 3.0    # max std dev for a valid capture (pixel-space PCCR vectors)

# Polynomial calibration coefficients (6 per axis)
poly_coeffs_x = None
poly_coeffs_y = None

# Toggle for 120Hz resolution/blurring logic
HIGH_FPS_MODE = False

# --- PCCR globals ---
last_pccr_vector = None       # np.array([dx, dy])
last_pupil_center = None      # (x, y)
last_glint_centroid = None    # (x, y)

GLINT_SEARCH_RADIUS = 80     # pixels around pupil to search
GLINT_MIN_AREA = 3            # min contour area for a glint blob
GLINT_MAX_AREA = 150          # max contour area
NUM_GLINTS = 2                # expected LED count

pccr_buffer = []              # recent valid PCCR vectors for median
PCCR_BUFFER_SIZE = 5          # frames of history
PCCR_JUMP_THRESH = 30.0       # max pixel jump from running median to accept

calibrated = False

# External camera / screen params (for 640x480)
EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2

circle_x = EXT_CX
circle_y = EXT_CY

# Function to detect available cameras
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

def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (20, 20))
    margin = 20
    roi = blurred[margin:-margin, margin:-margin]
    min_loc = cv2.minMaxLoc(roi)[3]
    return (min_loc[0] + margin, min_loc[1] + margin)

def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2
    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    all_contours = np.concatenate(contours[0], axis=0)
    spacing = int(len(all_contours)/25)
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)

    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]

        vec1 = prev_point - current_point
        vec2 = next_point - current_point

        with np.errstate(invalid='ignore'):
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        vec_to_centroid = centroid - current_point
        cos_threshold = np.cos(np.radians(60))

        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)

    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour
    return [largest_contour] if largest_contour is not None else []

def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, color, 2)
        return image
    else:
        return image

def check_contour_pixels(contour, image_shape, debug_mode_on):
    if len(contour) < 5:
        return [0, 0, None]

    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)

    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)

    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10)
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4)

    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)

    absolute_pixel_total_thick = np.sum(overlap_thick > 0)
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)

    total_border_pixels = np.sum(contour_mask > 0)
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0

    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0]
    if len(contour) < 5:
        return ellipse_goodness

    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)

    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))

    if ellipse_area == 0:
        return ellipse_goodness

    ellipse_goodness[0] = covered_pixels / ellipse_area
    axes_lengths = ellipse[1]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])

    return ellipse_goodness

def _detect_pupil_adaptive(gray_frame, darkest_point):
    """Fallback pupil detection using adaptive thresholding for IR washout."""
    h, w = gray_frame.shape[:2]

    # ROI size scales with frame dimensions (~40% of shorter dimension)
    roi_half = int(min(h, w) * 0.2)
    dx, dy = darkest_point
    x1 = max(0, dx - roi_half)
    y1 = max(0, dy - roi_half)
    x2 = min(w, dx + roi_half)
    y2 = min(h, dy + roi_half)
    roi = gray_frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    # Adaptive threshold detects local contrast edges (works when pupil isn't darkest)
    block = 15 if min(roi.shape[:2]) > 30 else 7
    binary = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=block, C=5
    )

    # Morphological close to fill glint holes inside pupil
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area bounds scale with frame resolution
    frame_area = h * w
    min_area = frame_area * 0.002
    max_area = frame_area * 0.05

    best_contour = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area or len(cnt) < 5:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.4 and area > best_area:
            best_area = area
            best_contour = cnt

    if best_contour is None:
        return None, None

    ellipse = cv2.fitEllipse(best_contour)
    # Shift ellipse center back to full-frame coordinates
    center_x = int(ellipse[0][0]) + x1
    center_y = int(ellipse[0][1]) + y1
    ellipse = ((center_x, center_y), ellipse[1], ellipse[2])
    return (center_x, center_y), ellipse


def detect_pupil(frame, gray_frame):
    """Detect pupil center and ellipse using multi-threshold approach."""
    darkest_point = get_darkest_area(frame)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

    thresholded_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
    thresholded_strict = mask_outside_square(thresholded_strict, darkest_point, 250)
    thresholded_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    thresholded_medium = mask_outside_square(thresholded_medium, darkest_point, 250)
    thresholded_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
    thresholded_relaxed = mask_outside_square(thresholded_relaxed, darkest_point, 250)

    image_array = [thresholded_relaxed, thresholded_medium, thresholded_strict]

    # Dynamic Elliptical Kernel
    k_size = int(gray_frame.shape[1] * 0.025)
    k_size = k_size if k_size % 2 != 0 else k_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    best_goodness = 0
    best_contours = []

    for thresh_img in image_array:
        dilated = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced) > 0 and len(reduced[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated, reduced[0], False)
            total_pixels = check_contour_pixels(reduced[0], dilated.shape, False)
            final_goodness = current_goodness[0] * total_pixels[0] * total_pixels[0] * total_pixels[1]

            if final_goodness > 0 and final_goodness > best_goodness:
                best_goodness = final_goodness
                best_contours = reduced

    if not best_contours:
        return _detect_pupil_adaptive(gray_frame, darkest_point)

    optimized = optimize_contours_by_angle(best_contours, gray_frame)
    if optimized is None or len(optimized) < 5:
        return None, None

    ellipse = cv2.fitEllipse(optimized)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    return center, ellipse

def detect_glints(gray_frame, pupil_center, search_radius=GLINT_SEARCH_RADIUS):
    """Detect IR glints (corneal reflections) near the pupil.

    Returns (glint_centroid, glint_points) or (None, None).
    glint_centroid is the average position of detected glints.
    glint_points is a list of individual glint centers.
    """
    h, w = gray_frame.shape[:2]
    px, py = pupil_center

    # Extract square ROI around pupil, clamped to image bounds
    x1 = max(0, px - search_radius)
    y1 = max(0, py - search_radius)
    x2 = min(w, px + search_radius)
    y2 = min(h, py + search_radius)
    roi = gray_frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    # Threshold at top ~2% brightness within ROI
    thresh_val = np.percentile(roi, 98)
    _, binary = cv2.threshold(roi, int(thresh_val), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and compute centroids
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if GLINT_MIN_AREA <= area <= GLINT_MAX_AREA:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1  # convert to full-frame coords
                cy = int(M["m01"] / M["m00"]) + y1
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                candidates.append((cx, cy, dist))

    if not candidates:
        return None, None

    # Sort by distance to pupil, take closest NUM_GLINTS
    candidates.sort(key=lambda c: c[2])
    selected = candidates[:NUM_GLINTS]
    glint_points = [(c[0], c[1]) for c in selected]

    # Centroid is average of selected glints
    centroid_x = np.mean([p[0] for p in glint_points])
    centroid_y = np.mean([p[1] for p in glint_points])
    glint_centroid = (centroid_x, centroid_y)

    return glint_centroid, glint_points

def smooth_pccr_vector(raw_vector):
    """Reject outlier PCCR vectors that jump too far from the running median.

    Returns the vector if accepted, or None if rejected.
    """
    global pccr_buffer

    raw = np.array(raw_vector, dtype=float)

    if len(pccr_buffer) == 0:
        pccr_buffer.append(raw)
        return raw

    median = np.median(pccr_buffer, axis=0)
    dist = np.linalg.norm(raw - median)

    if dist > PCCR_JUMP_THRESH:
        return None

    pccr_buffer.append(raw)
    if len(pccr_buffer) > PCCR_BUFFER_SIZE:
        pccr_buffer.pop(0)

    return raw

def process_frame(frame):
    """PCCR pipeline: detect pupil, detect glints, compute pupil-glint vector."""
    global last_pccr_vector, last_pupil_center, last_glint_centroid

    frame = crop_to_aspect_ratio(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if HIGH_FPS_MODE:
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    pupil_center, pupil_ellipse = detect_pupil(frame, gray_frame)

    if pupil_center is not None:
        last_pupil_center = pupil_center
        # Draw pupil
        cv2.circle(frame, pupil_center, 4, (0, 255, 0), -1)
        if pupil_ellipse is not None:
            cv2.ellipse(frame, pupil_ellipse, (20, 255, 255), 2)

        glint_centroid, glint_points = detect_glints(gray_frame, pupil_center)

        if glint_centroid is not None:
            last_glint_centroid = glint_centroid
            # Draw glints
            for gp in glint_points:
                cv2.circle(frame, gp, 3, (255, 150, 0), -1)

            # Compute PCCR vector
            dx = pupil_center[0] - glint_centroid[0]
            dy = pupil_center[1] - glint_centroid[1]
            raw_pccr = np.array([dx, dy])

            accepted = smooth_pccr_vector(raw_pccr)
            if accepted is not None:
                last_pccr_vector = accepted

                # Draw PCCR vector (cyan line from glint centroid through pupil)
                gc = (int(glint_centroid[0]), int(glint_centroid[1]))
                end_pt = (int(pupil_center[0] + dx), int(pupil_center[1] + dy))
                cv2.line(frame, gc, end_pt, (255, 255, 0), 2)

                # Display PCCR vector text
                pccr_text = f"PCCR: ({dx:.1f}, {dy:.1f})"
                cv2.putText(frame, pccr_text, (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Rejected outlier — hold last valid, show indicator
                cv2.circle(frame, (15, 15), 8, (0, 0, 255), -1)
                cv2.putText(frame, "SKIP", (28, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # No glints detected — don't update last_pccr_vector (hold-last-valid)
            last_glint_centroid = None
    else:
        # No pupil detected — don't update last_pccr_vector (hold-last-valid)
        last_pupil_center = None
        last_glint_centroid = None

    cv2.imshow("Eye Camera", frame)

def update_gaze_circle_from_current_gaze():
    global circle_x, circle_y, last_pccr_vector, calibrated, screen_buffer
    if not calibrated or last_pccr_vector is None:
        return
    if poly_coeffs_x is None or poly_coeffs_y is None:
        return

    dx, dy = last_pccr_vector[0], last_pccr_vector[1]
    feat = _build_poly_features(dx, dy)

    u = feat @ poly_coeffs_x
    v = feat @ poly_coeffs_y

    screen_buffer.append((u, v))
    if len(screen_buffer) > BUFFER_SIZE:
        screen_buffer.pop(0)

    avg_u = np.mean([p[0] for p in screen_buffer])
    avg_v = np.mean([p[1] for p in screen_buffer])

    circle_x = int(np.clip(avg_u, 0, EXT_WIDTH - 1))
    circle_y = int(np.clip(avg_v, 0, EXT_HEIGHT - 1))

def _build_poly_features(gx, gy):
    """Build 2nd-degree polynomial feature row: [1, gx, gy, gx^2, gy^2, gx*gy]"""
    return np.array([1.0, gx, gy, gx*gx, gy*gy, gx*gy])

def compute_polynomial_calibration():
    """Fit 2nd-degree polynomial from PCCR vectors to screen coords (least-squares)."""
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

    cx, res_x, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, res_y, _, _ = np.linalg.lstsq(A, by, rcond=None)
    poly_coeffs_x = cx
    poly_coeffs_y = cy

    # Leave-one-out cross-validation
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
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data.npz")

def _history_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_history.npz")

def _save_calibration():
    """Save polynomial coefficients and raw calibration data."""
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

    # Append to history for future ML training
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
    """Load saved polynomial calibration. Returns True on success."""
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
    global calib_state, calib_points_screen, calib_vectors_eye, calib_collecting, calib_collect_frames
    global calib_total_points, pccr_buffer
    calib_state = 1
    pccr_buffer.clear()
    calib_vectors_eye = []
    calib_collecting = False
    calib_collect_frames = []

    # Generate 3x4 grid of calibration targets with margins
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
    """Start multi-sample collection for the current calibration point."""
    global calib_collecting, calib_collect_frames
    calib_collecting = True
    calib_collect_frames = []

def tick_capture():
    """Called each frame during collection. Returns True when done collecting."""
    global calib_collecting, calib_collect_frames, calib_state, calibrated
    if not calib_collecting:
        return False
    if last_pccr_vector is None:
        return False

    calib_collect_frames.append(last_pccr_vector.copy())
    if len(calib_collect_frames) < CALIB_SAMPLES:
        return False

    # Collection complete — compute median and check quality
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

# Process video from the selected eye camera + external camera preview
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

        # Tick multi-sample collection if active
        if calib_collecting:
            tick_capture()

        if external_cap is not None:
            ret_ext, ext_frame = external_cap.read()
            if ret_ext:
                ext_frame_resized = cv2.resize(ext_frame, (EXT_WIDTH, EXT_HEIGHT))

                if calib_state > 0:
                    target = calib_points_screen[calib_state - 1]
                    cv2.circle(ext_frame_resized, target, 15, (0, 0, 255), -1)
                    # Draw small dots for all targets
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

    tk.Label(root, text="PCCR Eye Tracker", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select Camera:").pack(pady=5)

    selected_camera = tk.StringVar()
    selected_camera.set(str(cameras[0]) if cameras else "No cameras found")

    camera_dropdown = ttk.Combobox(root, textvariable=selected_camera, values=[str(cam) for cam in cameras])
    camera_dropdown.pack(pady=5)

    tk.Button(root, text="Start Camera", command=lambda: [root.destroy(), process_camera()]).pack(pady=5)
    tk.Button(root, text="Browse Video", command=lambda: [root.destroy(), process_video()]).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    selection_gui()
