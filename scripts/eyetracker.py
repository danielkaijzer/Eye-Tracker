"""Pupil-only eye tracker.

Uses the Pupil Labs `pupil_detectors.Detector2D` (dark-pupil, 2014 paper)
for detection, and a 2nd-degree bivariate polynomial to map pupil center
to scene-camera coordinates.

A hand-rolled reference port lives in `eyetracker_pupil.py`, along with
`docs/pupil_detector_port_notes.md` — that file is for future port work
and is not imported from here.

"""
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
calib_state = 0
calib_total_points = 12

calib_collecting = False
calib_collect_frames = []
CALIB_SAMPLES = 15
CALIB_INLIERS = 10
CALIB_STD_THRESH = 12.0
CALIB_WARMUP = 5
CALIB_MAX_RETRIES = 3
_calib_warmup_remaining = 0
_calib_retries_at_point = 0

poly_coeffs_x = None
poly_coeffs_y = None

HIGH_FPS_MODE = False

last_pupil_center = None
last_confidence = 0.0

_pupil_detector_lib = None

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

EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2

circle_x = EXT_CX
circle_y = EXT_CY

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


def _get_pupil_detector():
    global _pupil_detector_lib
    if _pupil_detector_lib is None:
        from pupil_detectors import Detector2D
        _pupil_detector_lib = Detector2D()
    return _pupil_detector_lib


def detect_pupil(gray_frame):
    """Returns (center, ellipse, confidence).

    center: (int, int) for cv2.circle
    ellipse: ((cx, cy), (minor, major), angle_deg) for cv2.ellipse
             (the library already shifts angle by -90° to OpenCV's convention)
    """
    result = _get_pupil_detector().detect(gray_frame)
    conf = float(result["confidence"])
    if conf <= 0.0:
        return None, None, 0.0
    ell = result["ellipse"]
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

    Polynomial outputs gaze in screen pixels; we rescale proportionally into
    the scene-cam window for display.
    """
    global circle_x, circle_y
    if not calibrated or last_pupil_center is None:
        return
    if poly_coeffs_x is None or poly_coeffs_y is None:
        return
    if screen_width is None or screen_height is None:
        return

    px, py = last_pupil_center[0], last_pupil_center[1]
    feat = _build_poly_features(px, py)

    u_screen = feat @ poly_coeffs_x
    v_screen = feat @ poly_coeffs_y

    u = u_screen * EXT_WIDTH / screen_width
    v = v_screen * EXT_HEIGHT / screen_height

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
             coord_space="screen",
             screen_width=screen_width,
             screen_height=screen_height,
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
    global screen_width, screen_height
    path = _calibration_path()
    if not os.path.exists(path):
        print("No saved calibration found.")
        return False
    data = np.load(path, allow_pickle=True)
    if "coord_space" not in data.files or str(data["coord_space"]) != "screen":
        print("  ERROR: saved calibration predates the screen-coord flow. Recalibrate with 'c'.")
        return False
    poly_coeffs_x = data['poly_coeffs_x']
    poly_coeffs_y = data['poly_coeffs_y']
    screen_width = int(data['screen_width'])
    screen_height = int(data['screen_height'])
    ts = float(data['timestamp'])
    age_hrs = (time.time() - ts) / 3600
    calibrated = True
    print(f"Calibration loaded (age: {age_hrs:.1f}h, screen={screen_width}x{screen_height}).")
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
    global calib_collecting, calib_collect_frames, calib_total_points
    global screen_width, screen_height
    global calib_session_dir, calib_labels_path
    global calib_pending_rows, calib_pending_image_paths
    global calib_tk_root, calib_tk_canvas, _tk_key_queue

    calib_state = 1
    pupil_buffer.clear()
    calib_vectors_eye = []
    calib_collecting = False
    calib_collect_frames = []
    calib_pending_rows = []
    calib_pending_image_paths = []
    _tk_key_queue = []

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

    calib_tk_root.bind("<KeyPress-c>", lambda e: _on_tk_key("c"))
    calib_tk_root.bind("<KeyPress-q>", lambda e: _on_tk_key("q"))
    calib_tk_root.focus_force()
    calib_tk_root.update()

    margin_x = 120
    margin_y = 120
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
        if i < len(calib_vectors_eye):
            r = 8
            calib_tk_canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="#00b400", outline="")
        elif i == active_idx:
            r = 20
            calib_tk_canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="red", outline="")
            if calib_collecting:
                rr = 30
                calib_tk_canvas.create_oval(x - rr, y - rr, x + rr, y + rr,
                                            outline="#ffa500", width=3)
        else:
            r = 6
            calib_tk_canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill="#505050", outline="")

    status = f"Point {calib_state}/{calib_total_points}"
    if calib_collecting:
        status += f" - collecting [{len(calib_collect_frames)}/{CALIB_SAMPLES}]"
    else:
        status += " - press 'c' to capture, 'q' to quit"
    calib_tk_canvas.create_text(40, 40, text=status, fill="white",
                                anchor="nw", font=("Courier", 20))


def begin_capture():
    global calib_collecting, calib_collect_frames, _calib_warmup_remaining
    calib_collecting = True
    calib_collect_frames = []
    _calib_warmup_remaining = CALIB_WARMUP


def tick_capture():
    """Collect samples for the current target. Returns True when one point is done."""
    global calib_collecting, calib_collect_frames, calib_state, calibrated
    global calib_pending_rows, calib_pending_image_paths
    global _calib_warmup_remaining
    if not calib_collecting:
        return False
    if last_pupil_center is None or last_eye_frame is None:
        return False
    if _calib_warmup_remaining > 0:
        _calib_warmup_remaining -= 1
        return False

    fixation_idx = calib_state - 1
    sample_idx = len(calib_collect_frames)
    img_name = f"fix{fixation_idx:02d}_sample{sample_idx:02d}.png"
    img_path = os.path.join(calib_session_dir, img_name)
    cv2.imwrite(img_path, last_eye_frame)

    target = calib_points_screen[fixation_idx]
    px = float(last_pupil_center[0])
    py = float(last_pupil_center[1])
    calib_pending_rows.append([
        img_name, fixation_idx, target[0], target[1],
        f"{px:.3f}", f"{py:.3f}", f"{last_confidence:.4f}", f"{time.time():.3f}",
    ])
    calib_pending_image_paths.append(img_path)

    calib_collect_frames.append(last_pupil_center.copy())
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
        return False

    median_vec = np.median(inliers, axis=0)
    calib_vectors_eye.append(median_vec)
    with open(calib_labels_path, "a", newline="") as f:
        csv.writer(f).writerows(calib_pending_rows)
    calib_pending_rows = []
    calib_pending_image_paths = []
    calib_collecting = False
    calib_collect_frames = []

    if calib_state >= calib_total_points:
        if compute_polynomial_calibration():
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
    cv2.moveWindow("Eye Camera", 50, 50)

    if external_cap is not None:
        cv2.namedWindow("External Camera (Gaze)")
        cv2.moveWindow("External Camera (Gaze)", 720, 50)

    print("Controls: 'c' = calibrate, 'l' = load calibration, 'q' = quit, space = pause")

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
                ext_frame_resized = cv2.resize(ext_frame, (EXT_WIDTH, EXT_HEIGHT))

                if calibrated and calib_state == 0:
                    update_gaze_circle_from_current_gaze()
                    cv2.circle(ext_frame_resized, (circle_x, circle_y), 8, (0, 255, 0), -1)

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
