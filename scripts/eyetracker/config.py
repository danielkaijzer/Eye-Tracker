"""Tunable constants, grouped by concern.

Lives at the package root so any module can `from scripts.eyetracker.config import X`
without crossing concerns. Anything you'd reach for to retune behavior lives here.
"""
import math


# ---- Eye camera --------------------------------------------------------------
# 0.3 MP (640x480) with 80° lens. If the spec turns out to be diagonal FOV
# rather than horizontal, EYE_CAM_FOV_IS_DIAGONAL controls the focal-length math.
EYE_CAM_RESOLUTION = (640, 480)
EYE_CAM_FOV_DEG = 80.0
EYE_CAM_FOV_IS_DIAGONAL = True


def _compute_eye_focal_length_px():
    w, h = EYE_CAM_RESOLUTION
    ref = math.sqrt(w * w + h * h) if EYE_CAM_FOV_IS_DIAGONAL else w
    return (ref / 2.0) / math.tan(math.radians(EYE_CAM_FOV_DEG / 2.0))


EYE_CAM_FOCAL_LENGTH_PX = _compute_eye_focal_length_px()

HIGH_FPS_MODE = False

# Exposure (see "Exposure controls" below). Manual by default so the IR-lit
# pupil stays at a fixed brightness instead of the sensor hunting.
EYE_AUTO_EXPOSURE = False
# USB vendor:product for the uvc-util exposure path (Sonix GC0308).
EYE_UVC_ID = "0x0c45:0x6366"
EYE_GAIN = None  # None = leave at device default; tune live with [ / ]


# ---- Scene camera ------------------------------------------------------------
SCENE_REQUEST_WIDTH = 1920
SCENE_REQUEST_HEIGHT = 1080

# Pin the scene exposure (auto-exposure off) by default so the frame doesn't
# drift as the user looks around — auto-exposure shifts the whole image and
# throws off the ArUco/gaze mapping. Toggle back to auto in-app with 'a'.
SCENE_AUTO_EXPOSURE = False
# USB vendor:product for the uvc-util exposure path (Realtek OV5640).
SCENE_UVC_ID = "0x0bda:0xd565"
SCENE_GAIN = None


# ---- Exposure controls -------------------------------------------------------
# OpenCV/AVFoundation can't set exposure on macOS, so we shell out to uvc-util
# (jtfrey/uvc-util), selecting cameras by the *_UVC_ID above. Build it and put
# it on PATH or set UVC_UTIL_PATH (see requirements.txt). On the current modules
# exposure-time is a cosmetic no-op, so GAIN is the brightness lever and
# auto-exposure-mode is the on/off switch. EXPOSURE_STEP is the per-keypress
# gain nudge for the '[' / ']' hotkeys.
EXPOSURE_STEP = 1.0


# ---- Display window ----------------------------------------------------------
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


# ---- Pupil detection ---------------------------------------------------------
# Library confidence = min(0.99, support/circ) * (support/total_edges)^2.
# Clean pupils typically score > 0.8; 0.25 is permissive.
CONF_THRESH = 0.25

# Outlier rejection on pupil center. A fast saccade to an extreme-gaze
# angle moves the pupil ~200-300 px in a single 30fps frame, so the
# threshold needs to accommodate that — only catches catastrophic flips.
PUPIL_BUFFER_SIZE = 7
PUPIL_JUMP_THRESH = 400.0


# ---- Gaze smoothing (1€ filter) ----------------------------------------------
# min_cutoff: low-speed cutoff (Hz). Lower = steadier fixations, more lag.
# beta: speed coefficient. Higher = tracks saccades faster, less smoothing.
# d_cutoff: cutoff on the speed estimate; rarely needs tuning.
GAZE_MIN_CUTOFF = 1.0
GAZE_BETA = 0.05
GAZE_D_CUTOFF = 1.0


# ---- Calibration -------------------------------------------------------------
CALIB_SAMPLES = 15
CALIB_INLIERS = 10
CALIB_STD_THRESH = 12.0
CALIB_SCENE_STD_THRESH = 10.0
CALIB_WARMUP = 5

# Quick cal: 4x3 grid, degree-2 polynomial.
CALIB_QUICK_ROWS = 3
CALIB_QUICK_COLS = 4
CALIB_QUICK_MARGIN = 220
CALIB_QUICK_DEGREE = 2

# Detailed cal: 5x4 grid (20 pts) degree-3 polynomial (10 coeffs, so 2x
# overdetermined). Smaller margin pushes targets closer to the edges
# (where the polynomial bends hardest); 2-pass recapture trims outliers.
# Margin must clear the ArUco quiet zone + active-target halo (~30px):
# 140 quiet zone + 30 halo + 10 slack = 180.
CALIB_DETAILED_ROWS = 4
CALIB_DETAILED_COLS = 5
CALIB_DETAILED_MARGIN = 180
CALIB_DETAILED_DEGREE = 3
CALIB_DETAILED_RECAPTURE_WORST = 5


# ---- ArUco corner markers ----------------------------------------------------
# IDs: 0=TL, 1=TR, 2=BR, 3=BL.
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_MARKER_PX = 120
ARUCO_QUIET_ZONE_PX = 140
ARUCO_IDS = (0, 1, 2, 3)
