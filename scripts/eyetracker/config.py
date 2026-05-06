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


# ---- Scene camera ------------------------------------------------------------
SCENE_REQUEST_WIDTH = 1920
SCENE_REQUEST_HEIGHT = 1080


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


# ---- ArUco corner markers ----------------------------------------------------
# IDs: 0=TL, 1=TR, 2=BR, 3=BL.
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_MARKER_PX = 160
ARUCO_QUIET_ZONE_PX = 200
ARUCO_IDS = (0, 1, 2, 3)
