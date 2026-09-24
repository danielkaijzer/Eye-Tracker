"""ChArUco board definitions shared by the board generator and the calibration
scripts. The generator and the scripts that detect a board must build it from
the same spec, so every board is defined once, here.

Two families:

- SCREEN: shown fullscreen on a monitor for scene-cam intrinsics
  (calibrate_scene_intrinsics.py). Unitless squares, since intrinsic
  calibration doesn't depend on the board's absolute scale. DICT_5X5_100.
- PRINT_BOARDS ("small" / "medium" / "large"): printed at exact metric size on
  US Letter for the dual-camera calibration jig. Each tiles as many squares as
  fit on the page and uses its own marker-ID range, so boards of different
  sizes never collide. DICT_5X5_1000.

Neither family can be confused with the screen-corner markers (DICT_4X4_50)
the gaze pipeline tracks.
"""
import math

import cv2
import numpy as np

# ---- Screen board (scene-cam intrinsics) ----
SCREEN_DICT_NAME = "DICT_5X5_100"
SCREEN_SQUARES_X = 10
SCREEN_SQUARES_Y = 7
SCREEN_SQUARE_LEN = 1.0
SCREEN_MARKER_LEN = 0.75

# ---- Printed jig boards ----
PRINT_DICT_NAME = "DICT_5X5_1000"
PRINT_MARKER_RATIO = 0.72   # marker edge as a fraction of square edge

PAGE_W_MM = 215.9  # US Letter
PAGE_H_MM = 279.4
# Top margin is taller to fit the 50 mm reference ruler. The board is anchored
# at the bottom, since the cameras are more likely to see the lower part of
# the panel than the top.
MARGIN_TOP_MM = 14.0
MARGIN_SIDE_MM = 8.0
MARGIN_BOTTOM_MM = 8.0

PRINT_BOARDS = {
    # name:    (square_mm, id_offset)
    "small":  (15.0, 0),
    "medium": (22.0, 200),
    "large":  (30.0, 400),
}


def build_screen_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, SCREEN_DICT_NAME))
    return cv2.aruco.CharucoBoard(
        (SCREEN_SQUARES_X, SCREEN_SQUARES_Y),
        SCREEN_SQUARE_LEN, SCREEN_MARKER_LEN, dictionary,
    )


def print_grid(square_mm: float) -> tuple:
    """(squares_x, squares_y) that fit inside the US Letter margins."""
    usable_w = PAGE_W_MM - 2 * MARGIN_SIDE_MM
    usable_h = PAGE_H_MM - MARGIN_TOP_MM - MARGIN_BOTTOM_MM
    return math.floor(usable_w / square_mm), math.floor(usable_h / square_mm)


def build_print_board(name: str) -> cv2.aruco.CharucoBoard:
    """Printed jig board by name ("small" / "medium" / "large"). Lengths are in mm."""
    square_mm, id_offset = PRINT_BOARDS[name]
    squares_x, squares_y = print_grid(square_mm)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, PRINT_DICT_NAME))
    n_markers = (squares_x * squares_y) // 2
    ids = np.arange(id_offset, id_offset + n_markers, dtype=np.int32)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, square_mm * PRINT_MARKER_RATIO,
        dictionary, ids,
    )
