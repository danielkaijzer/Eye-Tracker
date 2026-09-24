"""ChArUco board definitions shared by the board generator and the calibration
scripts. The generator and the scripts that detect a board must build it from
the same spec, so every board is defined once, here.

Two families:

- SCREEN: shown fullscreen on a monitor for scene-cam intrinsics
  (calibrate_scene_intrinsics.py). Unitless squares, since intrinsic
  calibration doesn't depend on the board's absolute scale. DICT_5X5_100.
- PRINT_BOARDS ("tiny" / "small" / "medium" / "large"): printed at exact metric
  size on US Letter for the dual-camera calibration jig. Each tiles as many
  squares as fit on the page (up to an optional cap) and uses its own marker-ID
  range, so boards of different sizes never collide. DICT_5X5_1000, so all ID
  ranges must stay below 1000.

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
    # name:    (square_mm, id_offset, max_squares (x, y) or None = fill the page)
    "small":  (15.0, 0, None),
    "medium": (22.0, 200, None),
    "large":  (30.0, 400, None),
    # For the eye cam at its eye working distance (~45-50 mm), so neither eye
    # intrinsics nor extrinsics need a refocus. There it sees roughly 50 x 40 mm,
    # about 10 x 8 of these squares. Capped because a full page would need ~990
    # marker IDs; 30 x 24 (150 x 120 mm, IDs 500-859) still leaves room to rotate.
    "tiny":   (5.0, 500, (30, 24)),
}


def build_screen_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, SCREEN_DICT_NAME))
    return cv2.aruco.CharucoBoard(
        (SCREEN_SQUARES_X, SCREEN_SQUARES_Y),
        SCREEN_SQUARE_LEN, SCREEN_MARKER_LEN, dictionary,
    )


def print_grid(name: str) -> tuple:
    """(squares_x, squares_y) for a printed board: what fits inside the US
    Letter margins, capped at the board's max_squares."""
    square_mm, _, max_squares = PRINT_BOARDS[name]
    usable_w = PAGE_W_MM - 2 * MARGIN_SIDE_MM
    usable_h = PAGE_H_MM - MARGIN_TOP_MM - MARGIN_BOTTOM_MM
    squares_x, squares_y = math.floor(usable_w / square_mm), math.floor(usable_h / square_mm)
    if max_squares is not None:
        squares_x, squares_y = min(squares_x, max_squares[0]), min(squares_y, max_squares[1])
    return squares_x, squares_y


def build_print_board(name: str) -> cv2.aruco.CharucoBoard:
    """Printed jig board by name (a PRINT_BOARDS key). Lengths are in mm."""
    square_mm, id_offset, _ = PRINT_BOARDS[name]
    squares_x, squares_y = print_grid(name)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, PRINT_DICT_NAME))
    n_markers = (squares_x * squares_y) // 2
    ids = np.arange(id_offset, id_offset + n_markers, dtype=np.int32)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, square_mm * PRINT_MARKER_RATIO,
        dictionary, ids,
    )
