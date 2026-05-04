"""Capture ChArUco frames from the scene cam and solve for intrinsics.

Workflow:
1. Run scripts/extras/generate_charuco_board.py and display its output
   fullscreen on a screen the scene cam can see (second monitor is easiest).
2. Run this script. Preview window shows the cam feed with detected markers
   and ChArUco corners overlaid.
3. Vary scene-cam pose (close/far, tilted, rotated, board near image corners).
   Press SPACE on each pose to capture. Aim for 15+ captures across varied
   distances and angles.
4. Press C to compute calibration. Press R to clear captures. Q to quit.

Output: scripts/eyetracker/scene_intrinsics.npz with K, dist, image_size,
reproj_rms, timestamp. Use K[0,0] (fx) to convert pixel error to angular:
    angular_deg = degrees(atan(pixel_err / fx))
"""
import argparse
import math
import sys
import time

import cv2
import numpy as np

from scripts.eyetracker.cameras.opencv_source import CameraSettings, OpenCVCamera
from scripts.eyetracker.config import SCENE_REQUEST_HEIGHT, SCENE_REQUEST_WIDTH

DICT_NAME = "DICT_5X5_100"
SQUARES_X = 10
SQUARES_Y = 7
SQUARE_LEN = 1.0
MARKER_LEN = 0.75

MIN_CORNERS_PER_FRAME = 8
MIN_FRAMES_FOR_CALIBRATION = 10

OUTPUT_PATH = "scripts/eyetracker/scene_intrinsics.npz"
WINDOW_NAME = "calibrate_scene_intrinsics"


def _build_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary,
    )


def _print_K_summary(K, image_size):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size
    hfov = math.degrees(2 * math.atan(w / (2 * fx)))
    vfov = math.degrees(2 * math.atan(h / (2 * fy)))
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  HFOV={hfov:.2f}°   VFOV={vfov:.2f}°")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-index", type=int, default=0,
                        help="OpenCV camera index for the scene cam")
    parser.add_argument("--out", default=OUTPUT_PATH)
    args = parser.parse_args()

    board = _build_board()
    detector = cv2.aruco.CharucoDetector(board)

    cam = OpenCVCamera(args.cam_index, CameraSettings(
        request_width=SCENE_REQUEST_WIDTH,
        request_height=SCENE_REQUEST_HEIGHT,
    ))
    if not cam.open():
        sys.exit(f"Could not open camera at index {args.cam_index}")

    print(f"Scene cam: {cam.width}x{cam.height}")
    print(f"Board:     {SQUARES_X}x{SQUARES_Y} squares, {DICT_NAME}")
    print()
    print("SPACE = capture | C = calibrate | R = reset | Q = quit")
    print("Vary pose: close/far, tilted, rotated, board near image corners.")
    print()

    captures = []
    image_size = None

    while True:
        frame = cam.read()
        if frame is None:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
        n_corners = 0 if ch_corners is None else len(ch_corners)

        viz = frame.copy()
        if m_ids is not None and len(m_ids) > 0:
            cv2.aruco.drawDetectedMarkers(viz, m_corners, m_ids)
        if ch_corners is not None and ch_ids is not None and len(ch_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(viz, ch_corners, ch_ids)

        status = f"corners: {n_corners:>3d}    captures: {len(captures):>2d}"
        cv2.putText(viz, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, viz)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            if n_corners < MIN_CORNERS_PER_FRAME:
                print(f"  skipped: {n_corners} corners (need {MIN_CORNERS_PER_FRAME}+)")
            else:
                captures.append((ch_corners, ch_ids))
                print(f"  captured frame {len(captures)} ({n_corners} corners)")
        elif k == ord("r"):
            captures.clear()
            print("  captures reset")
        elif k == ord("c"):
            if len(captures) < MIN_FRAMES_FOR_CALIBRATION:
                print(f"  need {MIN_FRAMES_FOR_CALIBRATION}+ captures, have {len(captures)}")
                continue
            print(f"Calibrating with {len(captures)} frames...")
            corners_list = [c for c, _ in captures]
            ids_list = [i for _, i in captures]
            rms, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                corners_list, ids_list, board, image_size, None, None,
            )
            print(f"  reproj RMS: {rms:.4f} px")
            _print_K_summary(K, image_size)
            np.savez(args.out,
                     K=K, dist=dist,
                     image_width=image_size[0],
                     image_height=image_size[1],
                     reproj_rms=rms,
                     timestamp=time.time())
            print(f"  saved -> {args.out}")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
