"""Capture ChArUco frames from the eye cam and solve for its intrinsics.

Mirrors calibrate_scene_intrinsics.py for the eye camera. Frames go through the
app's own eye-camera settings (mode, flip) and the same 4:3 crop + resize to
640x480 (`cameras/utils.py`) before detection, so K matches the frames the
pupil detector sees and the ones calibrate_extrinsics.py solves on.

IMPORTANT: calibrate at the focus you'll use. Refocusing the M12 lens changes
the intrinsics, so if you refocus the eye cam onto its jig board for
calibrate_extrinsics.py, calibrate here at that same focus and don't touch the
lens in between.

Workflow:
1. Print a board (`generate_charuco_board.py --board small`, laser printer: the
   eye cam sees in IR, where some inkjet inks are nearly invisible).
2. Run this script. The preview shows the feed with detections overlaid.
3. Vary the board pose: close/far, tilted, rotated, near the image corners.
   Press SPACE on each pose to capture. Aim for 15+ varied captures.
4. Press C to compute. R to reset. Q to quit.

Output: scripts/eyetracker/eye_intrinsics.json with K, dist, image_size,
reproj_rms, timestamp (same format as scene_intrinsics.json).
"""
import argparse
import json
import math
import sys
import time

import cv2

from scripts.eyetracker.calibration.paths import eye_intrinsics_path
from scripts.eyetracker.cameras.discovery import detect_cameras, eye_first
from scripts.eyetracker.cameras.opencv_source import OpenCVCamera
from scripts.eyetracker.cameras.utils import crop_to_aspect_ratio
from scripts.eyetracker.config import EYE_UVC_ID
from scripts.extras.charuco_boards import PRINT_BOARDS, build_print_board, build_screen_board

MIN_CORNERS_PER_FRAME = 8
MIN_FRAMES_FOR_CALIBRATION = 10

OUTPUT_PATH = eye_intrinsics_path()
WINDOW_NAME = "calibrate_eye_intrinsics"


def _print_K_summary(K, image_size):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size
    hfov = math.degrees(2 * math.atan(w / (2 * fx)))
    vfov = math.degrees(2 * math.atan(h / (2 * fy)))
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  HFOV={hfov:.2f}°   VFOV={vfov:.2f}°")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-index", type=int, default=None,
                        help="OpenCV index of the eye cam (default: found by USB id "
                             "on Linux, else the first camera)")
    parser.add_argument("--board", default="small", choices=["screen", *PRINT_BOARDS],
                        help="ChArUco board to detect (see generate_charuco_board.py)")
    parser.add_argument("--out", default=OUTPUT_PATH)
    args = parser.parse_args()

    # The app's eye-camera settings, so this calibrates the frames the app uses.
    from scripts.eyetracker.__main__ import _eye_cam_settings

    board = build_screen_board() if args.board == "screen" else build_print_board(args.board)
    detector = cv2.aruco.CharucoDetector(board)

    cam_index = args.cam_index
    if cam_index is None:
        cameras = eye_first(detect_cameras(), EYE_UVC_ID)
        cam_index = cameras[0] if cameras else 0
    cam = OpenCVCamera(cam_index, _eye_cam_settings())
    if not cam.open():
        sys.exit(f"Could not open camera at index {cam_index}")

    print(f"Eye cam {cam_index}: {cam.width}x{cam.height} native, "
          f"calibrating the cropped/resized frames")
    print(f"Board:   {args.board}")
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
        frame = crop_to_aspect_ratio(frame)
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
            with open(args.out, "w") as f:
                json.dump({
                    "K": K.tolist(),
                    "dist": dist.tolist(),
                    "image_width": image_size[0],
                    "image_height": image_size[1],
                    "reproj_rms": float(rms),
                    "board": args.board,
                    "timestamp": time.time(),
                }, f, indent=2)
            print(f"  saved -> {args.out}")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
