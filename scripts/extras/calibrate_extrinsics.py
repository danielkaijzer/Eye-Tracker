"""Calibrate the rigid extrinsic transform between the eye cam and scene cam.

THE PROBLEM
  The two cameras face opposite directions and never see the same target, so
  ordinary stereo calibration doesn't apply. Instead the calibration jig (three
  stainless steel sheets at ~90°, see 3d-files/README.md) holds two ChArUco
  boards at a fixed but unmeasured relative pose: the eye cam sees the board on
  one sheet, the scene cam sees the board on another.

THE METHOD
  This is a robot-world / hand-eye problem. With the two cameras rigidly linked
  to each other (the headset) and the two boards rigidly linked to each other
  (the jig), presenting the headset at many varied poses lets
  cv2.calibrateRobotWorldHandEye recover BOTH unknown constant transforms:
    - eye<->scene camera extrinsic   (what we want)
    - board<->board transform        (recovered for free; sanity-checked below)
  We do NOT need to know where the boards sit or how square the sheets are. The
  jig only has to be RIGID.

  Per captured pair, for each camera:
    ChArUco detect -> board.matchImagePoints -> solvePnP -> board pose (mm).
  Then, in OpenCV's notation (cTw = cTg · gTb · bTw):
    world2cam    := eye-cam pose of board A
    base2gripper := scene-cam pose of board B
    calibrateRobotWorldHandEye(...) ->
        gripper2cam == T_eye_scene  (maps scene-cam coords -> eye-cam coords)
        base2world  == T_A_B        (maps board-B coords -> board-A coords)

FRAME GEOMETRY
  Both cameras are opened with the app's own settings, and eye frames get the
  app's 4:3 crop + resize to 640x480, so the result applies to the frames the
  app records. The intrinsics files must describe those same frames
  (calibrate_eye_intrinsics.py, calibrate_scene_intrinsics.py); the script
  refuses to run if their image sizes don't match the live frames.

WORKFLOW
  1. Calibrate eye intrinsics (calibrate_eye_intrinsics.py) at the eye focus.
  2. Place the headset so the eye cam sees the tiny board at about eye distance
     (~45-50 mm, no refocus) and the scene cam sees a board on another sheet.
  3. Run this script. Both feeds show with detections overlaid. SPACE captures a
     pair (only when BOTH boards are detected). Hold still for each capture.
     Between captures, ROTATE the headset about different axes, not just slide
     it: pure translation is degenerate for hand-eye. Aim for 12+ poses.
  4. C computes and saves. R resets. Q quits.

OUTPUT
  rig_calibrations/<rig_id>.json (schema in docs/dataset_format.md), including
  `extrinsics_eye_to_scene`: R, t (mm) with p_scene = R @ p_eye + t, which is
  the direction gaze projection needs (eye-cam ray -> scene-cam frame).
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from scripts.eyetracker.calibration.paths import (
    eye_intrinsics_path, rig_calibrations_root, scene_intrinsics_path,
)
from scripts.eyetracker.cameras.discovery import detect_cameras, eye_first, pick_scene_index
from scripts.eyetracker.cameras.opencv_source import OpenCVCamera
from scripts.eyetracker.cameras.utils import crop_to_aspect_ratio
from scripts.eyetracker.config import EYE_UVC_ID, SCENE_UVC_ID
from scripts.extras.charuco_boards import PRINT_BOARDS, build_print_board

MIN_CORNERS_PER_FRAME = 6
MIN_PAIRS = 8
LOW_ROTATION_WARN_DEG = 20.0

WINDOW_NAME = "calibrate_extrinsics"
PREVIEW_H = 480


# ---- intrinsics + board pose ------------------------------------------------

def load_intrinsics(path: str, label: str) -> dict:
    """Read a K/dist JSON written by calibrate_{eye,scene}_intrinsics.py."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{label} intrinsics not found: {path}\n"
                 f"Run calibrate_{label}_intrinsics.py or pass --{label}-intrinsics.")
    missing = [k for k in ("K", "dist", "image_width", "image_height") if k not in data]
    if missing:
        sys.exit(f"{label} intrinsics {path} is missing {missing}")
    return dict(K=np.asarray(data["K"], float),
                dist=np.asarray(data["dist"], float).reshape(-1),
                image_size=(int(data["image_width"]), int(data["image_height"])),
                reproj_rms=data.get("reproj_rms"),
                source=os.path.basename(path))


def solve_board_pose(detector, board, gray, K, dist):
    """Return (rvec, tvec, n_corners) or (None, None, n_corners)."""
    ch_c, ch_id, _, _ = detector.detectBoard(gray)
    n = 0 if ch_c is None else len(ch_c)
    if n < MIN_CORNERS_PER_FRAME:
        return None, None, n
    obj_pts, img_pts = board.matchImagePoints(ch_c, ch_id)
    if obj_pts is None or len(obj_pts) < MIN_CORNERS_PER_FRAME:
        return None, None, n
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None, None, n
    return rvec, tvec, n


# ---- rigid-transform helpers ------------------------------------------------

def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def inv_T(T):
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def rotation_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def max_pairwise_rotation(rvecs):
    """Largest relative rotation among captured eye-board poses (deg). Proxy for
    pose diversity — hand-eye is degenerate without rotation about >1 axis."""
    Rs = [cv2.Rodrigues(r)[0] for r in rvecs]
    worst = 0.0
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            worst = max(worst, rotation_angle_deg(Rs[i].T @ Rs[j]))
    return worst


# ---- solve -------------------------------------------------------------------

def solve_extrinsics(pairs):
    """Solve eye<->scene and board<->board from captured pose pairs.

    Each pair is a dict with eye_rvec/eye_tvec (eye-cam pose of board A) and
    scene_rvec/scene_tvec (scene-cam pose of board B). Runs SHAH and LI and
    keeps the one with the lower per-pair residual.
    """
    Rw2c = [cv2.Rodrigues(p["eye_rvec"])[0] for p in pairs]
    tw2c = [np.asarray(p["eye_tvec"], float).ravel() for p in pairs]
    Rb2g = [cv2.Rodrigues(p["scene_rvec"])[0] for p in pairs]
    tb2g = [np.asarray(p["scene_tvec"], float).ravel() for p in pairs]

    candidates = []
    for method, name in [
        (cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH, "SHAH"),
        (cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI, "LI"),
    ]:
        Rb2w, tb2w, Rg2c, tg2c = cv2.calibrateRobotWorldHandEye(
            Rw2c, tw2c, Rb2g, tb2g, method=method
        )
        T_eye_scene = to_T(Rg2c, tg2c)   # gripper2cam
        T_A_B = to_T(Rb2w, tb2w)         # base2world == board B -> board A
        T_B_A = inv_T(T_A_B)

        # Per-pair residual: predicted eye-board pose vs measured.
        rot_errs, trans_errs = [], []
        for Rm, tm, Rs, ts in zip(Rw2c, tw2c, Rb2g, tb2g):
            T_eye_A_meas = to_T(Rm, tm)
            T_eye_A_pred = T_eye_scene @ to_T(Rs, ts) @ T_B_A
            rot_errs.append(rotation_angle_deg(
                T_eye_A_pred[:3, :3].T @ T_eye_A_meas[:3, :3]))
            trans_errs.append(float(np.linalg.norm(
                T_eye_A_pred[:3, 3] - T_eye_A_meas[:3, 3])))
        candidates.append(dict(
            method=name, T_eye_scene=T_eye_scene, T_A_B=T_A_B,
            rot_errs=np.array(rot_errs), trans_errs=np.array(trans_errs),
            score=float(np.mean(rot_errs)) + float(np.mean(trans_errs)) / 10.0,
        ))

    best = min(candidates, key=lambda c: c["score"])
    best["candidates"] = candidates
    best["pose_diversity_deg"] = max_pairwise_rotation([p["eye_rvec"] for p in pairs])
    best["n_pairs"] = len(pairs)
    return best


def rig_calibration_record(result, eye_intr, scene_intr, eye_board, scene_board,
                           rig_id, notes=None):
    """Build the rig_calibrations/<rig_id>.json record (docs/dataset_format.md)."""
    T_scene_eye = inv_T(result["T_eye_scene"])   # eye-cam coords -> scene-cam coords
    T_A_B = result["T_A_B"]

    def intr(d):
        return {"K": d["K"].tolist(), "dist": d["dist"].tolist(),
                "image_width": d["image_size"][0], "image_height": d["image_size"][1],
                "reproj_rms": d["reproj_rms"], "source": d["source"]}

    return {
        "rig_id": rig_id,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "notes": notes,
        "intrinsics": {"eye": intr(eye_intr), "scene": intr(scene_intr)},
        "extrinsics_eye_to_scene": {
            "R": T_scene_eye[:3, :3].tolist(),
            "t": T_scene_eye[:3, 3].tolist(),
            "units": "mm",
            "method": f"calibrateRobotWorldHandEye/{result['method']}",
            "residual_rot_deg": {"mean": float(result["rot_errs"].mean()),
                                 "max": float(result["rot_errs"].max())},
            "residual_trans_mm": {"mean": float(result["trans_errs"].mean()),
                                  "max": float(result["trans_errs"].max())},
        },
        "capture": {
            "n_pairs": result["n_pairs"],
            "pose_diversity_deg": float(result["pose_diversity_deg"]),
            "eye_board": eye_board,
            "scene_board": scene_board,
            "board_b_to_a": {"R": T_A_B[:3, :3].tolist(), "t": T_A_B[:3, 3].tolist()},
        },
    }


def _report(result, board_distance_hint):
    for c in result["candidates"]:
        print(f"  {c['method']}: mean residual {c['rot_errs'].mean():.3f}° / "
              f"{c['trans_errs'].mean():.2f} mm")
    div = result["pose_diversity_deg"]
    print(f"\nPose diversity (max pairwise eye-board rotation): {div:.1f}°")
    if div < LOW_ROTATION_WARN_DEG:
        print(f"  WARNING: low rotation diversity (<{LOW_ROTATION_WARN_DEG:.0f}°). "
              "Recapture with more varied tilts/rotations; the solve is "
              "ill-conditioned otherwise.")

    T_eye_scene = result["T_eye_scene"]
    board_distance = float(np.linalg.norm(result["T_A_B"][:3, 3]))
    print(f"\nUsing {result['method']} (lower residual).")
    print(f"  camera baseline (||t||):      {np.linalg.norm(T_eye_scene[:3, 3]):.2f} mm")
    print(f"  inter-camera rotation:        {rotation_angle_deg(T_eye_scene[:3, :3]):.2f}°")
    print(f"  recovered board distance:     {board_distance:.2f} mm", end="")
    if board_distance_hint:
        print(f"  (you measured ~{board_distance_hint:.0f} mm)")
        if abs(board_distance - board_distance_hint) > 0.15 * board_distance_hint:
            print("  WARNING: >15% off your measurement. Check the board presets "
                  "and the intrinsics.")
    else:
        print()
    print(f"  residual: {result['rot_errs'].mean():.3f}° / "
          f"{result['trans_errs'].mean():.2f} mm mean, "
          f"{result['rot_errs'].max():.3f}° / {result['trans_errs'].max():.2f} mm max")


# ---- capture UI --------------------------------------------------------------

def _check_size(label, frame, intr):
    size = (frame.shape[1], frame.shape[0])
    if size != intr["image_size"]:
        sys.exit(f"{label} frames are {size[0]}x{size[1]} but {intr['source']} was "
                 f"calibrated at {intr['image_size'][0]}x{intr['image_size'][1]}. "
                 f"Recalibrate {label} intrinsics on the frames the app uses.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eye-cam-index", type=int, default=None,
                        help="default: found by USB id on Linux, else the first camera")
    parser.add_argument("--scene-cam-index", type=int, default=None,
                        help="default: found by USB id on Linux, else the next camera")
    parser.add_argument("--eye-intrinsics", default=eye_intrinsics_path())
    parser.add_argument("--scene-intrinsics", default=scene_intrinsics_path())
    parser.add_argument("--eye-board", default="tiny", choices=list(PRINT_BOARDS),
                        help="ChArUco preset on the sheet the EYE cam sees")
    parser.add_argument("--scene-board", default="large", choices=list(PRINT_BOARDS),
                        help="ChArUco preset on the sheet the SCENE cam sees")
    parser.add_argument("--board-distance-mm", type=float, default=None,
                        help="measured distance between the two boards' origin corners "
                             "(top-left corner of each printed pattern, label side up), "
                             "for a sanity check against the recovered value")
    parser.add_argument("--notes", default=None, help="free text stored in the output")
    parser.add_argument("--out-dir", default=rig_calibrations_root())
    args = parser.parse_args()

    # The app's camera settings, so the result matches the frames it records.
    from scripts.eyetracker.__main__ import _eye_cam_settings, _scene_cam_settings

    eye_intr = load_intrinsics(args.eye_intrinsics, "eye")
    scene_intr = load_intrinsics(args.scene_intrinsics, "scene")
    eye_board = build_print_board(args.eye_board)
    scene_board = build_print_board(args.scene_board)
    eye_det = cv2.aruco.CharucoDetector(eye_board)
    scene_det = cv2.aruco.CharucoDetector(scene_board)

    cameras = eye_first(detect_cameras(), EYE_UVC_ID)
    eye_index = args.eye_cam_index
    if eye_index is None:
        eye_index = cameras[0] if cameras else 0
    scene_index = args.scene_cam_index
    if scene_index is None:
        scene_index = pick_scene_index(eye_index, cameras, SCENE_UVC_ID)

    eye_cam = OpenCVCamera(eye_index, _eye_cam_settings())
    scene_cam = OpenCVCamera(scene_index, _scene_cam_settings())
    if not eye_cam.open():
        sys.exit(f"Could not open eye cam at index {eye_index}")
    if not scene_cam.open():
        sys.exit(f"Could not open scene cam at index {scene_index}")

    print(f"eye cam {eye_index}: {eye_cam.width}x{eye_cam.height} native, "
          f"board '{args.eye_board}'")
    print(f"scene cam {scene_index}: {scene_cam.width}x{scene_cam.height}, "
          f"board '{args.scene_board}'")
    print("\nSPACE = capture pair | C = compute | R = reset | Q = quit")
    print("Hold still for each capture. Between captures, ROTATE the headset "
          "about different axes. Aim for 12+ poses.\n")

    pairs = []
    checked = False
    while True:
        eye_frame = eye_cam.read()
        scene_frame = scene_cam.read()
        if eye_frame is None or scene_frame is None:
            continue
        eye_frame = crop_to_aspect_ratio(eye_frame)
        if not checked:
            _check_size("eye", eye_frame, eye_intr)
            _check_size("scene", scene_frame, scene_intr)
            checked = True

        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        scene_gray = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2GRAY)
        eye_rvec, eye_tvec, eye_n = solve_board_pose(
            eye_det, eye_board, eye_gray, eye_intr["K"], eye_intr["dist"])
        scene_rvec, scene_tvec, scene_n = solve_board_pose(
            scene_det, scene_board, scene_gray, scene_intr["K"], scene_intr["dist"])

        def annotate(frame, n, ok, label):
            viz = frame.copy()
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(viz, f"{label}: {n} corners", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            h = PREVIEW_H
            w = int(viz.shape[1] * h / viz.shape[0])
            return cv2.resize(viz, (w, h))

        eye_ok = eye_rvec is not None
        scene_ok = scene_rvec is not None
        combo = cv2.hconcat([
            annotate(eye_frame, eye_n, eye_ok, "EYE"),
            annotate(scene_frame, scene_n, scene_ok, "SCENE"),
        ])
        cv2.putText(combo, f"pairs: {len(pairs)}  (both green to capture)",
                    (10, PREVIEW_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, combo)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            if eye_ok and scene_ok:
                pairs.append(dict(eye_rvec=eye_rvec, eye_tvec=eye_tvec,
                                  scene_rvec=scene_rvec, scene_tvec=scene_tvec))
                print(f"  captured pair {len(pairs)} "
                      f"(eye {eye_n}, scene {scene_n} corners)")
            else:
                miss = "eye" if not eye_ok else "scene"
                print(f"  skipped: {miss} board not solved this frame")
        elif k == ord("r"):
            pairs.clear()
            print("  pairs reset")
        elif k == ord("c"):
            if len(pairs) < MIN_PAIRS:
                print(f"  need {MIN_PAIRS}+ pairs, have {len(pairs)}")
                continue
            result = solve_extrinsics(pairs)
            _report(result, args.board_distance_mm)
            rig_id = time.strftime("%Y%m%d_%H%M%S")
            record = rig_calibration_record(result, eye_intr, scene_intr,
                                            args.eye_board, args.scene_board,
                                            rig_id, args.notes)
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"{rig_id}.json")
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"  saved -> {out_path}")
            break

    eye_cam.release()
    scene_cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
