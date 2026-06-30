"""Calibrate the rigid extrinsic transform between the eye cam and scene cam.

THE PROBLEM
  The two cameras face opposite directions and never see the same target, so
  ordinary stereo calibration doesn't apply. Instead the calibration jig holds
  two ChArUco boards at a fixed (but not precisely known) relative pose: the eye
  cam sees the board on one panel, the scene cam sees the board on the other.

THE METHOD
  This is a robot-world / hand-eye problem. With the two cameras rigidly linked
  to each other and the two boards rigidly linked to each other, presenting the
  rig at many varied poses lets cv2.calibrateRobotWorldHandEye recover BOTH
  unknown constant transforms at once:
    - eye<->scene camera extrinsic   (what we want)
    - board<->board transform        (recovered for free; sanity-checked below)
  Crucially we do NOT need to know where the boards sit on the panels. The jig
  only has to be RIGID, not measured. This is the same "placement doesn't
  matter, just move it around" property that makes the jig work in the first
  place.

  Per captured pair, for each camera:
    ChArUco detect -> board.matchImagePoints -> solvePnP -> board pose (mm).
  Then:
    world2cam   := eye-cam pose of board A     (R/t_world2cam)
    base2gripper:= scene-cam pose of board B    (R/t_base2gripper)
    calibrateRobotWorldHandEye(...) ->
        gripper2cam  == T_eye_scene  (maps scene-cam coords -> eye-cam coords)
        base2world   == T_A_B        (board B -> board A; ||t|| ~ panel sep)

WORKFLOW
  1. Mount the glasses in the jig so the eye cam sees one board and the scene
     cam the other. Refocus the eye cam onto its panel (see witness-mark note
     in the README).
  2. Run this script with both camera indices and both intrinsics files.
  3. Both feeds show with detections overlaid. SPACE captures a pair (only when
     BOTH boards are detected with enough corners). Between captures, VARY the
     rig pose — tilt and rotate it about different axes, not just slide it. Pure
     translation is degenerate for hand-eye; rotation diversity is what makes
     the solve well-conditioned. Aim for 12+ varied poses.
  4. C computes. R resets. Q quits.

OUTPUT (npz)
  T_eye_scene / R_eye_scene / t_eye_scene : maps scene-cam coords -> eye-cam.
  T_scene_eye / R_scene_eye / t_scene_eye : maps eye-cam coords -> scene-cam
      (this is the one gaze projection wants: a gaze ray in the eye-cam frame
       transformed into the scene-cam frame).
  board_to_board : recovered T_A_B (4x4); ||translation|| should be close to
      your measured panel inner-face separation — a sanity check independent of
      the camera baseline.
  residual_rot_deg / residual_trans_mm : per-pair consistency of the solve.
"""
import argparse
import math
import sys
import time

import cv2
import numpy as np

from scripts.eyetracker.cameras.opencv_source import CameraSettings, OpenCVCamera

# Board presets — MUST match scripts/extras/.../generate_charuco_boards.py
# (square size in mm, page-fit grid, and marker-ID offset per size). Keep in
# sync if that generator changes. marker = square * 0.72.
MARKER_RATIO = 0.72
DICT_NAME = "DICT_5X5_1000"
BOARD_PRESETS = {
    # name:    (squares_x, squares_y, square_mm, id_offset)
    "small":  (13, 17, 15.0, 0),
    "medium": (9, 11, 22.0, 200),
    "large":  (6, 8, 30.0, 400),
}

MIN_CORNERS_PER_FRAME = 6
MIN_PAIRS = 8
LOW_ROTATION_WARN_DEG = 20.0

OUTPUT_PATH = "scripts/eyetracker/extrinsics_eye_scene.npz"
WINDOW_NAME = "calibrate_extrinsics"
PREVIEW_H = 480


def build_board(name: str):
    if name not in BOARD_PRESETS:
        sys.exit(f"unknown board preset '{name}' (choices: {list(BOARD_PRESETS)})")
    sx, sy, square_mm, id_off = BOARD_PRESETS[name]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    n_markers = (sx * sy) // 2
    ids = np.arange(id_off, id_off + n_markers, dtype=np.int32)
    board = cv2.aruco.CharucoBoard(
        (sx, sy), square_mm, square_mm * MARKER_RATIO, dictionary, ids
    )
    return board, cv2.aruco.CharucoDetector(board)


def load_intrinsics(path: str, label: str):
    try:
        data = np.load(path)
    except FileNotFoundError:
        sys.exit(f"{label} intrinsics not found: {path}\n"
                 f"Point --{label}-intrinsics at your saved K/dist npz.")
    if "K" not in data or "dist" not in data:
        sys.exit(f"{label} intrinsics {path} missing 'K'/'dist' keys")
    return np.asarray(data["K"], float), np.asarray(data["dist"], float)


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


def compute_and_save(pairs, out_path, board_b_to_a_sep_hint):
    Rw2c = [cv2.Rodrigues(p["eye_rvec"])[0] for p in pairs]
    tw2c = [p["eye_tvec"].ravel() for p in pairs]
    Rb2g = [cv2.Rodrigues(p["scene_rvec"])[0] for p in pairs]
    tb2g = [p["scene_tvec"].ravel() for p in pairs]

    div = max_pairwise_rotation([p["eye_rvec"] for p in pairs])
    print(f"\nPose diversity (max pairwise eye-board rotation): {div:.1f}°")
    if div < LOW_ROTATION_WARN_DEG:
        print(f"  WARNING: low rotation diversity (<{LOW_ROTATION_WARN_DEG:.0f}°). "
              f"Recapture with more varied tilts/rotations — the solve is "
              f"ill-conditioned otherwise.")

    best = None
    for method, mname in [
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
        for p in pairs:
            T_eye_A_meas = to_T(cv2.Rodrigues(p["eye_rvec"])[0], p["eye_tvec"])
            T_scene_B = to_T(cv2.Rodrigues(p["scene_rvec"])[0], p["scene_tvec"])
            T_eye_A_pred = T_eye_scene @ T_scene_B @ T_B_A
            dR = T_eye_A_pred[:3, :3].T @ T_eye_A_meas[:3, :3]
            rot_errs.append(rotation_angle_deg(dR))
            trans_errs.append(np.linalg.norm(
                T_eye_A_pred[:3, 3] - T_eye_A_meas[:3, 3]))
        score = float(np.mean(rot_errs)) + float(np.mean(trans_errs)) / 10.0
        cand = dict(method=mname, T_eye_scene=T_eye_scene, T_A_B=T_A_B,
                    rot_errs=np.array(rot_errs), trans_errs=np.array(trans_errs),
                    score=score)
        print(f"  {mname}: mean residual {np.mean(rot_errs):.3f}° / "
              f"{np.mean(trans_errs):.2f} mm")
        if best is None or score < best["score"]:
            best = cand

    T_eye_scene = best["T_eye_scene"]
    T_scene_eye = inv_T(T_eye_scene)
    T_A_B = best["T_A_B"]
    baseline = np.linalg.norm(T_eye_scene[:3, 3])
    inter_angle = rotation_angle_deg(T_eye_scene[:3, :3])
    recovered_sep = np.linalg.norm(T_A_B[:3, 3])

    print(f"\nUsing {best['method']} (lower residual).")
    print(f"  camera baseline (||t||):      {baseline:.2f} mm")
    print(f"  inter-camera rotation:        {inter_angle:.2f}°")
    print(f"  recovered board separation:   {recovered_sep:.2f} mm", end="")
    if board_b_to_a_sep_hint:
        print(f"  (you measured ~{board_b_to_a_sep_hint:.0f} mm)")
        if abs(recovered_sep - board_b_to_a_sep_hint) > 0.15 * board_b_to_a_sep_hint:
            print("  WARNING: recovered separation is >15% off your measured "
                  "panel spacing. Check board square sizes (mm) and intrinsics.")
    else:
        print()
    print(f"  residual: {best['rot_errs'].mean():.3f}° / "
          f"{best['trans_errs'].mean():.2f} mm mean, "
          f"{best['rot_errs'].max():.3f}° / {best['trans_errs'].max():.2f} mm max")

    np.savez(
        out_path,
        T_eye_scene=T_eye_scene,
        R_eye_scene=T_eye_scene[:3, :3], t_eye_scene=T_eye_scene[:3, 3],
        T_scene_eye=T_scene_eye,
        R_scene_eye=T_scene_eye[:3, :3], t_scene_eye=T_scene_eye[:3, 3],
        board_to_board=T_A_B,
        baseline_mm=baseline,
        inter_camera_deg=inter_angle,
        recovered_separation_mm=recovered_sep,
        residual_rot_deg=best["rot_errs"],
        residual_trans_mm=best["trans_errs"],
        method=best["method"],
        n_pairs=len(pairs),
        timestamp=time.time(),
    )
    print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eye-cam-index", type=int, default=0)
    parser.add_argument("--scene-cam-index", type=int, default=1)
    parser.add_argument("--eye-intrinsics",
                        default="scripts/eyetracker/eye_intrinsics.npz")
    parser.add_argument("--scene-intrinsics",
                        default="scripts/eyetracker/scene_intrinsics.npz")
    parser.add_argument("--eye-board", default="small",
                        choices=list(BOARD_PRESETS),
                        help="ChArUco preset on the panel the EYE cam sees")
    parser.add_argument("--scene-board", default="large",
                        choices=list(BOARD_PRESETS),
                        help="ChArUco preset on the panel the SCENE cam sees")
    parser.add_argument("--separation-mm", type=float, default=None,
                        help="measured inner-face panel separation (mm), for a "
                             "sanity check against the recovered value")
    parser.add_argument("--out", default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.eye_board == args.scene_board:
        print(f"NOTE: both panels set to '{args.eye_board}'. That's fine (each "
              f"camera only sees its own panel), but using two different sizes "
              f"matched to each camera usually detects better.")

    eye_K, eye_dist = load_intrinsics(args.eye_intrinsics, "eye")
    scene_K, scene_dist = load_intrinsics(args.scene_intrinsics, "scene")
    eye_board, eye_det = build_board(args.eye_board)
    scene_board, scene_det = build_board(args.scene_board)

    eye_cam = OpenCVCamera(args.eye_cam_index, CameraSettings())
    scene_cam = OpenCVCamera(args.scene_cam_index, CameraSettings())
    if not eye_cam.open():
        sys.exit(f"Could not open eye cam at index {args.eye_cam_index}")
    if not scene_cam.open():
        sys.exit(f"Could not open scene cam at index {args.scene_cam_index}")

    print(f"eye cam {args.eye_cam_index}: {eye_cam.width}x{eye_cam.height}, "
          f"board '{args.eye_board}'")
    print(f"scene cam {args.scene_cam_index}: {scene_cam.width}x{scene_cam.height}, "
          f"board '{args.scene_board}'")
    print("\nSPACE = capture pair | C = compute | R = reset | Q = quit")
    print("Between captures, VARY the rig pose (tilt/rotate about different "
          "axes). Aim for 12+ poses.\n")

    pairs = []
    while True:
        eye_frame = eye_cam.read()
        scene_frame = scene_cam.read()
        if eye_frame is None or scene_frame is None:
            continue

        eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        scene_gray = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2GRAY)
        eye_rvec, eye_tvec, eye_n = solve_board_pose(
            eye_det, eye_board, eye_gray, eye_K, eye_dist)
        scene_rvec, scene_tvec, scene_n = solve_board_pose(
            scene_det, scene_board, scene_gray, scene_K, scene_dist)

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
            compute_and_save(pairs, args.out, args.separation_mm)
            break

    eye_cam.release()
    scene_cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
