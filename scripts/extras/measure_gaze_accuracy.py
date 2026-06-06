"""Recompute gaze tracker accuracy from a saved session.

Loads a `session_<ts>/` (metadata.json for scene intrinsics + sizes, labels.csv
for the captured samples), refits the polynomial (deterministic — same lstsq as
live calibration), and prints LOO error in both pixels and degrees, plus a
per-point breakdown.

The fit uses the per-fixation median of the raw labels.csv rows, which closely
tracks the live-calibration fit (live uses the median of inlier samples; this
uses the median of all collected samples, so numbers can differ slightly).

Useful for:
- Checking accuracy of past calibration sessions without re-running.
- Spotting bad fixations (sort by error -> outliers float to the top).
- Tracking accuracy over time after hardware/config changes.

Run:  python -m scripts.extras.measure_gaze_accuracy [--session <dir>]
      (default: the most recent session under data/calibration/)
"""
import argparse
import datetime
import math
import sys

from scripts.eyetracker.calibration.paths import dataset_root
from scripts.eyetracker.dataset import load_session, session_dirs
from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper


def _resolve_session(arg: str) -> str:
    if arg:
        return arg
    dirs = session_dirs()
    if not dirs:
        sys.exit(f"No sessions found under {dataset_root()}")
    return dirs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=None,
                        help="session dir to analyze (default: most recent)")
    args = parser.parse_args()

    session = _resolve_session(args.session)
    metadata, df = load_session(session)
    if metadata is None:
        sys.exit(f"{session} has no metadata.json (legacy session) — "
                 "cannot read scene intrinsics.")
    scene_intr = (metadata.get("intrinsics") or {}).get("scene")
    if not scene_intr or scene_intr.get("K") is None:
        sys.exit("Session has no scene intrinsics. Run "
                 "scripts.extras.calibrate_scene_intrinsics first.")

    fx = float(scene_intr["K"][0][0])
    sw = int(metadata["scene_cam"]["width"])
    sh = int(metadata["scene_cam"]["height"])
    ts = datetime.datetime.fromtimestamp(float(metadata["timestamp"]))
    degree = int((metadata.get("fit") or {}).get("degree", 2))

    # Per-fixation median (one point per target), matching the live fit.
    grouped = df.groupby("fixation_id").median(numeric_only=True)
    V = grouped[["pupil_x", "pupil_y"]].to_numpy(dtype=float)
    S = grouped[["scene_target_x", "scene_target_y"]].to_numpy(dtype=float)
    SP = grouped[["x_screen", "y_screen"]].to_numpy(dtype=float)

    print("=" * 68)
    print(f"Session:             {metadata['session_id']}")
    print(f"Captured:            {ts}")
    print(f"Scene resolution:    {sw}x{sh}")
    print(f"fx (from intr):      {fx:.2f} px")
    print(f"Polynomial degree:   {degree}")
    print()
    print(f"scene_points x range: {float(S[:, 0].min()):7.1f} .. {float(S[:, 0].max()):7.1f}  "
          f"({(S[:, 0].max() - S[:, 0].min()) / sw * 100:.1f}% of frame width)")
    print(f"scene_points y range: {float(S[:, 1].min()):7.1f} .. {float(S[:, 1].max()):7.1f}  "
          f"({(S[:, 1].max() - S[:, 1].min()) / sh * 100:.1f}% of frame height)")
    print()

    report = PolynomialGazeMapper(degree=degree).fit(V, S)
    avg_deg = math.degrees(math.atan(report.loo_avg_err / fx))
    max_deg = math.degrees(math.atan(report.loo_max_err / fx))
    print(f"n_points:  {report.n_points}")
    print(f"LOO avg:   {report.loo_avg_err:7.3f} px   ->  {avg_deg:.4f}°")
    print(f"LOO max:   {report.loo_max_err:7.3f} px   ->  {max_deg:.4f}°")
    print()

    print("Per-point error (sorted worst → best):")
    print(f"  {'pt':>3} {'px':>9} {'deg':>8}   screen_px         pupil_px")
    errs = report.per_point_errs
    order = sorted(range(len(errs)), key=lambda i: -errs[i])
    for i in order:
        e_px = float(errs[i])
        e_deg = math.degrees(math.atan(e_px / fx))
        sp = tuple(int(x) for x in SP[i])
        vv = (float(V[i, 0]), float(V[i, 1]))
        print(f"  {i:>3} {e_px:>9.2f} {e_deg:>7.3f}°   {sp}   ({vv[0]:.1f}, {vv[1]:.1f})")
    print("=" * 68)


if __name__ == "__main__":
    main()
