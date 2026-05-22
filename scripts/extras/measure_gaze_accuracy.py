"""Recompute gaze tracker accuracy from saved calibration + intrinsics.

Loads scripts/eyetracker/calibration_pupil.npz and scene_intrinsics.npz,
refits the polynomial (deterministic — same lstsq as live calibration),
and prints LOO error in both pixels and degrees, plus a per-point breakdown.

Useful for:
- Checking accuracy of past calibration sessions without re-running.
- Spotting bad fixations (sort by error → outliers float to the top).
- Tracking accuracy over time after hardware/config changes.

Run:  python -m scripts.extras.measure_gaze_accuracy
"""
import argparse
import datetime
import math

import numpy as np

from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper


CAL_PATH = "scripts/eyetracker/calibration_pupil.npz"
INTR_PATH = "scripts/eyetracker/scene_intrinsics.npz"


def _infer_degree(cal) -> int:
    """Saved cals from before the degree-3 refactor don't have poly_degree.
    Fall back to inferring from the coefficient-vector length so this script
    works on both old and new files."""
    if "poly_degree" in cal.files:
        return int(cal["poly_degree"])
    n_coeffs = len(cal["poly_coeffs_x"])
    return {6: 2, 10: 3}.get(n_coeffs, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", default=CAL_PATH)
    parser.add_argument("--intr", default=INTR_PATH)
    args = parser.parse_args()

    cal = np.load(args.cal)
    intr = np.load(args.intr)

    fx = float(intr["K"][0, 0])
    sw = int(cal["scene_width"])
    sh = int(cal["scene_height"])
    ts = datetime.datetime.fromtimestamp(float(cal["timestamp"]))
    V = cal["vectors"]
    S = cal["scene_points"]
    SP = cal["screen_points"]
    degree = _infer_degree(cal)

    print("=" * 68)
    print(f"Calibration session: {ts}")
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
