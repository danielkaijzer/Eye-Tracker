"""Recompute gaze tracker accuracy from saved calibration + intrinsics.

Loads scripts/eyetracker/calibration_pupil.npz and scene_intrinsics.npz,
refits the polynomial (deterministic — same lstsq as live calibration), and
reports error in pixels and degrees. Two views:

- In-sample: leave-one-out error over the fit points, binned by eccentricity
  (angular distance from the scene-cam principal point). Because head-on
  calibration clusters points near scene-center, the inner bins dominate.
- Held-out: pass one or more `validation_*.npz` files (captured with 'v',
  ideally at steep head angles). The fit is evaluated on data it never saw,
  binned by eccentricity — this is what quantifies accuracy *outside* the
  calibrated region.

Run:  python -m scripts.extras.measure_gaze_accuracy
      python -m scripts.extras.measure_gaze_accuracy --val scripts/eyetracker/validation_*.npz
"""
import argparse
import datetime
import glob
import math

import numpy as np

from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper


CAL_PATH = "scripts/eyetracker/calibration_pupil.npz"
INTR_PATH = "scripts/eyetracker/scene_intrinsics.npz"

# Eccentricity bin edges in degrees; last bin is open-ended.
BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, float("inf")]


def _infer_degree(cal) -> int:
    """Saved cals from before the degree-3 refactor don't have poly_degree.
    Fall back to inferring from the coefficient-vector length so this script
    works on both old and new files."""
    if "poly_degree" in cal.files:
        return int(cal["poly_degree"])
    n_coeffs = len(cal["poly_coeffs_x"])
    return {6: 2, 10: 3}.get(n_coeffs, 2)


def _bin_label(i: int) -> str:
    lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
    return f">{lo:g}°" if hi == float("inf") else f"{lo:g}-{hi:g}°"


def _bin_index(ecc_deg: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= ecc_deg < BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2


def _eccentricity_deg(pts: np.ndarray, fx: float, cx: float, cy: float) -> np.ndarray:
    """Angular distance of each scene point from the principal point."""
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    return np.degrees(np.arctan(r / fx))


def _predict_errors(mapper: PolynomialGazeMapper,
                    pupil: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Pixel error between predicted and labelled scene point per sample."""
    preds = np.array([mapper.predict((float(p[0]), float(p[1]))) for p in pupil])
    return np.hypot(preds[:, 0] - scene[:, 0], preds[:, 1] - scene[:, 1])


def _print_ecc_table(title: str, ecc_deg: np.ndarray,
                     err_px: np.ndarray, fx: float) -> None:
    print(title)
    print(f"  {'band':>8} {'n':>4} {'mean':>9} {'mean°':>7} "
          f"{'max':>9} {'max°':>7}")
    err_px = np.asarray(err_px, dtype=float)
    for i in range(len(BIN_EDGES) - 1):
        mask = np.array([_bin_index(e) == i for e in ecc_deg])
        n = int(mask.sum())
        if n == 0:
            continue
        e = err_px[mask]
        mean_deg = math.degrees(math.atan(float(e.mean()) / fx))
        max_deg = math.degrees(math.atan(float(e.max()) / fx))
        print(f"  {_bin_label(i):>8} {n:>4} {e.mean():>7.1f}px {mean_deg:>6.2f}° "
              f"{e.max():>7.1f}px {max_deg:>6.2f}°")
    mean_deg = math.degrees(math.atan(float(err_px.mean()) / fx))
    max_deg = math.degrees(math.atan(float(err_px.max()) / fx))
    print(f"  {'overall':>8} {len(err_px):>4} {err_px.mean():>7.1f}px "
          f"{mean_deg:>6.2f}° {err_px.max():>7.1f}px {max_deg:>6.2f}°")
    print()


def _coverage_report(V: np.ndarray, S: np.ndarray,
                     fx: float, cx: float, cy: float) -> None:
    print("Coverage (fit set):")
    print(f"  pupil x: {V[:, 0].min():7.1f} .. {V[:, 0].max():7.1f} px"
          f"   y: {V[:, 1].min():7.1f} .. {V[:, 1].max():7.1f} px")
    ecc = _eccentricity_deg(S, fx, cx, cy)
    counts = np.zeros(len(BIN_EDGES) - 1, dtype=int)
    for e in ecc:
        counts[_bin_index(e)] += 1
    hist = "  ".join(f"{_bin_label(i)}:{counts[i]}"
                     for i in range(len(counts)) if counts[i])
    print(f"  gaze eccentricity: {ecc.min():.1f}° .. {ecc.max():.1f}°   [{hist}]")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", default=CAL_PATH)
    parser.add_argument("--intr", default=INTR_PATH)
    parser.add_argument("--val", nargs="*", default=[],
                        help="Held-out validation_*.npz file(s) to evaluate the "
                             "fit against (globs allowed).")
    args = parser.parse_args()

    cal = np.load(args.cal)
    intr = np.load(args.intr)

    K = intr["K"]
    fx = float(K[0, 0])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    sw = int(cal["scene_width"])
    sh = int(cal["scene_height"])
    ts = datetime.datetime.fromtimestamp(float(cal["timestamp"]))
    V = cal["vectors"]
    S = cal["scene_points"]
    SP = cal["screen_points"]
    degree = _infer_degree(cal)

    print("=" * 72)
    print(f"Calibration session: {ts}")
    print(f"Scene resolution:    {sw}x{sh}")
    print(f"fx / principal:      {fx:.1f} px / ({cx:.1f}, {cy:.1f})")
    print(f"Polynomial degree:   {degree}")
    print()

    _coverage_report(V, S, fx, cx, cy)

    mapper = PolynomialGazeMapper(degree=degree)
    report = mapper.fit(V, S)
    avg_deg = math.degrees(math.atan(report.loo_avg_err / fx))
    max_deg = math.degrees(math.atan(report.loo_max_err / fx))
    print(f"In-sample LOO over {report.n_points} fit points: "
          f"avg {report.loo_avg_err:.1f}px ({avg_deg:.2f}°), "
          f"max {report.loo_max_err:.1f}px ({max_deg:.2f}°)")
    ecc_fit = _eccentricity_deg(S, fx, cx, cy)
    _print_ecc_table("In-sample LOO error by eccentricity:",
                     ecc_fit, report.per_point_errs, fx)

    # Evaluate the (full-data) fit against any held-out validation sets.
    val_files = [f for pat in args.val for f in sorted(glob.glob(pat))]
    for vf in val_files:
        vd = np.load(vf, allow_pickle=True)
        Vv = vd["val_vectors"]
        Sv = vd["val_scene_points"]
        err = _predict_errors(mapper, Vv, Sv)
        ecc = _eccentricity_deg(Sv, fx, cx, cy)
        print("-" * 72)
        print(f"Held-out validation: {vf} ({len(Vv)} points)")
        _print_ecc_table("Held-out error by eccentricity:", ecc, err, fx)

    if not val_files:
        print("(no --val files given; pass validation_*.npz to measure "
              "held-out / wide-angle accuracy)")

    print("Worst in-sample points (sorted worst → best):")
    print(f"  {'pt':>3} {'px':>9} {'deg':>8}   screen_px         pupil_px")
    errs = report.per_point_errs
    order = sorted(range(len(errs)), key=lambda i: -errs[i])
    for i in order[:10]:
        e_px = float(errs[i])
        e_deg = math.degrees(math.atan(e_px / fx))
        sp = tuple(int(x) for x in SP[i])
        vv = (float(V[i, 0]), float(V[i, 1]))
        print(f"  {i:>3} {e_px:>9.2f} {e_deg:>7.3f}°   {sp}   ({vv[0]:.1f}, {vv[1]:.1f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
