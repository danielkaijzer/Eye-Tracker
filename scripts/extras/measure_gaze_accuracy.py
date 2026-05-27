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

Camera intrinsics use the standard OpenCV symbols (read from
scene_intrinsics.npz); they recur throughout this file:
    K       3x3 camera intrinsic matrix.
    fx      focal length in pixels, K[0, 0].
    cx, cy  principal point in pixels, K[0, 2] / K[1, 2] — the pixel the
            optical axis pierces, i.e. "straight ahead" in the scene image.
            Eccentricity is measured outward from here.

Run:  python -m scripts.extras.measure_gaze_accuracy
      python -m scripts.extras.measure_gaze_accuracy --val scripts/eyetracker/validation_*.npz
"""
import argparse
import datetime
import glob
import math

import numpy as np

from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper


CALIBRATION_PATH = "scripts/eyetracker/calibration_pupil.npz"
INTRINSICS_PATH = "scripts/eyetracker/scene_intrinsics.npz"

# Eccentricity bin edges in degrees; last bin is open-ended.
ECCENTRICITY_BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, float("inf")]


def _infer_degree(calibration) -> int:
    """Saved cals from before the degree-3 refactor don't have poly_degree.
    Fall back to inferring from the coefficient-vector length so this script
    works on both old and new files."""
    if "poly_degree" in calibration.files:
        return int(calibration["poly_degree"])
    n_coeffs = len(calibration["poly_coeffs_x"])
    return {6: 2, 10: 3}.get(n_coeffs, 2)


def _bin_label(bin_index: int) -> str:
    low, high = ECCENTRICITY_BIN_EDGES[bin_index], ECCENTRICITY_BIN_EDGES[bin_index + 1]
    return f">{low:g}°" if high == float("inf") else f"{low:g}-{high:g}°"


def _bin_index(eccentricity_deg: float) -> int:
    for i in range(len(ECCENTRICITY_BIN_EDGES) - 1):
        if ECCENTRICITY_BIN_EDGES[i] <= eccentricity_deg < ECCENTRICITY_BIN_EDGES[i + 1]:
            return i
    return len(ECCENTRICITY_BIN_EDGES) - 2


def _eccentricity_deg(scene_points: np.ndarray, fx: float,
                      cx: float, cy: float) -> np.ndarray:
    """Angular distance (degrees) of each scene point from the principal point
    (cx, cy), via the pinhole model with focal length fx in pixels."""
    radius_px = np.hypot(scene_points[:, 0] - cx, scene_points[:, 1] - cy)
    return np.degrees(np.arctan(radius_px / fx))


def _predict_errors(mapper: PolynomialGazeMapper,
                    pupil: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Pixel error between predicted and labelled scene point per sample."""
    pred_pts = np.array([mapper.predict((float(p[0]), float(p[1]))) for p in pupil])
    return np.hypot(pred_pts[:, 0] - scene[:, 0], pred_pts[:, 1] - scene[:, 1])


def _print_eccentricity_table(title: str, eccentricity_deg: np.ndarray,
                              errors_px: np.ndarray, fx: float) -> None:
    """Print mean/max error per eccentricity band. fx (focal length, px)
    converts pixel error to degrees."""
    print(title)
    print(f"  {'band':>8} {'n':>4} {'mean':>9} {'mean°':>7} "
          f"{'max':>9} {'max°':>7}")
    errors_px = np.asarray(errors_px, dtype=float)
    for i in range(len(ECCENTRICITY_BIN_EDGES) - 1):
        in_band = np.array([_bin_index(angle) == i for angle in eccentricity_deg])
        count = int(in_band.sum())
        if count == 0:
            continue
        band_errors = errors_px[in_band]
        mean_deg = math.degrees(math.atan(float(band_errors.mean()) / fx))
        max_deg = math.degrees(math.atan(float(band_errors.max()) / fx))
        print(f"  {_bin_label(i):>8} {count:>4} {band_errors.mean():>7.1f}px {mean_deg:>6.2f}° "
              f"{band_errors.max():>7.1f}px {max_deg:>6.2f}°")
    mean_deg = math.degrees(math.atan(float(errors_px.mean()) / fx))
    max_deg = math.degrees(math.atan(float(errors_px.max()) / fx))
    print(f"  {'overall':>8} {len(errors_px):>4} {errors_px.mean():>7.1f}px "
          f"{mean_deg:>6.2f}° {errors_px.max():>7.1f}px {max_deg:>6.2f}°")
    print()


def _coverage_report(pupil_vectors: np.ndarray, scene_points: np.ndarray,
                     fx: float, cx: float, cy: float) -> None:
    """Print pupil-pixel span and the eccentricity histogram of the fit set.
    fx / (cx, cy) are the focal length and principal point (see module doc)."""
    print("Coverage (fit set):")
    print(f"  pupil x: {pupil_vectors[:, 0].min():7.1f} .. {pupil_vectors[:, 0].max():7.1f} px"
          f"   y: {pupil_vectors[:, 1].min():7.1f} .. {pupil_vectors[:, 1].max():7.1f} px")
    eccentricity = _eccentricity_deg(scene_points, fx, cx, cy)
    counts = np.zeros(len(ECCENTRICITY_BIN_EDGES) - 1, dtype=int)
    for angle in eccentricity:
        counts[_bin_index(angle)] += 1
    histogram = "  ".join(f"{_bin_label(i)}:{counts[i]}"
                          for i in range(len(counts)) if counts[i])
    print(f"  gaze eccentricity: {eccentricity.min():.1f}° .. {eccentricity.max():.1f}°   [{histogram}]")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cal", default=CALIBRATION_PATH)
    parser.add_argument("--intr", default=INTRINSICS_PATH)
    parser.add_argument("--val", nargs="*", default=[],
                        help="Held-out validation_*.npz file(s) to evaluate the "
                             "fit against (globs allowed).")
    args = parser.parse_args()

    calibration = np.load(args.cal)
    intrinsics = np.load(args.intr)

    # Camera intrinsics — see module docstring for the K / fx / cx / cy notation.
    K = intrinsics["K"]
    fx = float(K[0, 0])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    scene_width = int(calibration["scene_width"])
    scene_height = int(calibration["scene_height"])
    session_time = datetime.datetime.fromtimestamp(float(calibration["timestamp"]))
    pupil_vectors = calibration["vectors"]
    scene_points = calibration["scene_points"]
    screen_points = calibration["screen_points"]
    degree = _infer_degree(calibration)

    print("=" * 72)
    print(f"Calibration session: {session_time}")
    print(f"Scene resolution:    {scene_width}x{scene_height}")
    print(f"fx / principal:      {fx:.1f} px / ({cx:.1f}, {cy:.1f})")
    print(f"Polynomial degree:   {degree}")
    print()

    _coverage_report(pupil_vectors, scene_points, fx, cx, cy)

    mapper = PolynomialGazeMapper(degree=degree)
    report = mapper.fit(pupil_vectors, scene_points)
    avg_deg = math.degrees(math.atan(report.loo_avg_err / fx))
    max_deg = math.degrees(math.atan(report.loo_max_err / fx))
    print(f"In-sample LOO over {report.n_points} fit points: "
          f"avg {report.loo_avg_err:.1f}px ({avg_deg:.2f}°), "
          f"max {report.loo_max_err:.1f}px ({max_deg:.2f}°)")
    eccentricity_fit = _eccentricity_deg(scene_points, fx, cx, cy)
    _print_eccentricity_table("In-sample LOO error by eccentricity:",
                              eccentricity_fit, report.per_point_errs, fx)

    # Evaluate the (full-data) fit against any held-out validation sets.
    val_files = [f for pat in args.val for f in sorted(glob.glob(pat))]
    for val_file in val_files:
        val_data = np.load(val_file, allow_pickle=True)
        val_pupil = val_data["val_vectors"]
        val_scene = val_data["val_scene_points"]
        errors = _predict_errors(mapper, val_pupil, val_scene)
        eccentricity = _eccentricity_deg(val_scene, fx, cx, cy)
        print("-" * 72)
        print(f"Held-out validation: {val_file} ({len(val_pupil)} points)")
        _print_eccentricity_table("Held-out error by eccentricity:",
                                  eccentricity, errors, fx)

    if not val_files:
        print("(no --val files given; pass validation_*.npz to measure "
              "held-out / wide-angle accuracy)")

    print("Worst in-sample points (sorted worst → best):")
    print(f"  {'pt':>3} {'px':>9} {'deg':>8}   screen_px         pupil_px")
    point_errors = report.per_point_errs
    order = sorted(range(len(point_errors)), key=lambda i: -point_errors[i])
    for i in order[:10]:
        error_px = float(point_errors[i])
        error_deg = math.degrees(math.atan(error_px / fx))
        screen_pt = tuple(int(x) for x in screen_points[i])
        pupil_pt = (float(pupil_vectors[i, 0]), float(pupil_vectors[i, 1]))
        print(f"  {i:>3} {error_px:>9.2f} {error_deg:>7.3f}°   {screen_pt}   ({pupil_pt[0]:.1f}, {pupil_pt[1]:.1f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
