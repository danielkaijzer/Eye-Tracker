# Calibration Coverage and Eccentricity Validation

Why the tracker is accurate when you look near the center of your view and worse
when you look off to the side — how we *measure* that gap, and (in
[`multipose_calibration.md`](multipose_calibration.md)) how we close it.

This doc assumes you've read [`polynomial_gaze_mapping.md`](polynomial_gaze_mapping.md),
which explains the pupil→scene polynomial and the ArUco homography. Here we care
about *where in the user's field of view the fit is trustworthy*, not the fit math.

## The current calibration setup, in one paragraph

The user looks at a grid of red dots drawn on a monitor. Four ArUco markers at the
screen corners let us compute a per-frame **screen→scene homography**, which turns
each dot's screen pixel into its location in the scene-camera image. For each dot we
record one `(pupil_pixel, scene_pixel)` pair — pupil center from the eye camera,
scene pixel from the homography. After the grid we least-squares-fit a degree-2 or
degree-3 bivariate polynomial `pupil → scene` (`scripts/eyetracker/gaze/polynomial.py`).
Two grids ship today: quick (`c`, 4×3, degree 2) and detailed (`d`, 5×4, degree 3,
with worst-point recapture). Grid sizes, margins, and degrees live in
`scripts/eyetracker/config.py` under `CALIB_QUICK_*` / `CALIB_DETAILED_*`.

### What it solves

- **Per-subject, per-rig geometry without modeling it.** The eye is a sphere viewed
  off-axis by the eye camera; the exact pupil→gaze relationship depends on eye shape,
  camera placement, and how the headset sits. The polynomial learns that relationship
  empirically from a dozen-ish points, so we never have to measure kappa, eye radius,
  or camera extrinsics to get a usable gaze estimate.
- **Head-pose invariance at runtime.** The homography absorbs *where the screen is*
  during calibration, so the learned polynomial is purely `pupil → scene-cam`, a
  property of the headset rig. Move your whole body around the room afterward and the
  polynomial still holds (see the "Mental model" section of the polynomial doc).
- **Noise averaging.** More fixations than coefficients (e.g. 20 points for 10
  degree-3 coefficients) means `lstsq` averages out pupil jitter and marker shake
  instead of memorizing it.

### What it does *not* solve — the coverage problem

Every calibration dot lives on the monitor, and the monitor occupies a small patch
near the **center** of the scene camera's view. So:

- The **labels** (scene pixels) cluster in a small central region.
- The **inputs** (pupil positions) only span the eye rotations needed to scan that
  small region — a narrow band of pupil pixels.

A polynomial is reliable *inside* the cloud of data it was fit on and unreliable
outside it. When you later look somewhere the calibration never sampled — past the
edge of where the screen was — the model **extrapolates**, and a degree-3 polynomial's
cubic terms diverge fast. That is the edge degradation: it is a property of *coverage*,
not of slippage or hardware.

> A common misread is "just spread the dots wider on the screen." On a normal monitor
> that barely widens the *pupil* range, because the maximum eye rotation is capped at
> roughly half the screen's angular width. Widening coverage means getting the eye to
> rotate further — which is what [multi-pose calibration](multipose_calibration.md)
> does by moving the head, not the dots.

## Why measure it as *eccentricity*

To know whether a fit is trustworthy "in the center" vs "at the edge," we need a single
number for "how far off-axis is the user looking." That number is **eccentricity**: the
angular distance of a gaze point from the scene camera's optical axis (its principal
point).

For a point at scene pixel `(u, v)` with camera intrinsics `K`:

```
r        = hypot(u - cx, v - cy)          # pixels from the principal point
ecc_deg  = degrees(atan(r / fx))          # angular distance from the optical axis
```

where `fx = K[0,0]` and `(cx, cy) = (K[0,2], K[1,2])`. See `_eccentricity_deg` in
`scripts/extras/measure_gaze_accuracy.py`.

Eccentricity is the right axis to bin by, for three reasons:

1. **It's the variable the polynomial extrapolates along.** Error grows with distance
   from the calibrated cloud, and that cloud is centered on the optical axis. Binning by
   eccentricity exposes exactly the trend that a single average hides.
2. **It's resolution- and lens-independent.** "30 px of error" means different things
   at 720p vs 4K, or with a wide vs narrow lens. Degrees of visual angle are comparable
   across rigs and are the unit accuracy is quoted in (Pupil Labs Neon is ~0.5–1°).
3. **It maps onto how the eye actually works.** The scene camera points roughly along
   head-forward, so scene-frame eccentricity ≈ the eye's rotation away from
   head-forward — i.e. where in the **oculomotor range** the user is. People turn their
   *head* to look far off-axis, so in practice gaze rarely exceeds the comfortable
   oculomotor range (~±20–25°); that range is exactly what we want covered and measured.

   (This is an approximation: it assumes the scene-cam optical axis ≈ head-forward and
   ignores eye↔scene-camera parallax and target depth. Good enough to rank center vs
   edge; see "What could be improved.")

## The validation tooling

A single average error over the calibration points is misleading twice over: the points
are clustered in the center, *and* leave-one-out error only tells you about regions you
calibrated. Two pieces fix this.

### 1. Held-out capture — the `v` key

`v` runs the same grid as the detailed calibration but in **collect-only** mode: it
fits nothing and never touches the live `calibration_pupil.npz` or the in-memory mapper
(`CalibrationRoutine(fit_on_finish=False)` in `scripts/eyetracker/calibration/routine.py`).
On finish it writes the captured `(pupil, scene)` median pairs to a timestamped
`validation_<ts>.npz` (`save_validation` in `calibration/persistence.py`).

The point: capture a `v` set **at a steep head angle** so its gaze points land at high
eccentricity — out where the calibration never sampled. That gives you ground-truth
pairs the fit has never seen, in the region you most want to test.

### 2. Eccentricity-binned measurement

`python -m scripts.extras.measure_gaze_accuracy [--val validation_*.npz]` refits the
polynomial from a saved calibration and reports error binned by eccentricity:

- **In-sample**, using **leave-one-out** error (refit without each point, predict it).
  LOO is used here because these points *were* in the fit; training residuals would
  flatter the fit by measuring how well it memorized its own labels.
- **Held-out**, using the **plain full-data fit** to predict each `--val` point. No LOO
  needed — the validation points were never in the fit, so there's no leakage.
- A **coverage report**: the pupil-pixel range and the eccentricity histogram of the
  calibration labels, so under-coverage is visible before you even look at error.

### Reading the output

A sample run on a head-on detailed calibration (no `--val` yet):

```
Coverage (fit set):
  pupil x:   269.0 ..   377.0 px   y:   211.0 ..   267.0 px
  gaze eccentricity: 2.0° .. 24.5°   [0-5°:1  5-10°:5  10-15°:6  15-20°:7  >20°:1]

In-sample LOO error by eccentricity:
      band    n      mean   mean°       max    max°
      0-5°    1     9.8px   0.30°     9.8px   0.30°
     5-10°    5    11.3px   0.34°    17.6px   0.53°
    10-15°    6    18.5px   0.56°    32.7px   0.99°
    15-20°    7    23.9px   0.72°    73.1px   2.21°
      >20°    1    35.8px   1.08°    35.8px   1.08°
   overall   20    19.0px   0.58°    73.1px   2.21°
```

Two things to notice. First, the **pupil range is tiny** (≈108 px wide, ≈56 px tall) —
the eye barely moved during calibration. Second, error **climbs monotonically with
eccentricity** (0.30° at center → 0.72° by 15–20°), even *within* the calibrated region.
Add a wide-angle `--val` set and the outer bins blow up further, because that's pure
extrapolation. That growth curve, not the single "overall" number, is the thing to drive
down.

## What could be improved

- **Eccentricity is an approximation to true gaze angle.** It's measured in the scene-cam
  frame relative to the principal point, assuming the scene cam looks along head-forward
  and ignoring eye↔scene parallax and target depth. With the planned extrinsics jig (see
  `docs/data_collection.md`) and a depth estimate, this could become a true
  eye-relative visual angle.
- **The polynomial itself is the ceiling.** Widening coverage fixes catastrophic
  extrapolation but a global low-degree polynomial has fixed capacity — fitting a wider,
  more nonlinear domain can trade a little center accuracy for the edges. The real fix is
  higher-capacity or local models, ultimately the planned image-to-gaze approach, which
  this richer, eccentricity-spanning data collection is meant to feed.
- **Validation labels still ride on the ArUco homography**, so a steep-angle `v` set is
  only as good as marker detection at that angle (all four corners must stay in frame).
