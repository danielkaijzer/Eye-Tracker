# Leave-one-out (LOO) error — notes, past regression, and caveats

Working notes on the calibration LOO error: what it's for, a regression that
briefly broke it, and the caveats that are still live. For the conceptual
mapper/LOO walkthrough see [polynomial_gaze_mapping.md](polynomial_gaze_mapping.md).

## Status (2026-06-29)

**LOO is currently implemented correctly.** `PolynomialGazeMapper.fit()`
(`scripts/eyetracker/gaze/polynomial.py:52`) computes a real leave-one-out error
in the loop at lines ~79-88. A quick check confirms it returns non-zero errors on
noisy data (see the regression test at the bottom). This doc is a record so the
regression below doesn't silently return.

## What LOO is and why it matters here

For each calibration point *i*: refit the polynomial on the other *n*−1 points,
predict the held-out point *i*, and measure the reprojection error in scene-cam
pixels. Averaged (and max'd) over all *n* points, plus a per-point array.

It's used for three things — none of which affect the live fit/predict (those use
the full-data `lstsq` coefficients), but all of which are calibration *quality*
machinery:

1. **Recapture targeting** — the detailed routine's two-pass mode picks the
   worst-N fixations to re-prompt via `np.argsort(per_point_errs)`
   (`scripts/eyetracker/calibration/routine.py:435`).
2. **Quality reporting + warning gate** — `_report_and_save` prints
   `LOO error: avg/max` (`routine.py:472`) and warns if `loo_avg_err` exceeds
   ~4% of scene width (`routine.py:478-480`).
3. **Accuracy measurement** — `scripts/extras/measure_gaze_accuracy.py:157-185`
   reports avg/max/per-point LOO in px and degrees.

(`persistence.py` does **not** record LOO metrics — it saves the polynomial
coefficients via `state_dict`. So a LOO regression does not corrupt saved
calibrations, only the reporting/recapture.)

## The regression that happened

At one point `fit()` was reduced to a stub:

```python
errors = np.zeros(n)          # <-- the real refit loop had been removed
return FitReport(..., loo_avg_err=0.0, loo_max_err=0.0, per_point_errs=errors)
```

It slipped in during the degree-2/3 refactor: the intent (a `TODO` at the top of
the file) was to replace the O(n) refit loop with the hat-matrix shortcut, but
only the zeros stub landed and got committed. The fit itself stayed correct, so
tracking worked — the breakage was invisible except in the quality machinery.

### Blast radius while stubbed

- **Recapture re-prompted the wrong points.** `np.argsort([0,0,…,0])` is
  `[0,1,…,n-1]`, so "worst N" was always the *last* N captured fixations
  (row-major → bottom-right of the grid), regardless of actual error. Two-pass
  refinement was effectively choosing arbitrary points. Note `_should_run_pass_two`
  guards on `per_point_errs is None`, but a zeros array isn't `None`, so pass two
  still ran — on garbage selection.
- **Reported accuracy was a lie** — every calibration printed `avg=0.0px,
  max=0.0px`.
- **The high-error warning was dead** — `0 > err_threshold` is never true.
- **`measure_gaze_accuracy.py` reported 0.000px / 0.00°** and all-zero per-point
  errors. Any ~0.39° figure on record predates the stub.

## The fix (now in place)

The explicit refit loop, degree-agnostic (uses the design matrix `A` as built):

```python
errors = np.zeros(n)
for i in range(n):
    A_loo = np.delete(A, i, axis=0)
    cx_loo, *_ = np.linalg.lstsq(A_loo, np.delete(bx, i), rcond=None)
    cy_loo, *_ = np.linalg.lstsq(A_loo, np.delete(by, i), rcond=None)
    errors[i] = math.hypot(A[i] @ cx_loo - bx[i], A[i] @ cy_loo - by[i])
```

## Remaining caveats (still live)

1. **LOO is meaningless when `n == k`** (`k` = feature count: 6 for degree 2, 10
   for degree 3). The refit drops to `n−1 = k−1` rows < `k` columns →
   underdetermined; `lstsq` returns a min-norm solution (no crash) but the LOO
   number is junk. `fit()` only guards `n >= k`, so LOO is only trustworthy at
   `n >= k + 1`. The configured grids are all comfortably above this (quick 9-12
   vs 6, detailed 20 vs 10, multipose 3×12 vs 10), and `_should_run_pass_two`
   keeps `n` above the minimum after a recapture drop — so it doesn't bite today.
   If a smaller grid is ever configured, consider skipping LOO (leave zeros) when
   `n <= k` rather than reporting a misleading value.

2. **Performance: O(n) refits.** Each `fit()` does `2n` extra `lstsq` solves.
   For n ≤ ~36 this is microseconds — not worth optimizing now. The closed-form
   alternative (the old `TODO`) uses the hat matrix `H = A (AᵀA)⁻¹ Aᵀ`: the LOO
   residual is `residual_i / (1 − h_ii)`, so one fit + the leverages `h_ii` gives
   all per-point LOO errors with no refitting. Revisit only if grids get large.

## Guard against recurrence

A minimal regression test (LOO must be non-zero on noisy quadratic data):

```python
import numpy as np
from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper

rng = np.random.default_rng(0)
pupil = rng.uniform(-1, 1, (12, 2))
scene = np.column_stack([
    3 + 2*pupil[:, 0] - pupil[:, 1] + 0.5*pupil[:, 0]**2,
    1 - pupil[:, 0] + 2*pupil[:, 1] + 0.3*pupil[:, 1]**2,
]) + rng.normal(0, 0.05, (12, 2))

rep = PolynomialGazeMapper(degree=2).fit(pupil, scene)
assert rep.loo_avg_err > 0 and np.any(rep.per_point_errs > 0), "LOO is stubbed!"
```
