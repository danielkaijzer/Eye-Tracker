# Polynomial Gaze Mapping

How we turn a pupil pixel from the eye camera into a gaze pixel in the scene camera.

## Goal

Given a pupil center `(px, py)` in eye-camera pixels, predict where in the scene-camera image the user is looking — `(scene_x, scene_y)`.

## Why a polynomial?

The eye is a sphere, the cornea bulges, the eye camera sits at an off-axis angle, and the pupil moves on a curved manifold as gaze sweeps the world. The mapping `(px, py) → (scene_x, scene_y)` is not linear, but over the limited range of pupil motion it is smooth and well-approximated by a low-degree polynomial.

We use a 2nd-degree bivariate polynomial. The feature row (see `scripts/eyetracker/gaze/polynomial.py:15`) is:

```
φ(px, py) = [1, px, py, px², py², px·py]
```

Six terms, so:

```
scene_x = a₀ + a₁·px + a₂·py + a₃·px² + a₄·py² + a₅·px·py
scene_y = b₀ + b₁·px + b₂·py + b₃·px² + b₄·py² + b₅·px·py
```

Two completely independent fits — one for `x`, one for `y`. 12 unknowns total (6 per axis).

### 6 coefficients vs. 12 calibration points

A potential point of confusion: the default calibration pattern (`scripts/eyetracker/calibration/targets.py:23`) is a 4×3 grid = **12 fixation targets**, but each polynomial only has **6 coefficients**. Why so many points if there are only 6 unknowns?

Because we are *deliberately overdetermined*. Each fixation gives one `(px, py) → scene_x` equation and one `(px, py) → scene_y` equation. With 12 fixations we have 12 equations per polynomial in 6 unknowns — way more equations than unknowns. There is generally no exact solution; instead `np.linalg.lstsq` finds the 6 coefficients that minimize the total squared error across *all* 12 points simultaneously.

The 6 extra equations are not waste, they are noise averaging:

- The pupil center is noisy (per-frame jitter, ellipse fitting variance).
- The ArUco-derived label is noisy (marker detection shake).
- The user does not fixate perfectly.

If we used exactly 6 points, the polynomial would pass through every label exactly — including their noise. Those 6 coefficients would chase noise instead of true geometry, and predictions on new pupil positions would be worse. With 12 points the fit is forced to compromise, which pulls it toward the *underlying* pupil-to-scene relationship.

The hard minimum is 6 points (the `n < 6` guard). Below that the system is underdetermined — infinitely many polynomials fit. Above 6, more is generally better up to a point of diminishing returns.

## What calibration collects

For each fixation point we need a pair:

- **Input**: pupil center `(px, py)`, median-filtered over a small per-fixation buffer (`scripts/eyetracker/calibration/routine.py:197`).
- **Label**: where that fixation target actually appeared in the scene-cam image.

The label is the tricky part — that's where the homography enters.

## Where the homography fits in

We show a red dot at screen-pixel `(tx, ty)` on a monitor. The scene camera knows nothing about the screen — it just sees a 2D image of the world. We need to know: "the dot is at screen-pixel `(tx, ty)`; where in the scene-cam frame is that point right now?"

A flat screen viewed from an arbitrary angle becomes a quadrilateral in the camera image. The screen-plane → camera-image mapping is exactly a **homography** — a 3×3 matrix in projective coordinates.

Four ArUco markers are pinned at the screen corners. In `scripts/eyetracker/scene/aruco_homography.py:103`:

```python
H, _ = cv2.findHomography(screen_pts, scene_pts, method=0)
```

- `screen_pts` — the 4 marker centers in screen pixels (we drew them, so we know exactly where).
- `scene_pts` — the 4 marker centers detected in the scene-cam image.

OpenCV solves for the 3×3 `H` such that for any screen point `(tx, ty)`:

```
[u·w, v·w, w]ᵀ = H · [tx, ty, 1]ᵀ
scene_u = u·w / w
scene_v = v·w / w
```

That projection happens in `scripts/eyetracker/calibration/routine.py:171-177`. We recompute `H` on every calibration frame because the user's head moves and the screen-to-scene projection drifts. The resulting `(target_u, target_v)` is the "ground truth" scene-cam pixel for each pupil sample.

## The calibration loop

For each fixation:

1. User stares at the red dot at screen-pixel `(tx, ty)`.
2. Pupil pipeline gives us `(px, py)`.
3. ArUco gives us `H` for this frame.
4. `(target_u, target_v) = H · (tx, ty)` — the dot's location in scene-cam pixels.
5. Median-filter both over a few frames (`SampleCollector`) to get one clean `(pupil_median, scene_median)` pair.

After all accepted fixations (12 from the default grid, fewer if the user skipped any) we have matched arrays of `pupil_pts` and `scene_pts`.

## The fit

In `scripts/eyetracker/gaze/polynomial.py:35-46`: build the design matrix `A` (n rows × 6 columns, each row is `φ(px, py)`), and target vectors `bx` (n scene-x labels) and `by` (n scene-y labels). Solve two least-squares problems:

```python
cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)   # 6 coeffs for scene_x
cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)   # 6 coeffs for scene_y
```

`np.linalg.lstsq` minimizes `||A·c − b||²` — closed-form, no iteration. With 12 points and 6 unknowns the residual sum of squares has 6 degrees of freedom (n − p = 12 − 6), which is what lets the fit average out per-point noise instead of memorizing it. The hard minimum is 6 points (the `n < 6` guard); below that the system is underdetermined.

## Leave-one-out error

In `scripts/eyetracker/gaze/polynomial.py:48-57`: refit using `n−1` points, predict the held-out one, measure pixel error in scene-cam space. Average and max over all `n` rounds. This is more honest than training residuals — with 6 unknowns and only 6-12 points, training residuals would underestimate true error because the fit is close to interpolating its own training set.

## Prediction at runtime

After calibration the homography is gone — it did its job (labeling) and is no longer needed. At runtime (`scripts/eyetracker/gaze/polynomial.py:63-67`):

```python
feat = [1, px, py, px², py², px·py]
scene_x = feat · coeffs_x
scene_y = feat · coeffs_y
```

12 multiply-adds per frame. The polynomial takes pupil pixels and gives scene-cam pixels directly — no homography at inference, because calibration baked the geometry into the polynomial coefficients.

## Mental model

- **Polynomial** — "given a pupil pixel, where is the gaze in the scene cam?" The model used forever after calibration.
- **Homography** — "given the dot on the screen, where is it in the scene cam right now?" Only used during calibration to manufacture training labels. It's what turns "look at this screen pixel" into "the eye is fixating that scene pixel."

If the user moves their head, the homography changes (the screen appears somewhere different in the scene cam), but the polynomial doesn't care. Pupil-pixel-to-scene-cam-pixel is a property of how the eye + cameras are rigged together on the headset, invariant to where the user is standing.

The one assumption currently baked in: the eye and scene cameras are fixed relative to each other (head-mounted rig). If the rig flexes, the polynomial drifts and recalibration is needed.
