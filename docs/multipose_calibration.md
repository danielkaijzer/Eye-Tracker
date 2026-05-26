# Multi-Pose Calibration

How we widen the calibrated region of the field of view so gaze stays accurate when
the user looks off-center — the fix for the coverage problem described in
[`calibration_coverage.md`](calibration_coverage.md).

## The problem, recapped

A normal head-on calibration only samples a narrow band of eye rotations: every dot is
on a monitor that fills a small patch near the center of the scene camera's view. The
pupil→scene polynomial is then trustworthy only inside that small central cloud and
extrapolates badly outside it. We want to extend the cloud toward the full comfortable
oculomotor range (~±20–25°) without buying a bigger screen.

## The mechanism: move the head, not the dots

The key fact (same one that makes the polynomial head-pose invariant at runtime) is that
the scene camera is **head-mounted**, so the pupil→scene mapping is *head-relative*.

Rotate your **head** ~30° away from the screen but keep fixating the dots, and two things
happen at once:

- The screen now lands near the **edge** of the scene-camera frame, so the homography
  places the labels at high eccentricity.
- Your eye must rotate ~30° to keep fixating, so the **pupil** moves into a position a
  head-on calibration never reaches.

Both the input cloud (pupil) and the output cloud (scene pixels) extend together. The
control variable is **head orientation relative to the target**, not screen tilt — and
it reaches eye rotations that are simply unavailable when the screen sits dead-center
(where max eye rotation ≈ half the screen's angular width).

### Why discrete static poses, not continuous motion

Capture must happen while the head is **still**. The per-fixation gate
(`SampleCollector` in `scripts/eyetracker/calibration/collector.py`) rejects a fixation
if the scene label drifts more than `CALIB_SCENE_STD_THRESH` px ("Head may have moved").
That gate is correct and we keep it. So the routine is a sequence of **static poses**:
reposition the head → hold still → run a grid → reposition → next pose. Continuous
"look around while moving" would both trip the gate and desync the eye/scene cameras
(the pupil frame and the homography label would come from slightly different instants),
injecting label noise that scales with motion speed.

Do all poses in **one session without removing or adjusting the headset**, so headset
slippage doesn't drift between poses — every pose then samples the same pupil→scene map.

## The routine

Press **`m`** to start. Multi-pose reuses the existing grid + homography + collector; the
only new behavior is a loop over poses that accumulates into a single fit
(`CalibrationRoutine` with `num_poses > 1`, in `scripts/eyetracker/calibration/routine.py`).

Flow:

1. **Pose 1** is captured exactly like a normal grid. The overlay shows `Pose 1/5` and
   per-pose guidance.
2. When the grid completes and poses remain, the routine enters a **pose break**
   (`_begin_pose_break`): it stops collecting and the overlay shows the next pose's
   instruction (e.g. "turn your head LEFT") plus the live `aruco: N/4 markers visible`
   readout. Captured points are **not** cleared.
3. The user repositions and presses **`c`** to start the next pose (`begin_next_pose`):
   the same grid is re-armed, `current_idx` resets, but the accumulated buffers carry
   over.
4. After the **final** pose, all `(pupil, scene)` pairs across every pose feed one
   `lstsq` fit (the existing `_finish` path).

So 5 poses × a 12-point grid → ~60 points feeding one degree-3 fit, spanning a much wider
slice of the oculomotor range than 20 head-on points.

Two-pass **recapture is disabled** in multi-pose (`recapture_worst_n = 0`). Recapture
re-prompts the worst points, but "the worst point" came from a specific head pose; redoing
it at whatever pose the user happens to be in would be ambiguous.

### Configuration

In `scripts/eyetracker/config.py`:

| Constant | Meaning |
| --- | --- |
| `CALIB_POSES` | number of head poses (default 5: head-on, left, right, down, up) |
| `CALIB_MULTIPOSE_ROWS` / `_COLS` / `_MARGIN` | grid per pose (default 4×3, margin 180) |
| `CALIB_MULTIPOSE_DEGREE` | polynomial degree (default 3 — plenty of points to support it) |
| `CALIB_POSE_GUIDANCE` | per-pose instruction strings shown on the overlay |

`next_pose_guidance()` falls back to a generic prompt if you raise `CALIB_POSES` beyond
the number of guidance strings, so the two can be tuned independently.

### Practical limits

- **ArUco visibility bounds the pose.** At steep angles the screen rotates toward the
  edge of the scene frame and a corner marker can drop out. The existing "not all 4
  markers visible" guard simply won't capture until all four are back — the `aruco: N/4`
  readout tells the user when a pose is too extreme.
- **Oculomotor comfort bounds it too.** Beyond ~±25° of eye rotation fixation gets
  uncomfortable and unstable; that's also roughly where real users start turning their
  head instead, so it's a sensible ceiling rather than a limitation to fight.

## What it improves (and what it can't)

- **Improves:** the *usable* field of view. Outer-eccentricity error should drop sharply
  because those regions are now interpolation instead of extrapolation. It also enriches
  the per-session image dataset (eye + scene frames are saved per sample regardless), so
  the future image-to-gaze model gets training data spanning the full oculomotor range —
  the main motivation.
- **Can't:** raise the polynomial's *center* ceiling. A global degree-3 fit has fixed
  capacity; spreading data over a wider, more nonlinear domain may trade a little
  peak-center accuracy for the edges. Lifting the ceiling needs a higher-capacity or
  local model — the planned image-to-gaze pivot.

## Verifying it worked

Multi-pose is exactly the kind of change that needs the
[eccentricity validation tooling](calibration_coverage.md#the-validation-tooling) to
confirm — "feels better" isn't measurable. End-to-end:

1. Capture a held-out wide-angle set once with `v` (steep head angle). Keep this
   `validation_<ts>.npz` fixed so it's a fair yardstick across runs.
2. Measure your **current head-on** calibration against it:
   `python -m scripts.extras.measure_gaze_accuracy --val scripts/eyetracker/validation_*.npz`
   — expect large error in the outer eccentricity bins (the gap).
3. Recalibrate with `m` across all poses (don't touch the headset), confirming LOO stays
   under the usable threshold (`0.04 × scene_width`).
4. Re-measure against the same `--val` set. The outer bins should drop substantially.
   Watch the center bin too, to quantify any trade-off.

## Future direction

Multi-pose reuses the screen + ArUco rig, so coverage is still bounded by screen size and
marker visibility. A more scalable rig for the model dataset is **world-fixed fiducials**
— a printed ArUco wall or poster — letting targets span the full scene FOV independently
of any screen, and dovetailing with the planned extrinsics jig. That's deliberately out
of scope here; multi-pose is the no-new-hardware step.
