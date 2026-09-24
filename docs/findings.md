# Findings

What we've measured, decided, or learned the hard way, so it carries across
sessions and machines. One or two lines each, newest first within a section,
with a pointer (→) to the doc or code that holds the detail.

When a finding stops being true, don't delete it: mark it **Superseded** with the
date and what replaced it. "We tried X and it didn't work" is often the most
useful part. Entries marked *unverified* haven't been confirmed on the rig.

## Gaze model and calibration

- 2026-09-24 · Scene cam horizontal FOV is ~54° (fx 1894.7 px at 1920 wide), so
  multi-pose calibration is capped by when a corner marker leaves the frame, not
  by the lens. → `TODO_multipose_coverage.md`
- 2026-09-23 · Calibration labels, fit and live prediction check out (quick
  9-point LOO ~14–28 px). Remaining error is physical: gaze is only exact at the
  calibration depth (eye↔scene parallax), and the pupil-only mapping is very
  slip-sensitive (a ~40 px pupil shift between runs moved gaze ~900 px).
  → `TODO.md` (Accuracy)
- 2026-07-13 · *Unverified:* undistorting marker corners before `findHomography`
  cut geometric error at non-anchor points by ~1.2 px in a synthetic test with the
  real distortion coefficients. → branch `use-intrinsics-for-homography`
- 2026-06-29 · A half-finished hat-matrix LOO refactor once returned all-zero LOO
  errors, silently breaking recapture targeting and accuracy reports.
  → `docs/loo_error_notes.md`, `tests/test_polynomial_loo.py`
- 2026-06-06 · **Decision:** the roadmap is polynomial → CNN (eye image in, gaze
  point out), with no 3D-model step. First, data collection whose sessions stay
  usable across rig changes.
- Before 2026-06 · Polynomial ceiling: a narrow central grid gave 0.39° avg /
  0.94° max LOO; a wider grid made the fit worse everywhere (~1.56° avg). Don't
  tune the polynomial (degree, grid size, point count). → `docs/calibration_coverage.md`
- Standing · Benchmark: Pupil Labs Neon, ~0.5–1° (model-based, multi-glint, slip
  correction).

## Cameras

- 2026-09-24 · The app center-crops eye frames to 4:3 and resizes them to 640x480
  before pupil detection, so pupil coordinates and saved eye images are 640x480 in
  any mode. But the eye cam's native mode isn't requested (driver default), and
  `EYE_CAM_FOV_DEG` (80°) is the old Sonix lens; it feeds pye3d's focal length and
  `metadata.json`. → `config.py`, `TODO.md`
- 2026-09-24 · Eye cam illumination: one hand-soldered IR LED above the lens
  (off-axis): dark pupil, single glint, no retroreflection. → `3d-files/MEASUREMENTS.md`
- 2026-09-23 · OV9281 manual exposure works over UVC on Linux (0.5–9.9 ms
  verified). In auto mode it doesn't report the exposure it picked.
  → `docs/timestamping.md` (on branch `timestamping`)
- 2026-07-15 · OV9281 caps at 120 fps in every mode (MJPG 320x240 to 1280x800 at
  100/120 fps; YUYV only 10 fps; no 30 fps mode). Faster needs other hardware.
- 2026-07-15 · Trap (macOS): `ffmpeg -f avfoundation` "negotiates" 120 fps but
  delivers ~83. Always measure delivered fps.
- 2026-07-15 · The OV9281 reuses the Sonix bridge's USB id `0x0c45:0x6366`; tell
  them apart by serial (UC762). Camera indexes shift between enumerations, so
  select cameras by USB id, not index.
- 2026-06-28 · The scene cam (Realtek OV5640) is manual-exposure only;
  `exposure-time-abs` is the real brightness lever and `gain` only scales output.
  → `docs/uvc_exposure_cheatsheet.md`
- **Superseded 2026-06-28** · "Scene exposure-time is cosmetic." It was tested in
  auto mode, where UVC ignores exposure writes.
- **Superseded 2026-09** · "The eye cam's exposure can't be controlled." True only
  of the retired Sonix GC0308, whose auto-exposure ran internally.
- 2026-06-08 · uvc-util reports success even when a camera ignores or clamps a
  write, so every set is read back. On macOS, OpenCV can't set exposure at all.
  → `cameras/uvc_util.py`

## Timing and sync

- 2026-09-23 · Eye−scene timestamp offset (flash test): 4.8 ms, ±0.3 ms run to
  run, ~±2 ms systematic. Rerun if exposure or cameras change.
  → `docs/timestamping.md` (on branch `timestamping`)
- 2026-09-23 · The 5.15 kernel's UVC hardware timestamps are broken for the eye cam
  (its bridge reports a free-running counter; stamps drifted ~2600 ppm and jumped
  >100 ms), so PTS/SCR are converted in userspace. → `docs/timestamping.md`
  (on branch `timestamping`)
- 2026-09 · Without a grabber thread, V4L2 hands back the oldest queued frame: eye
  frames were 130–275 ms stale on the Jetson. → `cameras/opencv_source.py`
- 2026-09-24 · **Decision:** no streaming the rig's video to another machine for
  calibration or display for now. Calibration-dot onsets would then happen on a
  different machine's clock, bringing back the cross-machine sync problem.
  (`linux_cam_stream.py` removed; it's in git history.)
- 2026-09 · **Decision:** research capture runs on Linux. On macOS, raw UVC
  timestamps need root and exclusive device access, which rules out OpenCV
  capture. macOS stays a live-demo mode. → `README.md` (Platforms)

## Calibration jig and rig geometry

- 2026-09-24 · Synthetic check of the hand-eye extrinsics solve: exact recovery on
  clean data; 0.29° / 2.3 mm error with 0.2° / 0.5 mm per-pose noise over 20 pairs.
  → `tests/test_extrinsics_solve.py`
- 2026-09-24 · Eye-side board: at eye distance (~45–50 mm) 5 mm squares give
  45–113 ChArUco corners per frame in a synthetic render, vs ~6 for 15 mm squares.
  Refocusing the M12 lens changes the intrinsics by a few percent, so calibrate at
  eye focus. The print dictionary has only 1000 marker IDs, so board sizes are
  capped. → `scripts/extras/charuco_boards.py`, `tests/test_charuco_boards.py`
- 2026-09-24 · *Unverified:* many inkjet inks are nearly invisible in IR, so
  laser-print boards the eye cam must see.
- 2026-09-24 · **Decision:** the 3D-printed jig was abandoned for three stainless
  steel sheets at ~90° with pasted ChArUco boards. The hand-eye solve only needs
  the jig rigid, not measured. → `3d-files/README.md`, `calibrate_extrinsics.py`
- 2026-05 · Tape-measured camera offsets: ~50 mm lateral, 13 mm depth, 50 mm
  vertical (~72 mm baseline), on the old mounts. Re-measure before relying on them.

## Data

- 2026-09-24 · **Decision:** no frontend until the tracking fundamentals work. The
  paused Next.js dashboard and its tools were removed (they're in git history).

- 2026-09-24 · The pre-migration calibration history (27 Sonix-era sessions, 352
  pupil/target pairs, no images) is kept only on the Mac at `data/legacy/`, not in
  git.
- 2026-06-06 · Persistence moved off `.npz` to JSON + CSV (`calibration.json`,
  per-session `metadata.json` + `labels.csv`). → `docs/dataset_format.md`
