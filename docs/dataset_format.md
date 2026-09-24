# Dataset & calibration on-disk format

The pipeline records calibration/collection sessions as a **dataset of record** for
later analysis and CNN training. Each artifact uses the format that fits its job —
JSON for small structured/provenance data, CSV for append-during-capture sample
rows, image files for frames, Parquet only at consumption time.

## Layout

```
data/calibration/
  session_<ts>/
    metadata.json            # per-session record (intrinsics, extrinsics, provenance)
    labels.csv               # one row per accepted sample
    fixNN_sampleNN.png       # eye-cam frame
    fixNN_sampleNN_scene.png # scene-cam frame
  combined.parquet           # planned: cache for whole-dataset training loads
scripts/eyetracker/
  calibration.json           # live polynomial model restored at runtime
  scene_intrinsics.json      # scene-cam intrinsics (K, dist)
  eye_intrinsics.json        # eye-cam intrinsics, in the app's cropped 640x480 frames
rig_calibrations/
  <rig_id>.json              # written by scripts/extras/calibrate_extrinsics.py
```

## Artifacts

### `calibration.json` — the live model
Just what the runtime needs to restore the mapper. Raw samples are **not** stored
here; they live in the source session's `labels.csv` (named via `source_session`).

```json
{ "coord_space": "scene", "timestamp": 0.0, "source_session": "session_<ts>",
  "scene_size": [w, h], "screen_size": [w, h],
  "model": { "type": "polynomial", "degree": 2, "coeffs_x": [...], "coeffs_y": [...] } }
```

### `session_<ts>/metadata.json` — the dataset record
Self-contained: each session inlines a snapshot of the active intrinsics/extrinsics
so it stays interpretable even if the cameras or central calibration files change.
Fields the pipeline does not yet produce are `null` placeholders (filled later by the
extrinsics jig and richer per-frame capture).

```json
{ "session_id": "session_<ts>", "created_at": "<iso>", "timestamp": 0.0,
  "phase": "calibration",
  "subject_id": null, "glasses": null, "headset_model_version": null, "kappa_deg": null,
  "software": { "pupil_detector": null, "pye3d": null, "app_git_sha": null },
  "screen":   { "width": w, "height": h },
  "scene_cam":{ "width": w, "height": h, "fps": null, "identifier": null },
  "eye_cam":  { "width": w, "height": h, "fps": null, "fov_deg": 80.0, "identifier": null },
  "aruco":    { "dict_name": "...", "dict_id": n, "marker_px": n, "quiet_zone_px": n,
                "ids": [...], "screen_centers": [[x, y], ...] },
  "superseded_fixation_ids": [],
  "rig_calibration_id": null,
  "intrinsics": { "eye": null, "scene": { "K": [[...]], "dist": [...], "reproj_rms": n } },
  "extrinsics": null,
  "fit": { "degree": 2, "n_points": n, "loo_avg_px": n, "loo_max_px": n } }
```

### `labels.csv` — per-sample ground truth
Columns (see `LABELS_CSV_HEADER` in `calibration/persistence.py`):
`image_path, fixation_id, x_screen, y_screen, pupil_x, pupil_y, confidence,
timestamp, scene_target_x, scene_target_y`. Appended row-by-row during capture so a
mid-session crash keeps what was already written. `fixation_id` is unique within a
session (one per capture attempt, so rejected/aborted attempts leave gaps) and
names the images (`fix<id>_sample<n>.png`). A screen point can appear under
several ids: once per pose in multi-pose, and twice when the detailed routine
recaptures it. `metadata.json`'s `superseded_fixation_ids` lists the pass-1
ids a recapture replaced; drop them to reproduce the live fit. Sessions from
before 2026-09-23 reused ids across poses/recapture passes (and overwrote those
images), so their `fixation_id` is unreliable there. Richer per-frame fields
(ellipse params, pye3d 3D vectors, normalized-image paths) from
`docs/data_collection.md` are added as the pipeline starts producing them.

### `rig_calibrations/<rig_id>.json` — camera-rig calibration (planned)
Written by `scripts/extras/calibrate_extrinsics.py` from a jig capture. Sessions don't
reference it yet: `rig_calibration_id` and `extrinsics` in `metadata.json` are still
null (see `TODO.md`).

```json
{ "rig_id": "<ts>", "created_at": "<iso>", "notes": null,
  "intrinsics": {
    "eye":   { "K": [[...]], "dist": [...], "image_width": 640, "image_height": 480,
               "reproj_rms": r, "source": "eye_intrinsics.json" },
    "scene": { ...same keys... } },
  "extrinsics_eye_to_scene": {
    "R": [[...]], "t": [...], "units": "mm", "method": "calibrateRobotWorldHandEye/SHAH",
    "residual_rot_deg": { "mean": x, "max": x }, "residual_trans_mm": { "mean": x, "max": x } },
  "capture": { "n_pairs": n, "pose_diversity_deg": x, "eye_board": "small",
               "scene_board": "large", "board_b_to_a": { "R": [[...]], "t": [...] } } }
```

`extrinsics_eye_to_scene` maps eye-cam coordinates to scene-cam coordinates:
`p_scene = R @ p_eye + t`, in mm. The intrinsics are copied in so the record is
self-contained; the eye intrinsics describe the app's processed eye frames (4:3 crop,
resized to 640x480), not the camera's native mode.

## Loading for analysis / training

`scripts/eyetracker/dataset.py`:
- `load_session(dir) -> (metadata, DataFrame)` — one session.
- `session_dirs(root)` — list every `session_<ts>/` under a root.

Planned (TODO in `dataset.py`, build with the CNN training loop): `load_all` to
concatenate all sessions + cache `combined.parquet`, and an image-loading layer that
joins each row's `image_path` to its eye frame.

`scripts/extras/measure_gaze_accuracy.py --session <dir>` refits the polynomial from
a session and reports LOO error in px and degrees.
