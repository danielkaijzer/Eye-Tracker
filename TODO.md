# TODO

Running backlog of open work. Put the branch/PR next to an item while it's in
flight; check it off when the work is done on its branch, and delete it once
that merges (git history keeps it). Move any finding worth keeping (a
measurement, a gotcha) into the relevant doc first. Detailed per-feature plans
get their own `TODO_<topic>.md` (e.g. `TODO_multipose_coverage.md`).

Rig: Jetson Orin (JetPack 6, Ubuntu 22.04) + Philips 221V8LB @ 1920x1080 100 Hz.
Eye cam: Arducam OV9281 (`0x0c45:0x6366`). Scene cam: `0x0bda:0xd565`.
What we've learned so far: `docs/findings.md`.

## Next up

In order; details in the items below. Update this list at every milestone.

1. Test and merge `timestamping` (`git pull` on the Jetson first)
2. Pin the eye cam mode (Capture architecture)
3. On the rig: undistorted-homography check, then merge `use-intrinsics-for-homography`
4. Print and paste the jig boards; run eye intrinsics + extrinsics (Accuracy)
5. Record the rig calibration in session `metadata.json`, then polish data collection

Steps 3 and 4 can share one session at the rig.

## Linux port

- [ ] Check the scene exposure hotkeys and `metadata.json` from a real session

## Capture architecture

- [ ] Stamp each frame on arrival in the grabber thread (feeds timestamping below).
      Done on `timestamping`.
- [ ] App loop still blocks on the 30 fps scene read, so the eye is processed
      at ~30 Hz, not ~100. Decouple so every eye frame is processed.
- [ ] ArUco on full 1080p is ~45 ms/frame, so the loop drops to ~14 Hz during
      calibration. Try detecting on a downscaled frame and rescaling corners.
- [ ] Jetson power mode: currently 15W; try `sudo nvpmodel -m 2` (MAXN_SUPER)
- [ ] Main loop consumes the latest eye frame; scene frames matched by timestamp
- [ ] Pin the eye cam mode: request it in `_eye_cam_settings()` (today it's the
      driver default, so the cropped 640x480 frame's geometry can change). On the
      rig, compare the field of view at 640x480 vs 1280x800 (full sensor or a
      crop?), and consider 120 fps (640x480 MJPG measured 121 fps alongside the
      1080p scene on the shared USB 2 hub; `HIGH_FPS_MODE`).
- [ ] Then use the measured eye focal length (fx in `eye_intrinsics.json`) for
      pye3d and `metadata.json` instead of `EYE_CAM_FOV_DEG` (80°, the old Sonix
      lens)

## Timestamping + sync

- [ ] Hardware / driver frame timestamps for both cams (V4L2 buffer timestamps,
      UVC PTS/SCR), logged per frame
- [ ] Log calibration-dot onset times on the host clock (flip-accurate at 100 Hz)
- [ ] Clock offset / drift / round-trip estimation between camera clocks and
      the host: LSL, or our own NTP-style exchange
- [ ] Offline tool: align eye + scene + stimulus streams from logged timestamps

## Accuracy: depth / parallax + headset slip

Remaining error is physical (depth parallax + headset slip), not the
calibration code: see the 2026-09-23 entry in `docs/findings.md`.

- [ ] Undistorted homography (`use-intrinsics-for-homography`): on the rig, run
      one calibration with and one without `scene_intrinsics.json` and compare
      the `ArUco check` err near the frame edges, then merge. The loader doesn't
      check the file's image size against the capture resolution yet.
- [ ] Per-sample ArUco data in the session dataset (store raw, derive later):
      - `aruco.csv` per session, keyed by `image_path`: one row per detected
        marker, `marker_id` + 4 corners x (x, y) in scene px. Separate file so
        `labels.csv` stays narrow and partial detections (2-3 markers) fit.
        Screen-side marker positions are static and already in `metadata.json`.
      - `labels.csv`: add `homography_reproj_px` (per-sample quality flag) and
        `screen_distance_mm` (null until scene intrinsics exist)
      - Depth = solvePnP on the stored corners (marker ~29.8 mm, from
        `SCREEN_PHYSICAL_MM` / pixel pitch) -> screen distance + pose. Raw
        corners let it be backfilled for every session once intrinsics exist.
      - Per sample, not per fixation, so head motion during capture shows up
      - Update `docs/dataset_format.md` + a test
- [ ] Eye↔scene extrinsics on the rig. Pin the eye mode first (Capture architecture):
      eye intrinsics only hold for one native mode + crop. Laser-print the
      `tiny` board for the eye side (sized for eye distance, no refocus) and
      pick a scene-side size, paste them on the jig, then run
      `calibrate_eye_intrinsics.py` and `calibrate_extrinsics.py`
- [ ] Sessions reference the rig calibration in `metadata.json`
      (`rig_calibration_id`, `extrinsics`, eye intrinsics; all null today). Do it
      after `timestamping` merges, since both change `persistence.py`
- [ ] After `timestamping` merges: move `_eye_cam_settings` /
      `_scene_cam_settings` out of `__main__.py` into a shared module (the
      calibration scripts import them from `__main__` for now), and drop
      "(planned)" from the `rig_calibrations` heading in `docs/dataset_format.md`
      (left alone here to avoid a merge conflict)
- [ ] Multi-pose coverage at steep head angles: live marker count as pose
      guidance, then more border markers so any 4 well-spread ones work
      (`TODO_multipose_coverage.md`)
- [ ] Parallax model: calibrate at 2+ depths, use the eye-to-scene offset to
      correct gaze for a given depth (needs a runtime depth source: assumed,
      scene depth, or vergence from a second eye cam)
- [ ] Slip robustness: reintroduce the glint as a pupil-CR reference, or a
      quick 1-point drift correction before each use

## Housekeeping

- [ ] Eye cam resets / drops off USB often (~25x on 2026-09-23, incl.
      `can't read configurations, error -71`), worse through passive USB
      extension cables. Try a powered hub near the headset / short or active
      cables. Capture code should also survive a reset (reopen by USB id).
