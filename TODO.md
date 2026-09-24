# TODO

Running backlog. Check items off as they land. Put the branch/PR next to an
item while it's in flight. Detailed per-feature plans get their own
`TODO_<topic>.md` (e.g. `TODO_multipose_coverage.md`).

Rig: Jetson Orin (JetPack 6, Ubuntu 22.04) + Philips 221V8LB @ 1920x1080 100 Hz.
Eye cam: Arducam OV9281 (`0x0c45:0x6366`). Scene cam: `0x0bda:0xd565`.

## Linux port (`linux-jetson-support`)

- [x] V4L2 + MJPG capture (YUYV caps 1080p at 5 fps)
- [x] Scene-cam exposure via `v4l2-ctl`; pin frame rate (`exposure_dynamic_framerate=0`)
- [x] Pick scene cam / default eye cam by USB id (two `/dev/video` nodes per camera)
- [x] Monitor model + physical mm in session `metadata.json`
- [x] Monitor at 100 Hz, persisted in `~/.config/monitors.xml`
- [x] Headset-on quick calibration on the Jetson
- [ ] Check the scene exposure hotkeys and `metadata.json` from a real session
- [x] Open PR (#37, merged)

## Capture architecture

- [x] One grabber thread per camera keeping only the newest frame (eye frame
      age 275 ms -> 13 ms on the Jetson). ArUco only runs during calibration.
- [ ] Stamp each frame on arrival in the grabber thread (feeds timestamping below)
- [ ] App loop still blocks on the 30 fps scene read, so the eye is processed
      at ~30 Hz, not ~100. Decouple so every eye frame is processed.
- [ ] ArUco on full 1080p is ~45 ms/frame, so the loop drops to ~14 Hz during
      calibration. Try detecting on a downscaled frame and rescaling corners.
- [ ] Jetson power mode: currently 15W; try `sudo nvpmodel -m 2` (MAXN_SUPER)
- [ ] Main loop consumes the latest eye frame; scene frames matched by timestamp
- [ ] Eye cam at 120 fps (640x480 MJPG). Measured 121 fps alongside 1080p scene
      on the shared USB 2 hub, so bandwidth is fine. Update `HIGH_FPS_MODE` /
      `EYE_CAM_RESOLUTION` accordingly.
      Note: the OV9281 has no 30 fps mode (MJPG is 100/120 only; YUYV 10 fps),
      so it already runs at 100 by default, on the Mac too.
      Also: `EYE_CAM_FOV_DEG` (80° diagonal) is the old Sonix lens. Eye frames
      are cropped to 4:3 and resized to 640x480 before detection, so the
      effective FOV depends on the OV9281's native mode and that crop. It sets
      pye3d's focal length and goes into `metadata.json`: pin the eye mode
      explicitly, then derive the FOV (~70° horizontal lens) or use measured
      eye intrinsics (`calibrate_eye_intrinsics.py`, see the extrinsics item).

## Timestamping + sync

- [ ] Hardware / driver frame timestamps for both cams (V4L2 buffer timestamps,
      UVC PTS/SCR), logged per frame
- [ ] Log calibration-dot onset times on the host clock (flip-accurate at 100 Hz)
- [ ] Clock offset / drift / round-trip estimation between camera clocks and
      the host: LSL, or our own NTP-style exchange
- [ ] Offline tool: align eye + scene + stimulus streams from logged timestamps

## Accuracy: depth / parallax + headset slip

Finding (2026-09-23): calibration labels, fit and live predict path all check
out (quick 9-pt LOO ~14-28 px). Remaining error is physical. Gaze is only exact
at the calibration depth (scene cam sits a few cm from the eye), and the
pupil-only mapping is very slip-sensitive: a ~40 px pupil shift between two
runs moved gaze by ~900 px.

- [ ] Scene-cam intrinsics (`scripts/extras/calibrate_scene_intrinsics.py`);
      also unlocks degree-based accuracy in `measure_gaze_accuracy.py`
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
- [ ] Eye↔scene extrinsics: the jig is built (three steel sheets at ~90°; board
      sizes not chosen yet). Port `calibrate_extrinsics.py` /
      `calibrate_eye_intrinsics.py` from `claude/calibration-jig-review` (keep
      that branch until then). Fixes needed when porting:
      - Eye frames must match the app's geometry: same eye mode + the 4:3
        crop/resize to 640x480 (`cameras/utils.py`) and `EYE_CAM_FLIP_VERTICAL`.
        As written, intrinsics request 640x480 with no crop while extrinsics
        use the default mode, so K wouldn't match. Pin the eye mode first
        (see the FOV item above).
      - Extrinsics must open the scene cam at 1920x1080 like the app; it uses
        the default mode, which may not match `scene_intrinsics.json`
      - Read/write JSON (`scene_intrinsics.json`; output to
        `rig_calibrations/<rig_id>.json` per `docs/dataset_format.md`) instead
        of `.npz`, and build boards from `scripts/extras/charuco_boards.py`
      - Then have sessions reference/inline the rig calibration in
        `metadata.json` (currently null)
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
