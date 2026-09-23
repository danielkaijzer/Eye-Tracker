# TODO

Running backlog. Check items off as they land. Put the branch/PR next to an
item while it's in flight. Detailed per-feature plans get their own
`TODO_<topic>.md` (e.g. `TODO_calibration_ux.md`).

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
- [ ] Open PR

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

## Timestamping + sync

- [ ] Hardware / driver frame timestamps for both cams (V4L2 buffer timestamps,
      UVC PTS/SCR), logged per frame
- [ ] Log calibration-dot onset times on the host clock (flip-accurate at 100 Hz)
- [ ] Clock offset / drift / round-trip estimation between camera clocks and
      the host: LSL, or our own NTP-style exchange
- [ ] Offline tool: align eye + scene + stimulus streams from logged timestamps

## Housekeeping

- [ ] `record.py` / `camera_test.py` / `linux_cam_stream.py`: reuse
      `cameras/v4l2.py` enumeration instead of hardcoded indexes
- [ ] Eye cam dropped off USB once with `UVC probe control: -71` (fixed by
      replug). If it recurs under load, try a powered hub.
