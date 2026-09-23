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
- [ ] Headset-on quick calibration on the Jetson: pupil quality, all 4 markers, LOO error
- [ ] Check the scene exposure hotkeys and `metadata.json` from a real session
- [ ] Open PR

## Capture architecture

- [ ] One capture thread per camera, each stamping frames on arrival. Today
      `App.run()` reads eye then scene serially, so the eye cam (natively
      ~100 fps) is paced by the 30 fps scene cam and its frames queue in the
      driver, which adds latency.
- [ ] Main loop consumes the latest eye frame; scene frames matched by timestamp
- [ ] Eye cam at 120 fps (640x480 MJPG). Measured 121 fps alongside 1080p scene
      on the shared USB 2 hub, so bandwidth is fine. Update `HIGH_FPS_MODE` /
      `EYE_CAM_RESOLUTION` accordingly.

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
