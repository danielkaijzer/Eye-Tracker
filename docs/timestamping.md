# Frame timestamps (Linux)

Every eye and scene frame carries a capture timestamp on the host monotonic
clock (`CameraSource.last_timestamp`, same clock as `time.monotonic()`). The
timestamps are logged per calibration sample as `eye_frame_ts` /
`scene_frame_ts` (see `dataset_format.md`). This page covers where they come
from, what they mean, and how well they line up.

## Setup

The `uvcvideo` driver needs two settings, which the Jetson rig has in
`/etc/modprobe.d/uvcvideo.conf`:

```
options uvcvideo nodrop=1 hwtimestamps=0
```

Reload the driver after changing them: `sudo modprobe -r uvcvideo && sudo modprobe uvcvideo`,
with nothing using the cameras. At open, each camera prints
`[timestamps] cam N: camera clock (UVC PTS/SCR via /dev/videoM)`. If the setup
is wrong, it prints a warning and falls back to host arrival times.

Check: `python -m scripts.extras.probe_uvc_timestamps --seconds 30`.

## How it works (`cameras/uvc_meta.py`, `cameras/uvc_clock.py`)

UVC cameras put two clock fields in the header of each USB packet:

- **PTS**: the device clock (STC, nominally 15 MHz on both rig cams) at a fixed
  point in the frame.
- **SCR**: a device-clock sample, plus what the device claims is the USB bus
  frame number (SOF).

The kernel exposes these raw headers on each camera's second `/dev/video`
node, the metadata node. For every header, the kernel also records the host
time at which it processed that USB packet. `UvcTimestamper` reads the
metadata node. `DeviceClock` fits device clock to host time along the *lower
envelope* of those (SCR, host time) pairs, over a sliding 10 s window. Host
processing only ever adds delay, so the earliest arrivals trace the true
mapping. Each frame's PTS is then converted through the fit. The video frame
and its metadata are paired by V4L2 buffer timestamp: with `hwtimestamps=0`
the kernel copies the same value into both.

Why each setting is needed:

- **`nodrop=1`**: both cams put a fresh SCR in nearly every packet, so every
  frame overflows the kernel's fixed 1 KiB metadata buffer after 46 headers.
  With `nodrop=0` (the default) the driver silently discards every such buffer,
  and the node looks empty.
- **`hwtimestamps=0`**: the kernel's own PTS conversion is broken for the eye
  cam. It converts through the SOF in SCR, but the eye cam's bridge chip
  reports a free-running counter there (~1003 ticks/s, not the bus's 1000).
  Converted that way, the eye timestamps drifted ~2600 ppm and jumped by
  100-250 ms. Newer kernels (6.8+) have a quirk for cameras like this. The
  Jetson runs 5.15, so we convert in userspace and ignore the device SOF.

Measured on the rig:

| | Device clock | Fit residual | Frame period (std) | Drift vs host |
|---|---|---|---|---|
| Eye (Arducam OV9281) | 15.0014 MHz (+82-97 ppm) | ~0.01-0.02 ms | 9.9090 ms (0.0009) | within +/-5 ppm |
| Scene | 14.9994 MHz (-38-41 ppm) | ~0.004 ms | 33.3324 ms (0.0005) | within +/-5 ppm |

## What a timestamp means

Both cams set PTS ~0.06-0.3 ms before a frame's first USB packet. So the
timestamp marks when the camera **starts sending** the frame: after exposure,
readout and JPEG encoding. It also includes the minimum host USB processing
delay, which the envelope fit can't separate out. Both delays are fixed per
camera, but they differ between cameras. Frames paired by equal timestamps are
therefore off by a constant.

## Eye vs scene offset (`scripts/extras/flash_sync_test.py`)

The flash test points both cams at a flashing square and finds, for each
camera, when its timestamps say the light changed. Five handheld runs on
2026-09-23 (eye exposure 8 ms manual, scene 33.2 ms):

- **eye - scene: 4.8 ms** (per-run medians 4.64-5.31 ms). Eye timestamps run
  ~5 ms later than scene timestamps for the same light. To pair frames by
  light, use `eye_frame_ts - 0.0048`.
- Systematic uncertainty is about +/-2 ms. On flash-on edges the difference is
  7.3 ms; on flash-off edges it's 3.0 ms. The cams integrate over very
  different exposures (8 ms global shutter vs 33 ms rolling shutter), and the
  VA panel brightens and darkens at different speeds, so edge shape moves each
  camera's crossing differently. This is well under one frame for either cam.
- Latency from flip request to camera timestamp (includes display latency,
  which has ~2.9 ms spread from the 100 Hz refresh phase): eye 17 ms on /
  27 ms off, scene 10 ms on / 24 ms off. This is the figure to use when
  relating frames to stimulus times logged the same way, such as dot onsets.

The offset depends on exposure settings. Rerun the test if you change them,
or if you change cameras. A PTS taken after readout moves by half of any
exposure change relative to mid-exposure: up to ~5 ms across the eye cam's
0.5-9.9 ms range. That model is unverified: running the flash test at two eye
exposures would confirm it.

## Eye exposure is locked per session

The camera's own auto exposure keeps changing exposure mid-recording and never
reports what it picked; in auto mode the control just keeps the last manual
value. So the App opens the eye cam in **manual** mode and, with
`config.EYE_EXPOSURE = None`, runs a one-shot software AE at startup
(`cameras/exposure_settle.py`). It steps the manual exposure until the eye
image median reaches `EYE_EXPOSURE_TARGET_MEDIAN` (96, what the camera's AE
picked on the eye in good sessions), then leaves it. This takes under a second.
After that it changes only on request: `e` re-settles, `,` / `.` step it. It
never changes during a calibration. The value is recorded per session
(`eye_cam.exposure_ms` in `metadata.json`). Settle with the headset on: on
anything else, the target won't match the eye.

## Known issues

- With `nodrop=1`, uvcvideo also delivers video frames it flags as corrupt.
  One flash run (cameras handheld, cables moving) produced 3 truncated MJPEG
  frames (`Corrupt JPEG data: premature end of data segment`). 2 minutes of
  stationary streaming produced none in 15,662 frames. The flash analysis
  ignores single-frame glitches. The App does not yet reject corrupt frames
  (see `TODO.md`).
- The eye cam resets on USB when cables are moved or long passive extensions
  are used (see `TODO.md`). A reset restarts the device clock; the fit detects
  the jump and starts over.
