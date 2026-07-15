# Jetson camera test plan

Context for a fresh Claude Code session running on the Jetson Nano. This
file is the handoff — read it fully before doing anything, since the Mac
session that produced it isn't available to ask follow-up questions.

## Why we're here

The eye-tracker's data-collection pipeline needs research-grade frame
timestamps to synchronize the eye camera and scene camera. On macOS we hit a
hard architectural wall investigating this (full writeup below) and are
evaluating migrating the capture pipeline to Linux (this Jetson, or a
cheaper Linux mini-PC if the Jetson isn't necessary) to fix it at the root
instead of working around it. This is a **validation step, not a committed
migration yet** — the Mac session is leaning Linux but wants Jetson-side
evidence before treating it as decided.

Secondary goal: one of our two eye cameras (old Sonix module) completely
failed UVC negotiation in a strict userspace UVC stack on Mac. Worth
re-testing it here, because the Linux kernel's `uvcvideo` driver is a
different, more mature/tolerant UVC implementation than what failed it on
Mac — it's plausible this camera just works here.

## What we already found on macOS (don't re-derive this)

Built an instrumented copy of `libuvc` (patched `stream.c` to print raw UVC
payload-header PTS/SCR fields — the public libuvc API only exposes a
host-interpolated timestamp, not the device's raw clock) and probed all
three cameras via raw USB:

- **Scene camera** (Realtek/OV5640, USB ID `0bda:d565`): clean, real
  hardware clock. Frame-to-frame PTS deltas exactly 499968 ticks (~15MHz
  clock, ~67ns resolution), SCR monotonic with zero backward jumps across
  24,000 samples in 3s. This camera is not the problem.
- **Old eye camera** (Sonix GC0308, USB ID `0c45:6366`, serial `SN0001`):
  **fails UVC Probe/Commit negotiation entirely** in libuvc — every
  format/resolution combination advertised in its own descriptors
  (MJPEG and YUY2, at 320x240/640x480/160x120, several frame rates)
  returned `Invalid mode (-51)` on `uvc_get_stream_ctrl_format_size`. Full
  device descriptor dump and failure log are worth having handy — see
  "Reference data" below if you want the exact fixture. AVFoundation/OpenCV
  on Mac can still capture from it (Apple's driver tolerates whatever
  nonstandard thing it does), but a byte-correct UVC stack cannot. This is
  a firmware compliance problem in the sensor bridge, not a code bug on our
  side — the open question is whether Linux's `uvcvideo` (which carries a
  large quirks table for exactly this kind of cheap bridge chip) copes with
  it where raw libuvc negotiation did not.
- **New eye camera** (Arducam UC-844, OV9281 global-shutter mono — reuses
  the *same* USB ID `0c45:6366` as the old camera, disambiguate by serial
  `UC762`, not VID:PID): negotiates UVC cleanly, clean monotonic PTS/SCR on
  the same ~15MHz clock family as the scene camera. Confirmed via
  `ffplay`/AVFoundation it streams at 1280x800@120fps (global shutter — good
  for pupil tracking, avoids rolling-shutter skew). Note: MJPEG at its
  advertised default frame interval failed to negotiate (a descriptor quirk
  — default interval didn't match the discrete interval list) but YUY2
  uncompressed at 1280x720@10fps worked fine; expect to need to iterate
  formats/sizes here too rather than assuming the first one works.

**The macOS blocker that's driving this migration:** `uvc_open()` returns
`Access denied` unless run as root, because macOS auto-claims the UVC
streaming interface the instant a device enumerates (unlike Linux, which
lets you `modprobe -r uvcvideo` or just isn't exclusive to begin with). Even
running as root, libuvc's raw USB claim is *exclusive* and can't coexist
with the app's existing `cv2.VideoCapture`/AVFoundation capture path. Getting
real hardware timestamps on Mac would mean running the whole app as root
every session AND rewriting the capture layer away from OpenCV. On Linux,
V4L2 (the standard camera API, and what OpenCV already uses as its Linux
backend) exposes kernel-validated per-frame timestamps as a first-class
citizen with none of that — no root, no exclusivity fight.

**Important nuance carried over from the Mac session — don't oversell this
to yourself while testing:** hardware PTS/SCR (or V4L2 timestamps) prove a
single camera's *own* frame timing is precise and glitch-free. They do
**not** by themselves give you cross-camera sync — the eye and scene camera
have independent, non-genlocked crystals with no shared epoch, and will
drift relative to each other over time. Real cross-camera low-ms sync needs
either (a) both cameras timestamped against the *same host* monotonic clock
at kernel buffer-completion time (this is what V4L2 gives you cheaply and
is the practical target for this migration), or (b) actual hardware
trigger/genlock wiring between the two sensors (bigger hardware project,
not in scope for this pass — though worth a quick check of whether the
Arducam UC-844/OV9281 module exposes a trigger pin, since ArduCam sells
genlock stereo kits on this same sensor).

## What to actually do on the Jetson

Work through these in order. Stop and report back (don't push through
silently) if a step fails in a way that isn't just "format not supported,
try the next one."

### 1. Environment sanity check
- Confirm `v4l2-utils` is installed (`v4l2-ctl --version`); install via
  `apt` if not.
- `v4l2-ctl --list-devices` — enumerate what's connected and which
  `/dev/videoN` node maps to which camera. Cross-reference against
  `lsusb` (look for `0bda:d565` = scene cam, `0c45:6366` = one of the two
  eye cams — check `lsusb -v` or `udevadm info` for the serial number
  `SN0001` vs `UC762` to tell the old Sonix and new Arducam apart, since
  they share a USB ID).
- `dmesg | grep -i uvc` — see how the kernel driver characterized each
  device at enumeration time. This is a genuinely different code path than
  the libuvc userspace probe we ran on Mac, so don't assume the Mac
  failure log tells you what this will say.

### 2. Basic streaming test (plug in the cameras one at a time, or note
   they may plug into different `/dev/videoN` nodes if simultaneous)
- `v4l2-ctl -d /dev/videoN --list-formats-ext` — see what resolutions/rates
  each camera actually advertises to the kernel.
- `ffplay /dev/videoN` (or `-input_format mjpeg`/`-pixel_format yuyv422`
  plus explicit `-video_size`/`-framerate` if the default negotiation
  picks something you don't want) to confirm you can actually see a picture.
  This is the same kind of check we did on Mac with
  `ffplay -f avfoundation`.
- **Prioritize retesting the old Sonix eye camera here.** If it streams at
  all via V4L2/ffplay, that alone is useful signal (Linux's driver is more
  tolerant than raw libuvc negotiation was on Mac) — but also try
  `v4l2-ctl --stream-mmap --stream-count=100 --stream-to=/dev/null -d
  /dev/videoN` and watch for dropped-frame/negotiation errors under
  sustained streaming, not just a single preview frame.

### 3. Timestamp validation (the actual point of this exercise)
Goal: confirm V4L2 gives clean, monotonic, kernel-assigned per-frame
timestamps for each camera, analogous to what we validated for PTS/SCR on
Mac but via the API this pipeline would actually use.
- Simplest path: a short Python script using `v4l2py` (`pip install
  v4l2py` inside the `et` conda env, or a fresh venv if easier on the
  Jetson) to open each device and read frames, printing each frame's raw
  V4L2 buffer timestamp (not a Python-side `time.time()` call after the
  fact — the buffer's own `timestamp` field, which V4L2 stamps at DMA/
  interrupt completion). Confirm:
  - timestamps are strictly monotonically increasing, no backward jumps
  - deltas between consecutive frames are consistent with the negotiated
    frame rate (analogous to the "PTS delta should be constant" check we
    did on Mac)
  - check what timestamp *type* the driver reports
    (`V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` vs other flags — you want
    monotonic, not the wall-clock/copy variants) since that determines
    whether it's safe to use directly as a cross-camera sync reference.
- Do this for the scene camera and whichever eye camera(s) actually stream.
- If `v4l2py` isn't available/easy to get working, `cv2.VideoCapture(idx,
  cv2.CAP_V4L2)` + `cap.get(cv2.CAP_PROP_POS_MSEC)` is a fallback but is
  weaker evidence (that property isn't guaranteed to be the raw kernel
  buffer timestamp for all drivers) — prefer the raw ioctl/`v4l2py` route
  if you have time.

### 4. Cross-camera check (only after step 3 looks clean on both cameras)
- With scene cam and whichever eye camera works both streaming
  simultaneously, confirm both timestamp streams are on the *same* clock
  domain (both `CLOCK_MONOTONIC`-based via V4L2, as expected on Linux) so
  that a single host timestamp per frame is actually a valid sync
  reference across the two devices — this is the property that was
  fundamentally unavailable on Mac.

## What "success" looks like for this session

Not "port the whole app to Jetson" — that's a separate, later step. Success
here is narrowly:
1. Clear yes/no on whether V4L2 gives clean monotonic timestamps for both
   cameras, with actual delta/jitter numbers (mirroring the PTS delta
   analysis from the Mac session).
2. Clear yes/no on whether the old Sonix eye camera streams at all under
   Linux's `uvcvideo`, since that reopens the option of not having to
   remount the rig for the Arducam.
3. A short summary suitable for feeding back into the Mac-side project
   memory (`project_hardware_timestamps.md`) — findings only, not a
   decision about the migration itself.

## Reference data (only if useful for building a probe like the Mac one)

USB IDs and serials, all confirmed via `system_profiler
SPUSBDataType`/`lsusb` on the Mac side:
- Scene camera: `0bda:d565` (Realtek/OV5640)
- Old eye camera: `0c45:6366`, serial `SN0001` (Sonix GC0308)
- New eye camera: `0c45:6366`, serial `UC762` (Arducam UC-844 / OV9281)

If it turns out useful to compare against the exact libuvc-level failure
from Mac, the old eye camera advertised these formats/frames (all of which
failed `uvc_get_stream_ctrl_format_size` with `Invalid mode (-51)` in raw
libuvc — this has no bearing on whether V4L2 will succeed, it's just the
fixture for comparison):
- MJPEG: 320x240, 640x480, 160x120 (all @30fps, one variant @120fps)
- YUY2 (uncompressed): 320x240, 640x480, 160x120 (all @30fps)

## Don't do yet

- Don't start porting `scripts/eyetracker/cameras/opencv_source.py` or the
  exposure-control (`uvc_util.py`/IOKit) layer to Linux/V4L2 in this
  session unless explicitly asked — this pass is measurement only, to
  decide *whether* to commit to the migration, not to build it.
- Don't touch git remotes, commit, or push anything without asking first —
  standard project rule (see root `CLAUDE.md`), applies on this machine too.
