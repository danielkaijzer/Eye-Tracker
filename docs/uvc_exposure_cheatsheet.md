# UVC camera control cheat sheet (macOS)

Quick reference for driving the rig's cameras with **uvc-util** from the command
line, independent of the app. OpenCV/AVFoundation can't set exposure on macOS, so
this is the only working path. See `docs/` and the `project_macos_uvc_exposure`
notes for the why.

## Setup

The binary lives at `/opt/homebrew/bin/uvc-util` (on PATH). Every command selects
a camera by its USB **vendor:product** id (stable across replug, unlike the index):

```bash
B=/opt/homebrew/bin/uvc-util
SCENE="--select-by-vendor-and-product-id=0x0bda:0xd565"   # Realtek bridge / OV5640
EYE="--select-by-vendor-and-product-id=0x0c45:0x6366"     # Sonix GC0308
```

List what's connected / what a camera supports:

```bash
$B -d                       # list all UVC devices with their vendor:product ids
$B $SCENE -c                # list controls this camera implements
$B $SCENE -S exposure-time-abs   # show range/default for one control
$B $SCENE -S '*'            # show range/default for ALL controls
```

> **Heads-up:** `-s` (set) returns success even when the camera silently ignores
> or clamps the write. Always confirm with `-g` (get).

---

## Scene camera (`0x0bda:0xd565`)

This is the one that matters for ArUco. **`exposure-time-abs` is the real lever**
(it drives sensor integration time); `gain` only scales output luminance and adds
noise. This module is **manual-only** — it accepts no auto-exposure mode.

| Control | Range | Default | What it does |
|---|---|---|---|
| `exposure-time-abs` | 1–10000 | ~332 | sensor integration time — **the lever** |
| `gain` | 0–128 | 64 | output luminance + noise |
| `auto-exposure-mode` | 1 only | — | manual-only; auto writes revert to 1 |

```bash
# Read current state
$B $SCENE -g exposure-time-abs
$B $SCENE -g gain

# Make sure it's in manual (required for exposure-time to take effect)
$B $SCENE -s auto-exposure-mode=1

# Set exposure time (lower = darker/sharper, higher = brighter/more motion blur)
$B $SCENE -s exposure-time-abs=400

# Reset gain to a sane value (it persists wherever you last left it — the app
# doesn't touch it). 64 is the device default; lower = darker + less noise.
$B $SCENE -s gain=64
```

**Fixing blown-out ArUco markers** (the original problem): markers blow out when
exposure is too high. Drop `exposure-time-abs` until the white quiet-zone stops
clipping, keep `gain` moderate (lower = cleaner edges for detection):

```bash
$B $SCENE -s auto-exposure-mode=1
$B $SCENE -s gain=32
$B $SCENE -s exposure-time-abs=250   # tune down until markers are crisp, not glowing
```

---

## Eye camera (`0x0c45:0x6366`)

**The eye cam's auto-exposure runs internally on the sensor/bridge and can't be
disabled over UVC.** `auto-exposure-mode` writes register (read back as 1 or 8)
but don't stop the internal AEC, so there's no true manual exposure here.
`gain` is the only control with any image effect (scales output luminance), and
even that gets partly compensated by the AEC. `exposure-time-abs` is a no-op.
None of this matters in practice — the IR-lit pupil doesn't need exposure control.

| Control | Range | Default | What it does |
|---|---|---|---|
| `gain` | 0–100 | 0 | scales luminance; partly fought by the internal AEC |
| `exposure-time-abs` | 1–5000 | 157 | no-op on this module |
| `auto-exposure-mode` | 1, 8 | 8 | register toggles but doesn't disable the internal AEC |

```bash
$B $EYE -g gain
$B $EYE -s gain=10      # has some effect, but the AEC will compensate
```

---

## In-app hotkeys

The app drives the **scene cam's** `exposure-time-abs` live (the eye cam is not
driven from the app — use the terminal commands above for it):

| Key | Action |
|---|---|
| `[` / `]` | darker / brighter — **fine** step (~10 units) |
| `{` / `}` | darker / brighter — **coarse** step (~500 units) |

On-screen readout shows e.g. `scene:exp 400`.
