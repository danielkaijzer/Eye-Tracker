"""One-shot software auto exposure: settle, then leave the exposure locked.

The eye cam's own auto exposure keeps adjusting during a recording and never
reports the exposure it picked, which makes each frame's exposure (and so where
the frame's timestamp sits relative to mid-exposure) unknown. Instead the App
opens the eye cam in manual mode and, at startup, runs settle_exposure() once:
step the manual exposure until the eye image's median gray level hits a target,
then stop. Nothing changes the exposure afterwards unless the user asks
(App keys), and never during a calibration, so a session has one known
exposure, recorded in metadata.json.

Target: a median of ~96 matches what the camera's own AE chose on the eye in
the rig's well-calibrated sessions (2026-09-23: medians 89-107, glint only
saturated). Brightness vs exposure is monotonic but not linear (MJPEG gamma +
~16-24 black level), so each step is a proportional guess on the signal above
black, kept inside a bracket that shrinks every iteration.
"""
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from scripts.eyetracker.cameras.base import CameraSource

_BLACK_LEVEL = 16.0


@dataclass
class SettleResult:
    value: int              # exposure left on the camera (100 us units)
    median: float           # eye-image median gray at that exposure
    converged: bool         # within tolerance of the target
    iterations: int
    note: str = ""


def _measure(cam: CameraSource, after_ts: float, frames: int,
             timeout_s: float) -> Optional[float]:
    """Median gray over `frames` frames captured after `after_ts` (host
    monotonic s), i.e. exposed with the new setting."""
    vals = []
    deadline = time.monotonic() + timeout_s
    while len(vals) < frames and time.monotonic() < deadline:
        frame = cam.read()
        ts = cam.last_timestamp
        if frame is None or ts is None or ts < after_ts:
            continue
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(float(np.median(gray)))
    return float(np.median(vals)) if vals else None


def settle_exposure(cam: CameraSource, target: float, tolerance: float = 0.08,
                    max_iters: int = 10, frames: int = 3,
                    apply_delay_s: float = 0.03) -> Optional[SettleResult]:
    """Adjust `cam`'s manual exposure until the image median is within
    `tolerance` (fraction) of `target`, then leave it there. Returns None if
    the camera has no exposure control. apply_delay_s: how long after a change
    frames still carry the old exposure (~3 frame periods at 100 fps)."""
    limits = cam.exposure_limits()
    if limits is None:
        return None
    lo, hi = limits
    value = cam.exposure_value()
    value = (lo + hi) // 2 if value is None else int(max(lo, min(hi, value)))
    best = None                                  # (|error|, value, median)
    # Exclusive bounds of the values still worth trying: start just outside
    # the limits (nothing measured yet), then move in to each measured value.
    bracket_lo, bracket_hi = lo - 1, hi + 1
    it = 0
    for it in range(1, max_iters + 1):
        cam.set_exposure(value)
        median = _measure(cam, time.monotonic() + apply_delay_s, frames, timeout_s=1.0)
        if median is None:
            return SettleResult(value, float("nan"), False, it, "no frames")
        err = abs(median - target)
        if best is None or err < best[0]:
            best = (err, value, median)
        if err <= tolerance * target:
            return SettleResult(value, median, True, it)
        if median < target:
            bracket_lo = value
        else:
            bracket_hi = value
        if value >= hi and median < target:
            break                                # too dark even at max
        if value <= lo and median > target:
            break                                # too bright even at min
        if bracket_hi - bracket_lo <= 1:
            break
        guess = value * (target - _BLACK_LEVEL) / max(median - _BLACK_LEVEL, 1.0)
        nxt = int(round(max(bracket_lo + 1, min(bracket_hi - 1, guess))))
        if nxt == value:
            nxt = (bracket_lo + bracket_hi) // 2
        value = nxt
    _, value, median = best
    cam.set_exposure(value)
    if value >= hi and median < target:
        note = "image still dark at the longest exposure that fits a frame"
    elif value <= lo and median > target:
        note = "image still bright at the shortest exposure"
    else:
        note = "closest value found"
    return SettleResult(value, median, False, it, note)
