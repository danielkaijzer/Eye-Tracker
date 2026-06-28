"""macOS exposure control via the uvc-util CLI (jtfrey/uvc-util).

Why this exists: OpenCV's AVFoundation backend can't drive UVC exposure on
macOS — set() no-ops and get() returns 0. uvc-util sends UVC class requests
over IOKit *independently* of the capture session, so it works while OpenCV is
streaming.

Scope: this drives the **scene camera's** manual exposure only. The scene module
(Realtek OV5640) is manual-only and its real brightness lever is
`exposure-time-abs` (gain just scales output luminance). The eye module's
exposure runs internally on the sensor and isn't controllable over UVC, so the
app doesn't drive it — poke it from the terminal if needed
(docs/uvc_exposure_cheatsheet.md). The controller stays parameterized on the
control name so it isn't hard-wired to one cam.

Gotcha this defends against: uvc-util's `-s` (set) returns exit 0 even when the
device silently ignores or clamps the write, so every set is confirmed with a
read-back.

The binary is found via (in order): explicit path arg, UVC_UTIL_PATH env var,
PATH, then common Homebrew/usr-local locations. Build it once with the Xcode
CLT — see uvc-util's README — and drop it on PATH or point UVC_UTIL_PATH at it.
"""
import os
import re
import shutil
import subprocess
from typing import Optional, Tuple


def find_uvc_util(explicit: Optional[str] = None) -> Optional[str]:
    """Return a usable uvc-util binary path, or None if not installed."""
    candidates = (
        explicit,
        os.environ.get("UVC_UTIL_PATH"),
        shutil.which("uvc-util"),
        "/usr/local/bin/uvc-util",
        "/opt/homebrew/bin/uvc-util",
    )
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


# auto-exposure-mode bitmap value for manual exposure. We force this at probe so
# the manual lever actually drives the sensor.
_MANUAL_MODE = 1


class UvcExposureController:
    """Drives one camera's manual exposure via uvc-util, selecting it by USB
    vendor:product (e.g. "0x0bda:0xd565"). Selecting by id rather than index
    keeps it stable across replug / enumeration order.

    `control` names the UVC control used as the manual exposure lever
    (default "exposure-time-abs")."""

    def __init__(self, uvc_id: str, binary: Optional[str],
                 control: str = "exposure-time-abs"):
        self.uvc_id = uvc_id
        self.binary = binary
        self.control = control                 # manual lever (UVC control name)
        self._sel = f"--select-by-vendor-and-product-id={uvc_id}"
        self.ok = False
        self._value: Optional[int] = None      # last commanded value of `control`
        self._value_min = 0
        self._value_max = 0

    # ---- process plumbing ---------------------------------------------------

    def _run(self, *args: str) -> Optional[str]:
        if not self.binary:
            return None
        try:
            done = subprocess.run([self.binary, self._sel, *args],
                                  capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    def _get_int(self, control: str) -> Optional[int]:
        out = self._run("-g", control)
        if out is None:
            return None
        match = re.search(r"-?\d+", out)
        return int(match.group()) if match else None

    def _get_range(self, control: str) -> Tuple[Optional[int], Optional[int]]:
        out = self._run("-S", control)
        if out is None:
            return (None, None)
        lo = re.search(r"minimum:\s*(-?\d+)", out)
        hi = re.search(r"maximum:\s*(-?\d+)", out)
        return (int(lo.group(1)) if lo else None,
                int(hi.group(1)) if hi else None)

    def _set_verified(self, control: str, value: int) -> Optional[int]:
        """Write a control and confirm it took. uvc-util reports success even
        when the device ignores or clamps the write, so we read the value back.
        Returns the value the device actually holds (which may differ from the
        request if it snapped/clamped), or None if the write or read failed."""
        if self._run("-s", f"{control}={value}") is None:
            return None
        return self._get_int(control)

    # ---- lifecycle ----------------------------------------------------------

    def probe(self) -> bool:
        """Confirm the device is reachable, force manual exposure, and read the
        lever's range + current value. Returns False if uvc-util or the device
        is unavailable."""
        if not self.binary:
            return False
        lo, hi = self._get_range(self.control)
        if lo is None or hi is None:
            return False
        self._value_min, self._value_max = lo, hi
        self._set_verified("auto-exposure-mode", _MANUAL_MODE)
        self._value = self._get_int(self.control)
        self.ok = True
        return True

    def apply_initial(self, value: Optional[int]) -> None:
        if value is not None:
            self.set_exposure(value)

    # ---- controls -----------------------------------------------------------

    def set_exposure(self, value: int) -> bool:
        value = int(max(self._value_min, min(self._value_max, value)))
        got = self._set_verified(self.control, value)
        if got is None:
            return False
        self._value = got
        return True

    def nudge_exposure(self, direction: int, step_fraction: float) -> bool:
        """Step the lever by a fraction of its range (direction +1/-1). A
        fraction keeps the feel consistent across controls whose ranges differ
        by orders of magnitude (exposure-time ~1..10000 vs gain ~0..128)."""
        if self._value is None:
            return False
        span = self._value_max - self._value_min
        step = max(1, round(span * step_fraction))
        return self.set_exposure(self._value + direction * step)

    # ---- readout ------------------------------------------------------------

    def status_str(self) -> str:
        if self._value is None:
            return "manual"
        label = "exp" if self.control == "exposure-time-abs" else self.control
        return f"{label} {self._value}"
