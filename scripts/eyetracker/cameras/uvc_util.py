"""macOS exposure control via the uvc-util CLI (jtfrey/uvc-util).

Why this exists: OpenCV's AVFoundation backend can't drive UVC exposure on
macOS — set() no-ops and get() returns 0. uvc-util sends UVC class requests
over IOKit *independently* of the capture session, so it works while OpenCV is
streaming. On the rig's modules `exposure-time-abs` turned out to be a cosmetic
no-op (writes are accepted but never reach the sensor), while `gain` and
`auto-exposure-mode` are genuinely wired — so gain is the brightness lever here
and auto-exposure-mode=1 is the "stop auto-exposure" switch. See the
project_macos_uvc_exposure memory.

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


# UVC auto-exposure-mode is an 8-bit bitmap: 1=manual, 2=auto, 4=shutter-priority,
# 8=aperture-priority. Devices implement a subset; the rig cams support {1, 8}.
_MANUAL_MODE = 1
_DEFAULT_AUTO_MODE = 8  # fallback only; the real auto value is read per-device.


class UvcExposureController:
    """Drives one camera's exposure via uvc-util, selecting it by USB
    vendor:product (e.g. "0x0c45:0x6366"). Selecting by id rather than index
    keeps it stable across replug / enumeration order."""

    def __init__(self, uvc_id: str, binary: Optional[str]):
        self.uvc_id = uvc_id
        self.binary = binary
        self._sel = f"--select-by-vendor-and-product-id={uvc_id}"
        self.ok = False
        self._auto: Optional[bool] = None     # last commanded auto state
        self._auto_mode = _DEFAULT_AUTO_MODE   # device's "auto" mode value
        self._gain: Optional[int] = None
        self._gain_min = 0
        self._gain_max = 100

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
        out = self._run("-o", control)
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

    # ---- lifecycle ----------------------------------------------------------

    def probe(self) -> bool:
        """Confirm the device is reachable and learn its auto-mode value and
        gain range. Returns False if uvc-util or the device is unavailable."""
        if not self.binary:
            return False
        mode = self._get_int("auto-exposure-mode")
        if mode is None:
            return False
        if mode != _MANUAL_MODE:
            # Whatever non-manual mode it's in now is its "auto" — remember it
            # so toggling back to auto restores the right value.
            self._auto_mode = mode
        self._auto = (mode == self._auto_mode)
        self._gain = self._get_int("gain")
        lo, hi = self._get_range("gain")
        if lo is not None and hi is not None:
            self._gain_min, self._gain_max = lo, hi
        self.ok = True
        return True

    def apply_initial(self, auto: Optional[bool], gain: Optional[int]) -> None:
        if auto is not None:
            self.set_auto(auto)
        if gain is not None and auto is not True:
            self.set_gain(gain)

    # ---- controls -----------------------------------------------------------

    def set_auto(self, auto: bool) -> bool:
        mode = self._auto_mode if auto else _MANUAL_MODE
        if self._run("-s", f"auto-exposure-mode={mode}") is None:
            return False
        self._auto = auto
        # Re-assert gain when returning to manual, in case it drifted.
        if not auto and self._gain is not None:
            self._run("-s", f"gain={self._gain}")
        return True

    def set_gain(self, value: int) -> bool:
        value = max(self._gain_min, min(self._gain_max, int(value)))
        # Gain only sticks in manual mode.
        if self._auto is not False:
            if self._run("-s", f"auto-exposure-mode={_MANUAL_MODE}") is None:
                return False
            self._auto = False
        if self._run("-s", f"gain={value}") is None:
            return False
        self._gain = value
        return True

    # ---- readout ------------------------------------------------------------

    def state(self) -> Tuple[Optional[bool], Optional[int]]:
        return (self._auto, self._gain)

    def status_str(self) -> str:
        if self._auto:
            return "auto"
        if self._gain is not None:
            return f"gain {self._gain}"
        return "manual"
