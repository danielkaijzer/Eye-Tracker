"""Linux camera plumbing: V4L2 device enumeration + exposure via v4l2-ctl.

Why this exists: on Linux each UVC camera exposes TWO /dev/video nodes (a
capture node and a metadata node), so cv2 indexes are not contiguous (eye=0,
scene=2, ...) and "the other index" is not a safe way to find the second
camera. sysfs tells us which node is the capture node and which USB
vendor:product it belongs to, so cameras can be picked by id instead.

Exposure: the Linux counterpart of uvc_util.UvcExposureController. Same public
surface (probe / apply_initial / nudge_exposure / status_str) so OpenCVCamera
can hold either. Driven through `v4l2-ctl` (apt: v4l-utils) rather than
cv2.CAP_PROP_EXPOSURE because v4l2-ctl reports the control's real range and
readback, which cv2 does not.
"""
import glob
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

_SYSFS = "/sys/class/video4linux"


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _usb_id(node: str) -> Optional[str]:
    """"0xVVVV:0xPPPP" for a videoN sysfs node, or None if not a USB device.
    `device` links to the USB interface; idVendor/idProduct live on its parent."""
    usb_dev = os.path.dirname(os.path.realpath(os.path.join(_SYSFS, node, "device")))
    vid = _read(os.path.join(usb_dev, "idVendor"))
    pid = _read(os.path.join(usb_dev, "idProduct"))
    if vid is None or pid is None:
        return None
    return f"0x{vid}:0x{pid}"


def list_capture_devices() -> List[Tuple[int, Optional[str], str]]:
    """(index, usb_id, name) for every V4L2 capture node, sorted by index.
    Skips the per-camera metadata nodes (sysfs `index` != 0)."""
    devices = []
    for path in glob.glob(os.path.join(_SYSFS, "video*")):
        node = os.path.basename(path)
        if _read(os.path.join(path, "index")) not in (None, "0"):
            continue
        try:
            idx = int(node[len("video"):])
        except ValueError:
            continue
        devices.append((idx, _usb_id(node), _read(os.path.join(path, "name")) or node))
    return sorted(devices)


def index_for_usb_id(usb_id: str, exclude: Optional[int] = None) -> Optional[int]:
    """First capture-node index whose USB vendor:product matches `usb_id`."""
    want = usb_id.lower()
    for idx, uid, _ in list_capture_devices():
        if idx != exclude and uid is not None and uid.lower() == want:
            return idx
    return None


def metadata_node_for(capture_index: int) -> Optional[str]:
    """/dev/videoN of the UVC metadata node paired with a capture node (same
    USB interface, sysfs `index` 1), or None if the camera has none."""
    want = os.path.realpath(os.path.join(_SYSFS, f"video{capture_index}", "device"))
    for path in sorted(glob.glob(os.path.join(_SYSFS, "video*"))):
        if _read(os.path.join(path, "index")) != "1":
            continue
        if os.path.realpath(os.path.join(path, "device")) == want:
            return "/dev/" + os.path.basename(path)
    return None


def uvcvideo_flag(name: str) -> Optional[bool]:
    """A boolean-ish uvcvideo module parameter (e.g. "nodrop",
    "hwtimestamps"), or None if uvcvideo isn't loaded."""
    val = _read(f"/sys/module/uvcvideo/parameters/{name}")
    return None if val is None else val.strip() not in ("0", "N")


def uvc_timestamp_setup_problem() -> Optional[str]:
    """Why camera-clock timestamps (cameras/uvc_clock.py) can't work with the
    current uvcvideo settings, or None if they can. Needs nodrop=1 (else the
    metadata node delivers nothing) and hwtimestamps=0 (else video buffer
    timestamps are rewritten by the kernel's conversion, which drifts on the
    eye cam, and no longer match their metadata buffers). Persist with
    `options uvcvideo nodrop=1 hwtimestamps=0` in /etc/modprobe.d/uvcvideo.conf
    and reload the module."""
    nodrop, hw = uvcvideo_flag("nodrop"), uvcvideo_flag("hwtimestamps")
    if nodrop is None:
        return "uvcvideo not loaded"
    problems = []
    if not nodrop:
        problems.append("nodrop=0 (need 1)")
    if hw:
        problems.append("hwtimestamps=1 (need 0)")
    return ", ".join(problems) or None


# ---- exposure ---------------------------------------------------------------

# v4l2 control names changed in kernel 5.x (exposure_absolute ->
# exposure_time_absolute, exposure_auto -> auto_exposure); accept either.
_LEVER_NAMES = ("exposure_time_absolute", "exposure_absolute")
_AUTO_NAMES = ("auto_exposure", "exposure_auto")
# auto_exposure menu value for Manual Mode (UVC: 1=manual, 3=aperture priority).
_MANUAL_MODE = 1
# UVC "auto exposure priority": when on, the camera may stretch the frame
# period to fit a long exposure (e.g. 30 -> ~25 fps at exposure 332 = 33.2 ms).
# Off pins the frame rate — which cross-camera sync relies on — and caps the
# effective exposure at one frame period instead.
_DYNAMIC_FPS = "exposure_dynamic_framerate"

_CTRL_RE = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")


def find_v4l2_ctl() -> Optional[str]:
    return shutil.which("v4l2-ctl")


class V4l2ExposureController:
    """Drives one camera's manual exposure via v4l2-ctl on /dev/video<index>."""

    def __init__(self, index: int, binary: Optional[str]):
        self.device = f"/dev/video{index}"
        self.binary = binary
        self.ok = False
        self._lever: Optional[str] = None
        self._value: Optional[int] = None
        self._value_min = 0
        self._value_max = 0

    def _run(self, *args: str) -> Optional[str]:
        if not self.binary:
            return None
        try:
            done = subprocess.run([self.binary, "-d", self.device, *args],
                                  capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    def _controls(self) -> Dict[str, Dict[str, str]]:
        out = self._run("--list-ctrls") or ""
        ctrls = {}
        for line in out.splitlines():
            m = _CTRL_RE.match(line)
            if m:
                ctrls[m.group(1)] = dict(kv.split("=", 1) for kv in m.group(3).split()
                                         if "=" in kv)
        return ctrls

    def _get_int(self, control: str) -> Optional[int]:
        out = self._run("-C", control)
        match = re.search(r":\s*(-?\d+)", out or "")
        return int(match.group(1)) if match else None

    def _set_verified(self, control: str, value: int) -> Optional[int]:
        if self._run("-c", f"{control}={value}") is None:
            return None
        return self._get_int(control)

    def probe(self) -> bool:
        """Find the exposure lever, force manual mode + a fixed frame rate,
        read range + value."""
        ctrls = self._controls()
        lever = next((n for n in _LEVER_NAMES if n in ctrls), None)
        if lever is None:
            return False
        auto = next((n for n in _AUTO_NAMES if n in ctrls), None)
        if auto is not None:
            self._set_verified(auto, _MANUAL_MODE)
        if _DYNAMIC_FPS in ctrls:
            self._set_verified(_DYNAMIC_FPS, 0)
        info = ctrls[lever]
        self._lever = lever
        self._value_min = int(info.get("min", 0))
        self._value_max = int(info.get("max", 0))
        self._value = self._get_int(lever)
        self.ok = True
        return True

    def apply_initial(self, value: Optional[int]) -> None:
        if value is not None:
            self.set_exposure(value)

    def set_exposure(self, value: int) -> bool:
        if self._lever is None:
            return False
        value = int(max(self._value_min, min(self._value_max, value)))
        got = self._set_verified(self._lever, value)
        if got is None:
            return False
        self._value = got
        return True

    @property
    def value(self) -> Optional[int]:
        """Last confirmed exposure value (device units), or None."""
        return self._value

    @property
    def value_range(self) -> Tuple[int, int]:
        return self._value_min, self._value_max

    def nudge_exposure(self, direction: int, step_fraction: float) -> bool:
        if self._value is None:
            return False
        span = self._value_max - self._value_min
        step = max(1, round(span * step_fraction))
        return self.set_exposure(self._value + direction * step)

    def status_str(self) -> str:
        return "manual" if self._value is None else f"exp {self._value}"
