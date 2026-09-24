"""Per-frame camera-clock timestamps from UVC PTS/SCR, converted in userspace.

Why not the kernel's conversion (uvcvideo hwtimestamps=1): it maps the device
clock (STC) to host time *through the USB frame counter (SOF) the device
reports in each SCR*, assuming that counter is the bus's. The eye cam's bridge
reports a free-running counter instead (~1003 ticks/s, drifting against the
bus), so the kernel's timestamps drift ~3000 ppm and jump by ~120-250 ms
whenever the counter slips. The same defect in some Logitech cams got a quirk
(ignore the device SOF) in kernel 6.8+; the Jetson runs 5.15.

What this does instead: every SCR pairs an STC value with the host time the
kernel processed that USB packet (MetaEntry.host_ns). Host processing only
ever adds latency, so the *lower envelope* of (STC, host time) is the device
clock mapped onto the host clock. DeviceClock fits that envelope over a
sliding window (tracking crystal drift, ~+95 ppm on the eye cam), and each
frame's PTS is converted through it. Measured on the rig: envelope residual
~0.06-0.08 ms, no drift, no jumps. The envelope can only see device time plus
the *minimum* host processing delay, a small per-camera constant that ends up
in the per-camera offset below.

What the timestamp means: both rig cams set PTS ~0.06-0.3 ms before the
frame's first USB packet, i.e. when the camera starts *sending* the frame
(after exposure, readout and JPEG encoding). The fixed exposure-to-PTS delay
per camera is what scripts/extras/flash_sync_test.py measures.

Pairing with video frames: with hwtimestamps=0, a V4L2 video buffer's
timestamp and its metadata buffer's timestamp are the same value (the kernel
copies one into the other), so an OpenCV frame's CAP_PROP_POS_MSEC finds its
metadata frame exactly.

Needs uvcvideo `nodrop=1 hwtimestamps=0`, e.g. in /etc/modprobe.d/uvcvideo.conf:
    options uvcvideo nodrop=1 hwtimestamps=0
"""
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from scripts.eyetracker.cameras.uvc_meta import MetaEntry, UvcMetaReader

_WRAP = 2 ** 32


class _Unwrapper:
    """Unwrap a monotonically increasing 32-bit counter."""

    def __init__(self):
        self._last: Optional[int] = None
        self._base = 0

    def __call__(self, raw: int) -> int:
        if self._last is not None and raw < self._last - _WRAP // 2:
            self._base += _WRAP
        self._last = raw
        return self._base + raw


class DeviceClock:
    """Sliding-window map from device clock ticks (unwrapped STC) to host
    monotonic seconds, fitted to the lower envelope of SCR samples."""

    def __init__(self, nominal_hz: float = 15e6, window_s: float = 10.0,
                 refit_s: float = 0.5, min_span_s: float = 1.0,
                 bin_s: float = 0.5, keep_frac: float = 0.25):
        self.nominal_hz = nominal_hz
        self.window_s = window_s
        self.refit_s = refit_s
        self.min_span_s = min_span_s
        self.bin_s = bin_s
        self.keep_frac = keep_frac
        self._samples: Deque[Tuple[int, float]] = deque()   # (stc_u, host_s)
        self._fit: Optional[Tuple[float, float, int, float]] = None  # a, b, x0, y0
        self._last_fit_at = float("-inf")
        self.envelope_residual_ms: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self._fit is not None

    @property
    def rate_hz(self) -> Optional[float]:
        return None if self._fit is None else 1.0 / self._fit[0]

    def reset(self) -> None:
        self._samples.clear()
        self._fit = None
        self._last_fit_at = float("-inf")

    def add(self, stc_u: int, host_s: float) -> None:
        """One SCR sample: device ticks + host time the packet was processed.
        Callers pass the lowest-latency sample per frame (see add_frame)."""
        if self._fit is not None:
            # A device-clock jump (camera reset / restart) shows up as a sample
            # far off the fit; start over rather than bend the fit around it.
            if abs(host_s - self.to_host(stc_u)) > 0.05:
                self.reset()
        self._samples.append((stc_u, host_s))
        while self._samples and host_s - self._samples[0][1] > self.window_s:
            self._samples.popleft()
        span = host_s - self._samples[0][1]
        if span >= self.min_span_s and host_s - self._last_fit_at >= self.refit_s:
            self._refit()
            self._last_fit_at = host_s

    def _refit(self) -> None:
        arr = np.array(self._samples, dtype=np.float64)
        x0, y0 = arr[0, 0], arr[0, 1]
        x, y = arr[:, 0] - x0, arr[:, 1] - y0
        a, b = np.polyfit(x, y, 1)
        bins = np.floor(y / self.bin_s).astype(int)
        keep = np.ones(len(x), bool)
        for _ in range(3):
            r = y - (a * x + b)
            keep[:] = False
            for k in np.unique(bins):
                idx = np.flatnonzero(bins == k)
                n = max(2, int(len(idx) * self.keep_frac))
                keep[idx[np.argsort(r[idx])[:n]]] = True
            a, b = np.polyfit(x[keep], y[keep], 1)
        rate = 1.0 / a
        if abs(rate / self.nominal_hz - 1.0) > 0.01:
            return   # nonsense fit (too little data / garbage); keep the old one
        r = y - (a * x + b)
        self.envelope_residual_ms = float(np.std(r[keep]) * 1e3)
        self._fit = (a, b, int(x0), y0)

    def to_host(self, stc_u: int) -> float:
        a, b, x0, y0 = self._fit
        return a * (stc_u - x0) + b + y0


class UvcTimestamper:
    """Reads one camera's metadata node on a thread and answers "what is the
    camera-clock time of the video frame whose V4L2 timestamp is T?"."""

    def __init__(self, meta_device: str, keep_s: float = 2.0):
        self.meta_device = meta_device
        self.keep_s = keep_s
        self.clock = DeviceClock()
        self._reader = UvcMetaReader(meta_device, num_buffers=32)
        self._cond = threading.Condition()
        self._pending: Dict[int, Tuple[int, int]] = {}   # buf ts (us) -> (pts_u, seq)
        self._order: Deque[Tuple[float, int]] = deque()  # (host s, key) for pruning
        self._stc = _Unwrapper()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        self._reader.open()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"{self.meta_device}-uvcclock")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._reader.close()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while self._running:
            try:
                frame = self._reader.read(0.2)
            except OSError as e:          # camera unplugged / reset
                self.error = str(e)
                return
            if frame is None or frame.pts is None:
                continue
            first = self._add_frame(frame.entries)
            if first is None:
                continue
            stc_raw, stc_u = first
            pts_u = stc_u - ((stc_raw - frame.pts) % _WRAP)
            key = frame.buf_ts_ns // 1000
            now = time.monotonic()
            with self._cond:
                self._pending[key] = (pts_u, frame.sequence)
                self._order.append((now, key))
                while self._order and now - self._order[0][0] > self.keep_s:
                    self._pending.pop(self._order.popleft()[1], None)
                self._cond.notify_all()

    def _add_frame(self, entries: List[MetaEntry]) -> Optional[Tuple[int, int]]:
        """Unwrap this frame's SCRs, feed the clock its lowest-latency sample.
        Returns (raw, unwrapped) STC of the frame's first SCR."""
        first = None
        best = None
        for e in entries:
            if e.scr_stc is None:
                continue
            stc_u = self._stc(e.scr_stc)
            if first is None:
                first = (e.scr_stc, stc_u)
            host_s = e.host_ns / 1e9
            # Within one frame (~1 ms of packets) the nominal rate is plenty to
            # rank samples by latency.
            lateness = host_s - stc_u / self.clock.nominal_hz
            if best is None or lateness < best[0]:
                best = (lateness, stc_u, host_s)
        if best is not None:
            self.clock.add(best[1], best[2])
        return first

    def lookup(self, buf_ts_s: float, timeout_s: float = 0.05) -> Optional[float]:
        """Camera-clock time (host monotonic s) of the video frame whose V4L2
        timestamp is buf_ts_s, or None (no metadata for it / clock not fitted)."""
        key = int(round(buf_ts_s * 1e6))
        with self._cond:
            if not self._cond.wait_for(lambda: key in self._pending or not self._running,
                                       timeout=timeout_s):
                return None
            hit = self._pending.pop(key, None)
        if hit is None or not self.clock.ready:
            return None
        return self.clock.to_host(hit[0])
