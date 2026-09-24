"""DeviceClock / UvcTimestamper frame handling on synthetic UVC data.

Simulates a device clock with a known rate error and offset that wraps its
32-bit counter mid-run; SCR samples reach the host with one-sided, random USB
processing latency. The fitted STC -> host map must recover true host times of
frame PTS to well under 0.1 ms, across the wrap, and recover after a device
clock jump.

Runs standalone: python -m tests.test_uvc_clock (from the repo root).
"""
import numpy as np

from scripts.eyetracker.cameras.uvc_clock import DeviceClock, UvcTimestamper
from scripts.eyetracker.cameras.uvc_meta import HDR_PTS, HDR_SCR, MetaEntry

RATE = 15e6 * (1 + 95e-6)       # eye cam: +95 ppm
PERIOD = 0.00991
MIN_LATENCY = 0.0002            # floor of the simulated host processing delay


def _frames(rng, t0, n, stc_at_t0):
    """Per frame: (true host time of PTS, [MetaEntry...]) with 46 SCRs spread
    over the first ~2 ms of the frame's packets."""
    out = []
    for i in range(n):
        t_pts = t0 + i * PERIOD
        pts = int(stc_at_t0 + (t_pts - t0) * RATE) % 2 ** 32
        entries = []
        for j in range(46):
            t_pkt = t_pts + 0.0001 + j * 4e-5            # device sends packet
            stc = int(stc_at_t0 + (t_pkt - t0) * RATE) % 2 ** 32
            lat = MIN_LATENCY + rng.exponential(0.0015)  # host sees it later
            entries.append(MetaEntry(host_ns=int((t_pkt + lat) * 1e9), host_sof=0,
                                     flags=HDR_PTS | HDR_SCR, pts=pts,
                                     scr_stc=stc, scr_sof=0))
        out.append((t_pts, entries))
    return out


def _feed(ts, frames):
    """Push frames through UvcTimestamper's per-frame path; return
    (true time, estimated time) for frames once the clock is ready. The
    envelope fit can only see device time + the *minimum* host latency, so
    that floor is added to the truth."""
    got = []
    for t_true, entries in frames:
        stc_raw, stc_u = ts._add_frame(entries)
        pts_u = stc_u - ((stc_raw - entries[0].pts) % 2 ** 32)
        if ts.clock.ready:
            got.append((t_true + MIN_LATENCY, ts.clock.to_host(pts_u)))
    return np.array(got)


def test_recovers_pts_host_time_across_wrap():
    rng = np.random.default_rng(0)
    ts = UvcTimestamper("/dev/null")
    # Start 5 s before the 32-bit STC wraps (2^32 / 15 MHz ~ 286 s period).
    frames = _frames(rng, t0=1000.0, n=2000, stc_at_t0=2 ** 32 - int(5 * RATE))
    got = _feed(ts, frames)
    err_ms = (got[:, 1] - got[:, 0]) * 1e3
    assert len(got) > 1500
    assert np.abs(err_ms).max() < 0.1, np.abs(err_ms).max()
    assert abs(ts.clock.rate_hz / RATE - 1) < 5e-6


def test_recovers_after_clock_jump():
    rng = np.random.default_rng(1)
    ts = UvcTimestamper("/dev/null")
    before = _frames(rng, t0=50.0, n=400, stc_at_t0=123_456_789)
    after = _frames(rng, t0=54.0, n=600, stc_at_t0=7_000_000)   # device reset
    _feed(ts, before)
    got = _feed(ts, after)
    tail = got[-200:]
    assert np.abs(tail[:, 1] - tail[:, 0]).max() * 1e3 < 0.1


def test_nonsense_rate_rejected():
    clock = DeviceClock()
    for i in range(200):                 # claims 30 MHz: >1% off nominal
        clock.add(int(i * 0.01 * 30e6), 10.0 + i * 0.01)
    assert not clock.ready


def test_stalled_metadata_does_not_block():
    import threading
    import time
    ts = UvcTimestamper("/dev/null")
    ts._running = True                       # pretend the reader thread is up
    ts._thread = threading.Thread(target=time.sleep, args=(5,), daemon=True)
    ts._thread.start()
    for _ in range(5):                       # first misses wait the timeout...
        assert ts.lookup(1.0, timeout_s=0.02) is None
    t0 = time.monotonic()
    for _ in range(20):                      # ...then stop waiting
        assert ts.lookup(1.0, timeout_s=0.02) is None
    assert time.monotonic() - t0 < 0.05


if __name__ == "__main__":
    test_recovers_pts_host_time_across_wrap()
    test_recovers_after_clock_jump()
    test_nonsense_rate_rejected()
    test_stalled_metadata_does_not_block()
    print("ok")
