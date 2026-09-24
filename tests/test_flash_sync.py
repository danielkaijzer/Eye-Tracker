"""Check flash_sync_test's analysis recovers known camera offsets.

Simulates the stimulus and both cameras: each camera frame's brightness is
the fraction of its exposure window during which the square was lit, and the
camera stamps the frame `offset` after mid-exposure. The measured
eye - scene difference must equal the difference of those offsets (display
latency cancels), and per-camera latency must equal display latency + offset.

Runs standalone: python -m tests.test_flash_sync (from the repo root).
"""
import os
import tempfile

import numpy as np

from scripts.extras.flash_sync_test import analyze, _brightness, _crossings

DISPLAY_LATENCY = 0.020


def _simulate(rng, period, exposure, offset, events, on_s, t_end, size=(12, 16)):
    lit_spans = [(t + DISPLAY_LATENCY, t + DISPLAY_LATENCY + on_s)
                 for t, s in events if s == 1]
    stamps = np.arange(0.0, t_end, period) + rng.uniform(0, period)
    ts, frames = [], []
    for s in stamps:
        mid = s - offset
        a, b = mid - exposure / 2, mid + exposure / 2
        lit = sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in lit_spans) / exposure
        img = np.full(size, 20.0)
        img[4:8, 5:11] += 200 * lit                     # the square
        ts.append(s + rng.normal(0, 2e-5))               # ~0.02 ms stamp jitter
        frames.append(img + rng.normal(0, 1.5, size))
    return np.array(ts), np.clip(frames, 0, 255).astype(np.uint8)


def _events(rng, n=30, on_s=0.3):
    t, ev = 1.0, []
    for _ in range(n):
        ev += [(t, 1), (t + on_s, 0)]
        t += on_s + rng.uniform(0.4, 0.8)
    return ev, t + 1.0


def test_recovers_offsets():
    rng = np.random.default_rng(0)
    on_s = 0.3
    events, t_end = _events(rng, on_s=on_s)
    eye_off, scene_off = 0.004, -0.012
    eye_ts, eye_fr = _simulate(rng, 0.00991, 0.008, eye_off, events, on_s, t_end)
    sc_ts, sc_fr = _simulate(rng, 1 / 30, 0.0332, scene_off, events, on_s, t_end)
    for ts, fr, off in ((eye_ts, eye_fr, eye_off), (sc_ts, sc_fr, scene_off)):
        b, contrast = _brightness(fr.astype(np.float32), ts, events, on_s)
        assert contrast > 100
        rise, fall = _crossings(ts, b, events, on_s)
        assert np.isfinite(rise).all() and np.isfinite(fall).all()
        np.testing.assert_allclose(np.median(rise), DISPLAY_LATENCY + off, atol=1e-3)
        np.testing.assert_allclose(np.median(fall), DISPLAY_LATENCY + off, atol=1e-3)
    with tempfile.TemporaryDirectory() as tmp:          # end-to-end: save + analyze
        path = os.path.join(tmp, "flash.npz")
        np.savez_compressed(path, events=np.array(events), on_ms=300,
                            eye_clock="synthetic", scene_clock="synthetic",
                            eye_exposure=80, scene_exposure=332,
                            eye_ts=eye_ts, eye_frames=eye_fr, scene_ts=sc_ts, scene_frames=sc_fr)
        analyze(path)


if __name__ == "__main__":
    test_recovers_offsets()
    print("ok")
