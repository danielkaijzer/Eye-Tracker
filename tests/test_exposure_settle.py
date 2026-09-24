"""settle_exposure on a fake camera with a realistic brightness response.

Response shape follows the eye cam aimed at the monitor (2026-09-23): median
grows with exposure but concavely, from a ~20 black level. Checks it reaches
the target in a few steps, stays inside the frame-period limit, and gives up
sensibly when the target is out of reach.

Runs standalone: python -m tests.test_exposure_settle (from the repo root).
"""
import time

import numpy as np

from scripts.eyetracker.cameras.base import CameraSource
from scripts.eyetracker.cameras.exposure_settle import settle_exposure


class _FakeCam(CameraSource):
    def __init__(self, gain=1.0, value=99, limits=(1, 99)):
        self.gain, self.value, self.limits = gain, value, limits
        self.sets = []

    def _median(self):
        return min(255.0, 20 + self.gain * 110 * np.log1p(self.value / 25.0))

    def open(self):
        return True

    def release(self):
        pass

    def read(self):
        self.last_timestamp = time.monotonic()
        return np.full((48, 64), self._median(), dtype=np.float32).astype(np.uint8)

    def exposure_value(self):
        return self.value

    def exposure_limits(self):
        return self.limits

    def set_exposure(self, value):
        self.value = int(max(self.limits[0], min(self.limits[1], value)))
        self.sets.append(self.value)
        return True


def test_reaches_target():
    for start in (1, 50, 99):
        cam = _FakeCam(value=start)
        res = settle_exposure(cam, target=96, apply_delay_s=0)
        assert res.converged, res
        assert abs(res.median - 96) <= 0.08 * 96
        assert res.iterations <= 6, res
        assert cam.value == res.value            # left at the settled value


def test_too_dark_stops_at_max():
    for start in (1, 50, 99):                   # can't reach 96 at any exposure
        cam = _FakeCam(gain=0.2, value=start)
        res = settle_exposure(cam, target=96, apply_delay_s=0)
        assert not res.converged and res.value == 99 and "dark" in res.note, res
        assert max(cam.sets) <= 99


def test_no_exposure_control():
    class _NoCtl(_FakeCam):
        def exposure_limits(self):
            return None
    assert settle_exposure(_NoCtl(), target=96, apply_delay_s=0) is None


if __name__ == "__main__":
    test_reaches_target()
    test_too_dark_stops_at_max()
    test_no_exposure_control()
    print("ok")
