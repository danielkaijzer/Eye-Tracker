"""Check the quality of the cameras' frame timestamps (Linux).

Streams the eye + scene cams concurrently through the App's own camera class
(OpenCVCamera, same settings) and reports per camera, from the per-frame
timestamps it records (CameraSource.last_timestamp / last_timestamp_source):

- timestamp source per frame: "uvc_pts" (camera PTS converted through the
  SCR clock fit, cameras/uvc_clock.py) vs "v4l2_buffer" (host arrival
  fallback), plus the fitted device clock rate and envelope residual
- delivered frame rate, frame period (median) and period jitter (std)
- gaps: frames missing between consecutive timestamps (drops, or a consumer
  that fell behind the newest-frame handoff)
- latency: time.monotonic() when the App received the frame minus its
  timestamp (transfer + MJPG decode + handoff), and whether it trends. Real
  timestamp drift shows as hundreds-thousands of ppm (the kernel conversion:
  ~2600); tens of ppm on short runs is beating between the frame rate and the
  read loop (use --seconds 30+ to average it out)

Rig findings (2026-09-23, uvcvideo nodrop=1 hwtimestamps=0, uvc_pts): see the
commit adding uvc_clock.py. With the kernel's own conversion (hwtimestamps=1)
the eye cam drifted ~2600 ppm and jumped by >100 ms; its bridge reports a
free-running SOF counter in SCR that the 5.15 kernel trusts.

Run (app closed):
    python -m scripts.extras.probe_uvc_timestamps [--seconds 20]
"""
import argparse
import threading
import time
from collections import Counter

import numpy as np

from scripts.eyetracker.__main__ import _eye_cam_settings, _scene_cam_settings
from scripts.eyetracker.cameras.opencv_source import OpenCVCamera
from scripts.eyetracker.cameras.v4l2 import index_for_usb_id, uvc_timestamp_setup_problem
from scripts.eyetracker.config import EYE_UVC_ID, SCENE_UVC_ID

_WARMUP_S = 2.0      # the camera-clock fit needs ~1 s of SCR samples


def _capture(name, usb_id, settings, seconds, out):
    idx = index_for_usb_id(usb_id)
    if idx is None:
        out[name] = f"camera {usb_id} not found"
        return
    cam = OpenCVCamera(idx, settings)
    if not cam.open():
        out[name] = f"/dev/video{idx} failed to open"
        return
    rows, sources = [], []
    try:
        start = time.monotonic()
        while time.monotonic() - start < seconds + _WARMUP_S:
            if cam.read() is None or cam.last_timestamp is None:
                continue
            if time.monotonic() - start >= _WARMUP_S:
                rows.append((cam.last_timestamp, time.monotonic()))
                sources.append(cam.last_timestamp_source)
        clock = cam.timestamp_clock_info()
    finally:
        cam.release()
    out[name] = (f"/dev/video{idx}", np.array(rows), sources, clock)


def _report(name, dev, rows, sources, clock):
    print(f"\n{name} ({dev}): timestamps {dict(Counter(sources))}")
    print(f"  {clock or 'no camera-clock fit'}")
    if len(rows) < 10:
        print(f"  too few frames ({len(rows)})")
        return
    ts, recv = rows[:, 0], rows[:, 1]
    d = np.diff(ts) * 1e3
    period = np.median(d)
    steps = np.round(d / period)
    gaps = int(np.sum(steps - 1))
    lat = (recv - ts) * 1e3
    trend = np.polyfit(recv - recv[0], lat, 1)[0]     # ms per s
    fps = (len(ts) - 1) / (ts[-1] - ts[0])
    print(f"  {fps:.3f} fps, period {period:.4f} ms (std {np.std(d[steps == 1]):.4f} ms), gaps {gaps}")
    print(f"  latency to app: median {np.median(lat):.2f} ms (std {np.std(lat):.2f}), "
          f"trend {trend * 1e3:+.0f} ppm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    problem = uvc_timestamp_setup_problem()
    print(f"uvcvideo setup: {'ok (nodrop=1, hwtimestamps=0)' if problem is None else problem}")
    out = {}
    threads = [threading.Thread(target=_capture, args=(n, u, s, args.seconds, out))
               for n, u, s in (("eye", EYE_UVC_ID, _eye_cam_settings()),
                               ("scene", SCENE_UVC_ID, _scene_cam_settings()))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for name in ("eye", "scene"):
        res = out.get(name)
        if isinstance(res, tuple):
            _report(name, *res)
        else:
            print(f"\n{name}: {res}")


if __name__ == "__main__":
    main()
