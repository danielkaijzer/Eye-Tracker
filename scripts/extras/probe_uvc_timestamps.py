"""Check the quality of the cameras' frame timestamps (Linux).

Streams the eye + scene cams concurrently through the App's own camera class
(OpenCVCamera, same settings) and reports per camera, from the per-frame
capture timestamps it records (CameraSource.last_timestamp):

- delivered frame rate and frame period (median)
- period jitter (std): ~0.01-0.03 ms when uvcvideo hwtimestamps=1 (camera
  clock via PTS/SCR), ~2 ms when off (host arrival times)
- gaps: frames missing between consecutive timestamps (drops, or a consumer
  that fell behind the newest-frame handoff)
- latency: time.monotonic() when the App received the frame minus its capture
  timestamp; includes transfer + MJPG decode. Note each camera's timestamp has
  its own unmeasured fixed offset from true exposure, so this isn't an
  absolute exposure-to-app latency.

Findings on the Jetson rig (2026-09-23), hwtimestamps=1: scene 30.000 fps,
33.333 ms +/- 0.01 ms; eye 100.9 fps, 9.909 ms +/- 0.01-0.03 ms; no gaps.
Both cams send valid PTS/SCR on every frame (uvcvideo debugfs stats). The
uvcvideo metadata node (/dev/videoN+1, 'UVCH') delivers no buffers on this
5.15-tegra kernel, so raw PTS/SCR aren't readable from userspace; the kernel's
hwtimestamps conversion is the path that works.

Run (app closed):
    python -m scripts.extras.probe_uvc_timestamps [--seconds 6]
"""
import argparse
import threading
import time

import numpy as np

from scripts.eyetracker.__main__ import _eye_cam_settings, _scene_cam_settings
from scripts.eyetracker.cameras.opencv_source import OpenCVCamera
from scripts.eyetracker.cameras.v4l2 import index_for_usb_id, uvc_hw_timestamps_enabled
from scripts.eyetracker.config import EYE_UVC_ID, SCENE_UVC_ID

_WARMUP_FRAMES = 30


def _capture(name, usb_id, settings, seconds, out):
    idx = index_for_usb_id(usb_id)
    if idx is None:
        out[name] = f"camera {usb_id} not found"
        return
    cam = OpenCVCamera(idx, settings)
    if not cam.open():
        out[name] = f"/dev/video{idx} failed to open"
        return
    rows = []
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cam.read() is not None and cam.last_timestamp is not None:
                rows.append((cam.last_timestamp, time.monotonic()))
    finally:
        cam.release()
    out[name] = (f"/dev/video{idx}", np.array(rows[_WARMUP_FRAMES:]))


def _report(name, dev, rows):
    if len(rows) < 10:
        print(f"{name}: too few frames ({len(rows)})")
        return
    ts, recv = rows[:, 0], rows[:, 1]
    d = np.diff(ts) * 1e3
    period = np.median(d)
    gaps = int(np.sum(np.round(d / period) - 1))
    lat = (recv - ts) * 1e3
    fps = (len(ts) - 1) / (ts[-1] - ts[0])
    print(f"{name} ({dev}): {fps:.2f} fps, period {period:.3f} ms "
          f"(std {np.std(d[np.round(d / period) == 1]):.3f} ms), gaps {gaps}, "
          f"latency median {np.median(lat):.1f} ms (std {np.std(lat):.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()
    hw = uvc_hw_timestamps_enabled()
    print(f"uvcvideo hwtimestamps: {'on' if hw else 'OFF (host arrival times)' if hw is not None else 'uvcvideo not loaded'}")
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
            print(f"{name}: {res}")


if __name__ == "__main__":
    main()
