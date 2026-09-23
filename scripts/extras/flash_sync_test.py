"""Flash test: measure each camera's frame-timestamp offset against real light.

Why: frame timestamps (CameraSource.last_timestamp, uvcvideo hwtimestamps=1)
are on the host monotonic clock with ~0.01-0.04 ms jitter, but each camera
stamps a different, undocumented point of its capture (the eye cam's stamp
even lands after its first USB packet, so it isn't exposure start). Pairing
eye/scene frames by equal timestamps is therefore off by a fixed per-camera
amount. This test measures it.

How: a fullscreen window flashes a white square on black ~30 times at random
intervals and logs each flip on the host clock. Both cameras look at the
square. For each flash edge and camera, the frame brightness (over the pixels
that respond to the flash) is normalised between the dark and lit levels and
the 50% crossing is interpolated between frames: that's when the camera's
timestamps say the light changed.

Results, per camera and edge (rising = on, falling = off):
- latency = crossing - flip time: display latency (compositor + scanout +
  LCD response) + that camera's stamp offset. This is the number to use when
  relating camera frames to stimulus times logged the same way (e.g.
  calibration dot onsets), since those go through the same display path.
- eye - scene: display latency cancels, leaving the cross-camera offset.
  Subtract it from eye_frame_ts (or add to scene_frame_ts) to pair frames by
  the moment light hit the sensors.

The eye cam's exposure is pinned (manual, --eye-exposure) during the test so
the result isn't smeared by auto exposure, and restored afterwards. Keep the
scene cam's exposure as the App uses it.

Setup: take the headset off and point BOTH cameras at the centre of the
monitor so the square is well inside each view (the eye cam will be out of
focus; that's fine). Dim room lights if you can. App closed.

Run:      python -m scripts.extras.flash_sync_test
Reanalyse: python -m scripts.extras.flash_sync_test --analyze data/sync_tests/flash_<ts>.npz
Keys:     SPACE start (after aiming), Esc abort.
"""
import argparse
import datetime
import os
import random
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk

import cv2
import numpy as np

from scripts.eyetracker.__main__ import _eye_cam_settings, _scene_cam_settings
from scripts.eyetracker.calibration.paths import dataset_root
from scripts.eyetracker.cameras.opencv_source import OpenCVCamera
from scripts.eyetracker.cameras.v4l2 import (
    find_v4l2_ctl,
    index_for_usb_id,
    uvc_hw_timestamps_enabled,
)
from scripts.eyetracker.config import EYE_UVC_ID, SCENE_UVC_ID

# Largest |camera offset| the analysis windows tolerate. Flash timing below
# is derived from it so baseline/plateau windows never straddle an edge.
_MAX_OFFSET_S = 0.10
_SMALL = {"eye": (160, 120), "scene": (192, 108)}   # stored frame size


# ---- capture -----------------------------------------------------------------

class _Recorder:
    """Pulls frames from one camera, keeps (timestamp, small gray frame)."""

    def __init__(self, name, cam):
        self.name, self.cam = name, cam
        self.ts, self.frames = [], []
        self.latest = None
        self.recording = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        size = _SMALL[self.name]
        while not self._stop:
            frame = self.cam.read()
            if frame is None or self.cam.last_timestamp is None:
                continue
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size,
                               interpolation=cv2.INTER_AREA)
            self.latest = small
            if self.recording:
                self.ts.append(self.cam.last_timestamp)
                self.frames.append(small)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)


def _v4l2(dev, *args):
    ctl = find_v4l2_ctl()
    if not ctl:
        return ""
    return subprocess.run([ctl, "-d", dev, *args], capture_output=True,
                          text=True, timeout=3).stdout


def _get_ctrl(dev, name):
    out = _v4l2(dev, "-C", name)
    try:
        return int(out.split(":")[1].split()[0])
    except (IndexError, ValueError):
        return None


# ---- stimulus ----------------------------------------------------------------

class _FlashWindow:
    def __init__(self, recorders, args):
        self.recorders = recorders
        self.args = args
        self.events = []          # (host monotonic s, 1 = on / 0 = off)
        self.aborted = False
        self.root = tk.Tk()
        self.root.configure(bg="black", cursor="none")
        self.root.attributes("-fullscreen", True)
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="black",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        p = args.patch // 2
        self.patch = self.canvas.create_rectangle(w // 2 - p, h // 2 - p,
                                                  w // 2 + p, h // 2 + p,
                                                  fill="black", width=0)
        self.text = self.canvas.create_text(
            w // 2, 60, fill="#888", font=("Helvetica", 18),
            text="Aim BOTH cameras at the blinking square (previews below).  "
                 "SPACE = start,  Esc = abort")
        self._w, self._h = w, h
        self._previews = {}
        self._photos = []
        self._aiming = True
        self._blink = False
        self.root.bind("<KeyPress-space>", lambda _e: self._start())
        self.root.bind("<KeyPress-Escape>", lambda _e: self._abort())
        self.root.focus_force()

    # aim phase: slow blink + camera previews
    def _aim_tick(self):
        if not self._aiming:
            return
        self._blink = not self._blink
        self.canvas.itemconfig(self.patch, fill="white" if self._blink else "black")
        self.root.after(500, self._aim_tick)

    def _preview_tick(self):
        if not self._aiming:
            return
        self._photos = []
        for i, r in enumerate(self.recorders):
            if r.latest is None:
                continue
            img = cv2.resize(r.latest, (320, int(320 * r.latest.shape[0] / r.latest.shape[1])))
            rgb = np.dstack([img] * 3)
            hdr = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255\n".encode()
            photo = tk.PhotoImage(master=self.root, data=hdr + rgb.tobytes(), format="PPM")
            self._photos.append(photo)
            x = 40 + 160 if i == 0 else self._w - 40 - 160
            if r.name not in self._previews:
                self._previews[r.name] = (
                    self.canvas.create_image(x, self._h - 40, anchor="s", image=photo),
                    self.canvas.create_text(x, self._h - 40 - rgb.shape[0] - 14,
                                            fill="#888", text=r.name, font=("Helvetica", 14)))
            else:
                self.canvas.itemconfig(self._previews[r.name][0], image=photo)
        self.root.after(100, self._preview_tick)

    def _start(self):
        if not self._aiming:
            return
        self._aiming = False
        for img, label in self._previews.values():
            self.canvas.delete(img)
            self.canvas.delete(label)
        self.canvas.delete(self.text)
        self.canvas.itemconfig(self.patch, fill="black")
        self.root.update_idletasks()
        for r in self.recorders:
            r.recording = True
        self._remaining = self.args.flashes
        self.root.after(1000, self._flash_on)

    def _set(self, on):
        self.canvas.itemconfig(self.patch, fill="white" if on else "black")
        self.root.update_idletasks()      # draw + flush now, then stamp
        self.events.append((time.monotonic(), 1 if on else 0))

    def _flash_on(self):
        self._set(True)
        self.root.after(self.args.on_ms, self._flash_off)

    def _flash_off(self):
        self._set(False)
        self._remaining -= 1
        if self._remaining <= 0:
            self.root.after(1000, self.root.destroy)
            return
        self.root.after(random.randint(self.args.off_min_ms, self.args.off_max_ms),
                        self._flash_on)

    def _abort(self):
        self.aborted = True
        self.root.destroy()

    def run(self):
        self._aim_tick()
        self._preview_tick()
        if self.args.autostart is not None:
            self.root.after(int(self.args.autostart * 1000), self._start)
        self.root.mainloop()


# ---- analysis ----------------------------------------------------------------

def _crossings(ts, b, events, on_s):
    """Per edge: interpolated time the normalised brightness crosses 0.5, or
    NaN if the edge can't be resolved. Returns (rising, falling) arrays of
    crossing - flip_time, index-matched to the on/off events."""
    m = _MAX_OFFSET_S
    rising, falling = [], []
    for t, state in events:
        if state == 1:
            base = b[(ts > t - 3 * m) & (ts < t - m)]
            plat = b[(ts > t + m) & (ts < t + on_s - m)]
        else:
            base = b[(ts > t - on_s + m) & (ts < t - m)]
            plat = b[(ts > t + m) & (ts < t + 3 * m)]
        out = rising if state == 1 else falling
        if len(base) < 1 or len(plat) < 1:
            out.append(np.nan)
            continue
        lo, hi = np.median(base), np.median(plat)
        if abs(hi - lo) < 5:                 # no visible change for this edge
            out.append(np.nan)
            continue
        win = np.where((ts > t - m) & (ts < t + m))[0]
        n = (b[win] - lo) / (hi - lo)        # 0 before the edge, 1 after
        k = next((i for i in range(1, len(n)) if n[i - 1] < 0.5 <= n[i]), None)
        if k is None:
            out.append(np.nan)
            continue
        t0, t1 = ts[win[k - 1]], ts[win[k]]
        tc = t0 + (0.5 - n[k - 1]) / (n[k] - n[k - 1]) * (t1 - t0)
        out.append(tc - t)
    return np.array(rising), np.array(falling)


def _brightness(frames, ts, events, on_s):
    """Mean over the pixels that respond to the flash (lit - dark >= half of
    the strongest response). Returns (brightness per frame, contrast)."""
    m = _MAX_OFFSET_S
    lit = np.zeros(len(ts), bool)
    dark = np.zeros(len(ts), bool)
    for t, state in events:
        if state == 1:
            lit |= (ts > t + m) & (ts < t + on_s - m)
            dark |= (ts > t - 3 * m) & (ts < t - m)
    if not lit.any() or not dark.any():
        return None, 0.0
    diff = frames[lit].mean(0) - frames[dark].mean(0)
    contrast = float(diff.max())
    mask = diff >= 0.5 * contrast
    return frames[:, mask].mean(1), contrast


def _stats(x):
    x = x[np.isfinite(x)] * 1e3
    if len(x) == 0:
        return "no resolved edges"
    return (f"median {np.median(x):7.2f} ms  (std {np.std(x):5.2f}, "
            f"range {x.min():.2f}..{x.max():.2f}, n={len(x)})")


def analyze(path):
    d = np.load(path, allow_pickle=True)
    events = [tuple(e) for e in d["events"]]
    on_s = float(d["on_ms"]) / 1e3
    print(f"\n{os.path.basename(path)}: {sum(1 for _, s in events if s)} flashes, "
          f"hwtimestamps={'on' if d['hwtimestamps'] else 'OFF'}, "
          f"eye exposure {d['eye_exposure']}, scene exposure {d['scene_exposure']} "
          f"(units of 100 us)")
    res = {}
    for name in ("eye", "scene"):
        ts, frames = d[f"{name}_ts"], d[f"{name}_frames"].astype(np.float32)
        b, contrast = _brightness(frames, ts, events, on_s)
        period = np.median(np.diff(ts)) * 1e3 if len(ts) > 1 else float("nan")
        print(f"\n{name}: {len(ts)} frames, period {period:.3f} ms, flash contrast "
              f"{contrast:.0f} gray levels")
        if b is None or contrast < 10:
            print("  camera doesn't see the flash clearly; re-aim / raise exposure")
            continue
        rise, fall = _crossings(ts, b, events, on_s)
        res[name] = (rise, fall)
        print(f"  on  edge  latency vs flip: {_stats(rise)}")
        print(f"  off edge  latency vs flip: {_stats(fall)}")
    if len(res) == 2:
        dr = res["eye"][0] - res["scene"][0]
        df = res["eye"][1] - res["scene"][1]
        both = np.concatenate([dr, df])
        print("\neye - scene (display latency cancels):")
        print(f"  on  edges: {_stats(dr)}")
        print(f"  off edges: {_stats(df)}")
        print(f"  all edges: {_stats(both)}")
        off = np.nanmedian(both) * 1e3
        print(f"\n=> eye stamps run {abs(off):.2f} ms {'LATER' if off > 0 else 'EARLIER'} "
              f"than scene stamps for the same light. To pair frames by light, "
              f"use eye_frame_ts {'-' if off > 0 else '+'} {abs(off):.2f} ms.")


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analyze", metavar="NPZ", help="reanalyse a saved run and exit")
    ap.add_argument("--flashes", type=int, default=30)
    ap.add_argument("--on-ms", type=int, default=300)
    ap.add_argument("--off-min-ms", type=int, default=400)
    ap.add_argument("--off-max-ms", type=int, default=800)
    ap.add_argument("--patch", type=int, default=600, help="square side, screen px")
    ap.add_argument("--eye-exposure", type=int, default=80,
                    help="eye cam manual exposure during the test (100 us units)")
    ap.add_argument("--autostart", type=float, metavar="S",
                    help="start flashing after S seconds instead of waiting for SPACE")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
        return

    hw = uvc_hw_timestamps_enabled()
    if not hw:
        print("WARNING: uvcvideo hwtimestamps is off; results will reflect host "
              "arrival times, not camera clocks.")
    eye_idx, scene_idx = index_for_usb_id(EYE_UVC_ID), index_for_usb_id(SCENE_UVC_ID)
    if eye_idx is None or scene_idx is None:
        print(f"cameras not found (eye {eye_idx}, scene {scene_idx})")
        return
    eye_dev, scene_dev = f"/dev/video{eye_idx}", f"/dev/video{scene_idx}"
    eye_cam = OpenCVCamera(eye_idx, _eye_cam_settings())
    scene_cam = OpenCVCamera(scene_idx, _scene_cam_settings())
    if not (eye_cam.open() and scene_cam.open()):
        print("failed to open cameras")
        return
    # A kill (Ctrl-C in another terminal, timeout) must still run the finally
    # below, or the eye cam is left in manual exposure.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    eye_ae = _get_ctrl(eye_dev, "auto_exposure")
    _v4l2(eye_dev, "-c", f"auto_exposure=1,exposure_time_absolute={args.eye_exposure}")
    recorders = [_Recorder("eye", eye_cam), _Recorder("scene", scene_cam)]
    try:
        for r in recorders:
            r.start()
        win = _FlashWindow(recorders, args)
        win.run()
    finally:
        for r in recorders:
            r.stop()
        eye_exp = _get_ctrl(eye_dev, "exposure_time_absolute")
        scene_exp = _get_ctrl(scene_dev, "exposure_time_absolute")
        if eye_ae is not None:
            _v4l2(eye_dev, "-c", f"auto_exposure={eye_ae}")
        eye_cam.release()
        scene_cam.release()
    if win.aborted or not win.events:
        print("aborted; nothing saved")
        return

    out_dir = os.path.join(os.path.dirname(dataset_root()), "sync_tests")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"flash_{datetime.datetime.now():%Y%m%d_%H%M%S}.npz")
    np.savez_compressed(
        path, events=np.array(win.events), on_ms=args.on_ms,
        hwtimestamps=bool(hw), eye_exposure=eye_exp, scene_exposure=scene_exp,
        **{f"{r.name}_ts": np.array(r.ts) for r in recorders},
        **{f"{r.name}_frames": np.array(r.frames, dtype=np.uint8) for r in recorders})
    print(f"saved {path}")
    analyze(path)


if __name__ == "__main__":
    main()
