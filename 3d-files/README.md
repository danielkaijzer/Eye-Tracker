# 3D-printed camera mounts

Parametric OpenSCAD designs for the eye and scene camera mounts. Open a `.scad`
file, set `show = ...` near the top, then press F5 to preview or F6 to render and
export an STL. Hardware dimensions are in [MEASUREMENTS.md](MEASUREMENTS.md).

## Current designs (printed and in use)

| Camera | File | Notes |
|---|---|---|
| Eye (UC-844 Rev.B, IR) | [`eye_cam/uc844_mount_v9_DK.scad`](eye_cam/uc844_mount_v9_DK.scad) | Frame + two arms; clamps to the rod with the stock white clips. Needs `clip_ref.stl` in the same folder. |
| Scene (HUAQUE HQ-L103) | [`scene_cam/scene_cam_top_mount_v2.scad`](scene_cam/scene_cam_top_mount_v2.scad) | Backing plate + parametric `rod_clip()`, so no STL import is needed. |

When a new version replaces one of these, move the old file into `archive/`.

## Calibration jig

The eye↔scene extrinsics jig isn't 3D printed. It's three stainless steel sheets held at
roughly 90° with L brackets, with printed ChArUco boards pasted on
(`scripts/extras/generate_charuco_board.py --board print`) so the eye cam and the scene cam
each see a board on a different sheet. The angles don't need to be exact: the extrinsics
solver only needs the jig to be rigid (`scripts/extras/calibrate_extrinsics.py`; its
docstring has the capture workflow). An earlier 3D-printed design was never built; it's in
git history.

## Archive

`archive/` keeps earlier iterations for reference. They aren't maintained.

- `archive/eye_cam/`: `uc844_mount_v1`–`v8`, `glasses_rod_adapter.scad` (the peg
  adapter that v3's integrated clamp replaced), the v6 design notes, and the original Fusion STEP bottom mount.
  v7 and v8 import `../../eye_cam/clip_ref.stl`.
- `archive/scene_cam/`: `scene_cam_top_mount_v1.scad` and the original Fusion
  STEP top mounts (`top_mount_step/`, v1–v3.3).

## Design notes

- 60° seems to be the right angle for the internal eye cam.
- The backs of the mounts should be perforated for airflow. Consider a way to
  add thermal paste to help cool the PCBs.
