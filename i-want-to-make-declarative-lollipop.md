# Calibration Jig CAD — Prep Doc for the CAD Agent

> Hand this file to a fresh Claude Code session in the directory where you want
> the OpenSCAD work to live. That session does the CAD; this session was the
> scoping/planning step.

## 1. Mission

Design a **parametric OpenSCAD model** of a calibration jig for a head-mounted
eye tracker. The jig holds two flat ArUco-marker panels at a **mechanically-known
relative pose** so a downstream solver can compute the extrinsic transform
(rotation + translation) between the headset's two cameras.

Deliverables:

1. `calibration_jig.scad` — parametric model, renders cleanly in OpenSCAD.
2. `marker_positions.json` (or `.csv`) — 3D coordinates of every marker corner
   in the jig's coordinate frame. **This is the non-obvious must-have** — the
   solver consumes it later, and it must come from the CAD, not be re-measured.
3. A short README block in the `.scad` header explaining: how to render, how
   to export STL, what each parameter does, and where the origin sits.

Out of scope for this session: generating ArUco marker images, the calibration
solver, mounting the headset to the jig.

## 2. Why this jig exists

The headset has two cameras pointing in different directions — one inward
(eye), one outward (scene). They cannot both see the same calibration target.
So the user needs a rigid object with marker patterns on different faces, each
facing one camera, with a **precisely-known inter-panel transform**.

Paper boards taped to a wall would shift and tilt unpredictably; a 3D-printed
rigid jig encodes the inter-panel geometry mechanically, to the printer's
tolerance (~0.1–0.3 mm on FDM).

The solver chain is: camera₁ → panel₁ (via PnP on detected markers) → panel₂
(via the CAD-known rigid transform) → camera₂ (via PnP). The middle step is
what this jig provides.

## 3. Hardware constraints (measured by user)

Camera offsets on the headset:

- **Lateral**: < 2 inches (~50 mm)
- **Depth**: ~ 0.5 inches (~13 mm)
- **Vertical**: ~ 2 inches (~50 mm)

The cameras are diagonally offset in 3D. Reference photos in repo root:
`IMG_8034.jpg`, `IMG_8035.jpg`, `IMG_8036.jpg` (glasses with tape measure for
scale).

## 4. Geometry the user specified

- Two **flat marker panels**, separated by **6 inches (152.4 mm)** center-to-center.
- **"Step" shape** frame connecting them. Side view: `|__|` (two verticals
  joined by a horizontal base). Top view: a staircase offset, so the two panels
  are laterally offset, not collinear — mirroring the camera offset.
- **Whole-inch dimensions preferred** where they don't compromise function.
- Panels should sit a **comfortable working distance** from each camera (not
  tight to the headset) so markers are in focus and resolvable. Slack > snug.

## 5. Hard CAD requirements

1. **Parametric**: every dimension is a top-level OpenSCAD variable. The user
   will iterate on panel spacing, step offsets, panel size, etc.
2. **mm internally** (OpenSCAD + slicer convention). Inch values are fine in
   comments / parameter labels for readability.
3. **Marker faces are flat reference surfaces** for paper-printed ArUco
   stickers — NOT embossed plastic. FDM-embossed shapes have rounded edges and
   wreck ArUco detection.
4. **Sidecar marker-position file**: each marker corner in the jig frame, with
   a clearly-defined origin. JSON keyed by marker ID is fine. This must be
   generated from the same parameters as the .scad (don't hand-write it).
5. **Printability**:
   - Marker panels print flat-side-down to preserve surface flatness.
   - Base/frame has ribs or gussets to prevent warp.
   - No overhangs on the marker faces that would force support material.
   - Keep total footprint within a typical FDM build plate (confirm with user).

## 6. Open questions — ASK the user before writing any .scad

The user said "one thing at a time." Don't assume these; ask first.

1. **Panel orientation**: parallel panels (simplest), or each tilted to face
   its camera square-on (better marker detection, harder geometry)? Default
   recommendation: parallel, with the user verifying both cameras can resolve
   markers at the resulting oblique angle.
2. **Camera specs** — needed to size markers and set panel-to-camera distance:
   - Resolution of eye camera? Scene camera?
   - Approximate FoV of each?
   - Fixed focus? If so, at what distance? If variable, what's the
     near-focus limit?
3. **Marker layout per panel**: a ChArUco board (recommended — best pose
   accuracy from a single planar target), a small grid of standalone ArUco
   markers, or a single large marker?
4. **Mounting / use posture**: does the user wear the headset and look at the
   jig, or is the headset mounted and the jig held in front? Affects required
   working distance and panel size.
5. **Printer constraints**: build volume (max jig footprint), material
   (PLA/PETG — affects warp advice), nozzle size (affects minimum feature size).

## 7. Suggested starting frame (after the questions are answered)

```scad
// Origin: center of Panel A's marker face.
//   +X = lateral, toward Panel B
//   +Y = away from Panel A's marker face (toward Panel B in depth)
//   +Z = up

panel_separation_y    = 152.4;   // 6 in — primary inter-panel distance
panel_lateral_dx      = 50.8;    // 2 in — top-down step offset
panel_vertical_dz     = 50.8;    // 2 in — side-view step offset

panel_width           = 80;      // marker-face width  (tune to camera FoV)
panel_height          = 80;      // marker-face height
panel_thickness       = 4;

base_thickness        = 5;
frame_width           = 25;      // arm width of the step frame
rib_count             = 2;
```

The 2-inch lateral and vertical step values bracket the actual camera offsets
with slack; the CAD agent should expose them so the user can pull them in or
push them out.

## 8. Things the CAD agent should NOT do

- Don't generate ArUco marker images — user does that separately.
- Don't design the headset mount yet (open question #4).
- Don't embed markers in the print geometry — flat reference surfaces only.
- Don't add features beyond what's specified. No "while we're at it" extras.

## 9. Verification before declaring done

- Model renders without warnings in OpenSCAD (recommend the **development
  snapshot** build — the 2021 "stable" is stale).
- Sidecar marker-position file is generated and the coordinates are
  self-consistent (e.g., marker A1's center matches what the .scad places).
- A test STL export slices cleanly in the user's slicer (user verifies).
- Origin and axes are documented in the .scad header so the future solver
  knows what frame the marker coordinates are in.

## 10. Reference

- Photos: `IMG_8034.jpg`, `IMG_8035.jpg`, `IMG_8036.jpg` in repo root.
- OpenSCAD download: https://openscad.org/downloads.html — use the development
  snapshot, not the 2021.01 stable. Native Apple Silicon build available.
- ArUco / ChArUco background (OpenCV docs): user will pull this in when
  designing the marker images later.
