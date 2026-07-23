# Hardware Measurements & Constraints

Reference for the 3D-printed mount / CAD design work. Update as new measurements
arrive. Covers two separate projects: the **eye camera** mount and the **scene
camera** top mount.

---

## Eye camera — PCB (UC-844 Rev.B)

- **Board:** 38.0 × 38.0 mm, thickness 1.6 mm. Single-sided (components on the front).
- **Mounting holes:** 4-corner pattern, compatible with **both 34 × 34 mm and
  28 × 28 mm** pitch (two hole sets). Corner hole radii ≈ R1.30 / R1.40 mm
  (≈ Ø2.6 / Ø2.8). Mount design (v3/v4) uses Ø2.85 clearance at 34 mm pitch with
  M2.5 heat-set inserts.
- **Reference outlines on the datasheet:** 38 / 34 / 28 mm nested squares.
- **Imager / sensor:** centered on the board (19 mm from edges to center).
- **Lens:** centered on the board.
- **USB connector** (4-pin: VCC, DM, DP, GND): on the **back** of the PCB,
  **bottom-left**. Approx ~7.56 mm wide × ~3.81 mm tall, ~3.47 mm up from the bottom
  edge, ~8.24 mm in from the left edge.
  → **Keep the PCB back open** (connector access + heat dissipation).

## Eye camera — Lens (M12)

- **Mount:** M12 × P0.5 mm (threaded → refocusable in the holder).
- **EFL:** 2.8 mm. **FOV:** 70° (H). **Distortion:** < 1%.
- **Optical format:** 1/4".
- **Mechanical:** overall length 15.86 mm; front Ø14 ± 0.05; body Ø12; segment lengths
  4.3 / 8.9 / 13.2 mm; aperture ≈ Ø9.
- **Video:** 720p.
- **Focus:** datasheet "Macro Focus Range 100 cm" is a **typo** — bench-tested and it
  focuses **sharply down to ~1 cm**. No lens swap needed; focus does **not** constrain
  how close the camera can sit. (Working distance is limited by FOV/framing and face
  clearance, not focus.)

## Eye camera — aim geometry

- **Target lens-tip-to-eye distance:** ~44–51 mm (1.75–2″). Can go closer if desired
  (focus allows down to ~1 cm); limited by FOV/framing and face clearance.
- **Eyeball depth:** ~19 mm (0.75″) behind the glasses lens plane (the vertical plane
  through the top and bottom rims).
- **Preferred tilt:** ~60° for the internal eye cam (from NOTES.md).
- **Placement intent:** lens just **below** the eye (out of the sightline), angled
  **up** at the pupil, as close as practical.

## Eye camera — glasses rod (clamp target)

- **Cross-section:** ~2.5–3 mm tall (vertical, Z), ~5 mm deep (fore-aft, X), rounded
  (not a true cylinder).
- **Usable horizontal length for the clamp:** ~1 inch (~25 mm).
  → mount must be **compact and centered**; no side-hanging hardware / across-face
  overhang.

---

## Scene camera — PCB (HUAQUE HQ-L103) — separate "top mount" project

- **Board:** 62 × 9 × 1.0 mm (bare PCB). Lens Ø ~5.68 mm, protrudes ~4.68 mm off the
  front face.
- **Mounting holes:** Ø2.2 drilled (Ø3.2 pad).
  - Two at the **camera end**: 4.0 mm apart (± 2.0 mm off the width centerline),
    2.0 mm from the end edge.
  - Third near the **connector end**: Ø2.2, ~10.2 mm from the connector-end edge,
    ~2.0 mm off the centerline.
- **Feature positions (from the mounting-hole end):** camera 32.5 mm, LED 12.5 mm.
- **Rod clips:** reuses the existing white rod clips — Ø3.2 clip holes at **64 mm
  pitch** (taken from the current bar's STEP file).
- File: `3d-files/scene_cam_top_mount_v1.scad`.

---

## Mount design constraints (derived, eye camera)

- Reuse the v3/v4 **square frame + 4-corner PCB screw mount unchanged**.
- Keep the **PCB back open** (USB connector + heat).
- **Centered / compact** within the ~1 inch of usable rod; no across-face overhang.
- **Tilt adjustable** (friction pivot, axis parallel to the rod) + **standoff
  adjustable** (slot); final aim done on the live camera feed.
- Mount is **lens-agnostic** (holds the PCB; the lens protrudes through the open
  window), so any M12 lens change needs no design change.
