# v6 (uc844_mount_v6_arm.scad) — status & next steps

3-part eye-cam mount: **frame** + **arm** + **reused white clips**. Keeps v5's aim
pose; splits into separate prints; clips go to the sides (off the lens axis).

## Done / working
- Frame at v5 pose (reuses `frame_body()`): `echo` reports **lens tip 30.0 mm from
  pupil, axis 38° vs 38.1° needed** — aimed at the pupil.
- Eye / rod / gaze-ray mocks kept + a **70° FOV cone** mock; side view confirms the
  cone (camera view) clears the arm, clips and rod (all sit below the upward view).
- Arm reworked to attach at the frame's **bottom edge** (two lugs) and run up to the
  clips — this keeps it **below the FOV** and stops it crossing the board/PCB (the
  earlier top/back attach routed a strut through the PCB window).
- Arm is one printable piece (two struts + bottom cross-brace); clip-ears carry the
  Ø3.2 clip holes at the ~18.7 mm clip pitch.
- `show` = `assembly | exploded | frame | arm`.

## Update — joint + printability pass
- **Frame + arm both manifold** (CGAL Simple: yes). Aim unchanged: tip→pupil 30.0 mm,
  38° vs 38.1° needed.
- **Arm remodeled as a single FLAT PLATE** (was round struts) lying in the tilted
  plane through the clip-ears + frame pads → lays flat on the bed, prints without
  support. Hangs below the lens/FOV.
- **Frame prints flat** pocket-up; the two bottom lugs are in-plane (print flat too).
- **Bolt joint**: each frame lug and its arm pad share a coaxial hole along the frame
  normal → M2.5 bolt + nut laps them (thin lugs, no insert needed). Verified coaxial.
- `show="print"` lays both parts out (arm preview rotation still cosmetically off).

## Next steps (in order)
0. Cosmetic: fix `show="print"` arm rotation to sit truly flat (part already is flat).
1. **Re-render** side/front/iso after the bottom-attach rework; confirm no strut↔board
   collision and FOV still clear. Run CGAL manifold check on `frame` and `arm`.
2. **Clip interface (critical):** the clip-ear is currently a placeholder tab. Extract
   the exact ear geometry the existing white clips bolt to from
   `3d-files/bottom mount/v1 various internal eye cams.step` (Ø3.2 holes 18.7 mm
   apart, ~45° face, ear thickness) and match it; import a clip STL into the assembly
   render to visually confirm they mate. Fallback: design a fresh clip for the
   2.75×5 mm rod.
3. **Arm rigidity/joint:** firm up the frame-lug↔arm joint (single-lap Y-screw into an
   insert in the frame lug). Consider gusseting the struts. Decide adjustable
   (short arc slot at the lug for a few °) vs fixed (emit 30/40/50° variants like the
   originals). User: stability first, adjustability nice-to-have.
4. **USB clearance:** check the bottom lugs/arm don't block the PCB back USB connector
   (bottom-left) or its cable exit.
5. **Export STLs** (`frame`, `arm`), then physical test-fit: bolt arm to the existing
   clips on the rod, aim on the live feed.

## Key params (uc844_mount_v6_arm.scad)
`tilt_angle=38`, `px=-4.8`, `pz=-19.6` (pose → 30 mm standoff); `eye_depth=19`,
`eye_rise=27`; `clip_pitch=18.7`; `ear_pitch=18`; `lug_h=7`. Pose math + `echo` at
top of file.
