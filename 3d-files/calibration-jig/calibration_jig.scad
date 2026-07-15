// =============================================================================
// calibration_jig.scad — Parametric calibration jig for dual-camera eye tracker
// =============================================================================
//
// PURPOSE
//   Holds two flat ArUco-marker panels at a mechanically-known relative pose.
//   A downstream solver uses this fixed inter-panel transform to compute the
//   extrinsic between the headset's eye camera and scene camera (which cannot
//   both see the same target).
//
// GEOMETRY
//   U-shaped jig. Two parallel marker panels rise from a horizontal base.
//   Headset sits on the base between the panels. Eye cam sees Panel A's
//   inner face; scene cam sees Panel B's inner face.
//
//   Side profile:                    Top-down:
//       |       |                       | ----- |
//       |  HS   |                       |  HS   |
//       |_______|                       | ----- |
//   Panel A   Panel B                Panel A   Panel B
//
// COORDINATE FRAME (used by marker_positions.json)
//   Origin: midpoint of base top surface, between the two panel inner faces.
//   +X: from Panel A toward Panel B (along the inter-panel separation).
//   +Y: lateral, along panel width.
//   +Z: up (vertical).
//   Panel A inner face is at x = -panel_separation_x/2, normal = +X.
//   Panel B inner face is at x = +panel_separation_x/2, normal = -X.
//
// RENDER / EXPORT
//   Open in OpenSCAD (use a recent development snapshot — the 2021.01 stable
//   is stale): F5 = preview, F6 = full render, File > Export > STL.
//   The red marker outlines use OpenSCAD's `%` background modifier — they
//   appear in F5 preview only and are automatically excluded from F6
//   render and STL export. show_marker_outlines just toggles their
//   visibility in preview; it can't affect the print.
//
// SIDECAR MARKER POSITIONS
//   Run `python3 generate_marker_positions.py` to produce marker_positions.json.
//   That script parses the parameter values below directly from this file, so
//   keep all parameter declarations as simple `name = value;` lines.
//
// PRINTING
//   - Print upright on the base. Both panel inner faces are vertical during
//     the print. Use 0.15 mm layers and a slow first layer.
//   - Paper ArUco markers stick to the inner faces after printing. The plastic
//     is a reference plane, not the optical feature — layer texture under the
//     sticker doesn't affect detection.
//   - A brim helps adhesion on the tall walls.
// =============================================================================

// ---- PARAMETERS (simple `name = value;` lines — parsed by Python sidecar) ----

// Inter-panel geometry (mm)
panel_separation_x = 304.8;   // 12 in — between panel inner faces (~146 mm working distance per camera)
panel_width        = 130;     // along Y (lateral) — sized so a 3x3 grid of marker_size fits without overlap
panel_height       = 130;     // along Z (vertical)
panel_thickness    = 4;

// Base (connects the two panels)
base_thickness     = 5;
base_overhang      = 8;       // base extends past panel outer faces by this

// Stiffening ribs on the OUTSIDE of each panel (don't intrude on marker face)
rib_count          = 2;
rib_thickness      = 4;
rib_depth          = 18;      // how far the rib extends outward from panel
rib_height_frac    = 0.6;     // rib height as fraction of panel_height

// Marker slot layout — parametric starting point. Edit if you want a different
// grid; the sidecar JSON regenerates from these values. Sized so the eye cam
// (the lower-resolution of the two) resolves markers comfortably — the scene
// cam handles this size trivially.
marker_size        = 30;      // ArUco marker edge (mm), same on both panels
marker_grid_cols   = 3;       // markers along Y
marker_grid_rows   = 3;       // markers along Z
marker_margin      = 6;       // panel-edge → outer marker edge (mm)

// Visualization toggle (preview only; geometry is `%`-marked so it's never
// included in F6 render or STL export regardless of this setting)
show_marker_outlines = 1;

// ---- END PARAMETERS ----

$fn = 64;

// Derived helpers
base_length_x = panel_separation_x + 2 * panel_thickness + 2 * base_overhang;
base_width_y  = panel_width;

module base_plate() {
    translate([-base_length_x/2, -base_width_y/2, 0])
        cube([base_length_x, base_width_y, base_thickness]);
}

module marker_panel(side) {
    // side = -1 for Panel A, +1 for Panel B
    inner_face_x = side * panel_separation_x / 2;
    outer_face_x = inner_face_x + side * panel_thickness;
    x_lo = min(inner_face_x, outer_face_x);
    translate([x_lo, -panel_width/2, base_thickness])
        cube([panel_thickness, panel_width, panel_height]);
}

module rib(side, i) {
    inner_face_x = side * panel_separation_x / 2;
    outer_face_x = inner_face_x + side * panel_thickness;
    y_step   = panel_width / (rib_count + 1);
    y_center = -panel_width/2 + (i + 1) * y_step;
    rib_x = (side == 1) ? outer_face_x : outer_face_x - rib_depth;
    translate([rib_x, y_center - rib_thickness/2, base_thickness])
        cube([rib_depth, rib_thickness, panel_height * rib_height_frac]);
}

// Visualization only — thin rectangles on each panel's inner face showing
// where ArUco stickers go. Called with `%` from the bottom of the file so
// this geometry is preview-only (excluded from F6 render and STL export).
module marker_outlines() {
    x_thin = 0.2;
    color([0.9, 0.2, 0.2, 0.85])
    for (side = [-1, 1]) {
        inner_face_x = side * panel_separation_x / 2;
        y_span = panel_width - 2 * marker_margin - marker_size;
        z_min  = base_thickness + marker_margin + marker_size/2;
        z_max  = base_thickness + panel_height - marker_margin - marker_size/2;
        for (ci = [0 : marker_grid_cols - 1])
            for (ri = [0 : marker_grid_rows - 1]) {
                y_c = (marker_grid_cols > 1)
                    ? -y_span/2 + ci * y_span / (marker_grid_cols - 1)
                    : 0;
                z_c = (marker_grid_rows > 1)
                    ? z_min + ri * (z_max - z_min) / (marker_grid_rows - 1)
                    : (z_min + z_max) / 2;
                // place the slab on the inner-facing side of the panel
                x_slab = (side == 1) ? inner_face_x - x_thin : inner_face_x;
                translate([x_slab, y_c - marker_size/2, z_c - marker_size/2])
                    cube([x_thin, marker_size, marker_size]);
            }
    }
}

module assembly() {
    base_plate();
    marker_panel(-1);
    marker_panel(+1);
    for (s = [-1, 1])
        for (i = [0 : rib_count - 1])
            rib(s, i);
}

assembly();
// % makes marker_outlines preview-only — never in F6 render or STL.
if (show_marker_outlines) %marker_outlines();
