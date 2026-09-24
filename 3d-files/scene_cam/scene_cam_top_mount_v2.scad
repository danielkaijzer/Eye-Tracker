// ============================================================================
// scene_cam_top_mount_v2.scad
// Backing mount for the HUAQUE HQ-L103 SCENE camera + parametric rod clip.
//
// CHANGES FROM v1
//   * The imported white clip (clip_ref.stl) is replaced by a parametric
//     rod_clip() module, sized from rod_w / rod_t.  Same clip is used on the
//     UC-844 eye mount -> ONE clamp design for both cameras.
//   * The clip screw sits 5.40 mm from the rod center (identical to the eye
//     mount v8 interface: clip_dx - 3.2 = -5.4), so v8's arms are unchanged
//     apart from arm_th -> tab_t.
//   * The L-tab "ears" are gone.  At 5.40 mm the screw lands inside the plate
//     body, so the clip screws straight through the plate near its bottom
//     edge.  Stiffer than a cantilevered ear and a smaller overall envelope.
//   * plate_t is no longer free: it IS the tab thickness, derived from the
//     slot, so the plate is a snug fit in the clip.
//
// CLIP LOCAL FRAME (used by rod_clip):
//     origin = ROD CENTER
//     +X     = mouth direction (points out of the slot, toward the plate)
//      Y     = along the rod
//      Z     = slot normal (jaws stack in Z)
//
// PCB dimensions are from the manufacturer datasheet (62 x 9 x 1.0 mm board;
// two O2.2 holes 4.0 mm apart, 2.0 mm from the end).
// ============================================================================

/* =========================================================================
   VIEW  (pick one, then F5)
   ========================================================================= */
show = "assembly";   // "assembly" | "exploded" | "mount" | "clip" | "pcb"

$fn = 48;

// #########################################################################
// ## SHARED CLIP INTERFACE
// ## Keep byte-identical with the UC-844 eye mount, or better: move this
// ## block to clip_iface.scad and `include <clip_iface.scad>` from both.
// #########################################################################

/* ---- the rod (eyewear frame member the clip grips) ---------------------- */
rod_w        = 5.0;    // rod width, along the clip's X (mouth direction)
rod_t        = 3.5;    // rod thickness, along the clip's Z  <-- 2.75 on the
                       //   eye-camera rod; MEASURE both before printing
rod_clear_x  = 0.20;   // clearance behind the rod, in X

/* ---- fit ---------------------------------------------------------------- */
snap_fit     = 0.10;   // slot is rod_t - snap_fit -> interference snap.
                       //   0.10 in PETG is a firm push-on. Set 0 for a slip
                       //   fit, 0.20 if it still rotates on the rod.
tab_clear    = 0.10;   // slop between the slot and the plate/arm tab

/* ---- screw interface (DO NOT CHANGE - eye mount v8 depends on it) -------- */
screw_offset = 5.40;   // rod center -> screw axis, along +X
screw_dia    = 3.2;    // M3 clearance

/* ---- clip body ---------------------------------------------------------- */
jaw_t        = 2.00;   // wall above/below the slot (stock part was 0.65-1.15,
                       //   which is ~2 perimeters and splits under screw load)
clip_w       = 8.00;   // width along the rod (stock 6.0; wider = less roll)
back_wall    = 2.00;   // material behind the rod, at the closed end
nose_wall    = 1.20;   // material outboard of the screw hole
lead_in      = 1.00;   // mouth chamfer depth, for snapping over the rod
lead_in_z    = 0.45;   // mouth chamfer flare, per side
clip_r       = 1.20;   // outer corner radius

/* ---- derived: everything downstream keys off these ---------------------- */
slot_h    = rod_t - snap_fit;              // 3.40
tab_t     = slot_h - tab_clear;            // 3.30  <-- plate_t AND arm_th
clip_h    = slot_h + 2*jaw_t;              // 7.40
x_slot    = -(rod_w/2 + rod_clear_x);      // -2.70, slot closed end
x_back    = x_slot - back_wall;            // -4.70, clip back face
x_mouth   = screw_offset + screw_dia/2 + nose_wall;   // 8.20, clip front face

// #########################################################################
// ## END SHARED BLOCK
// #########################################################################


/* =========================================================================
   PARAMETRIC ROD CLIP
   Prints 2 per camera.  Orient it on the bed with the slot VERTICAL (lay it
   on a side face) so the jaws bend in-plane instead of peeling layers.
   Fastener: M3 SHCS + washer + nyloc nut.  The 2.0 mm jaw is too thin to
   pocket a nut or a heat-set insert; the nut goes on the outside.
   ========================================================================= */
module rod_clip() {
    difference() {
        // rounded-rectangle prism, swept along the rod (Y)
        hull()
            for (x = [x_back + clip_r, x_mouth - clip_r],
                 z = [-(clip_h/2 - clip_r), clip_h/2 - clip_r])
                translate([x, 0, z])
                    rotate([90, 0, 0])
                        cylinder(h = clip_w, r = clip_r, center = true);

        // the slot: rod pocket + tab channel, one continuous cut
        translate([x_slot, -clip_w/2 - 1, -slot_h/2])
            cube([x_mouth - x_slot + 2, clip_w + 2, slot_h]);

        // mouth chamfer, so the rod can be snapped in past the interference
        hull() {
            translate([x_mouth - lead_in, -clip_w/2 - 1, -slot_h/2])
                cube([0.01, clip_w + 2, slot_h]);
            translate([x_mouth + 1, -clip_w/2 - 1, -slot_h/2 - lead_in_z])
                cube([0.01, clip_w + 2, slot_h + 2*lead_in_z]);
        }

        // clip screw
        translate([screw_offset, 0, -clip_h])
            cylinder(h = 2*clip_h, d = screw_dia);
    }
}

// visual stand-in for the rod (not printed)
module rod_mock(len = 90) {
    color("Silver")
        translate([0, 0, 0])
            rotate([90, 0, 0])
                linear_extrude(len, center = true)
                    offset(r = min(rod_w, rod_t)/2 - 0.01)
                        square([rod_w - min(rod_w,rod_t) + 0.02,
                                rod_t - min(rod_w,rod_t) + 0.02], center = true);
}

/* =========================================================================
   PCB  (from datasheet - measured)
   ========================================================================= */
pcb_len       = 62.0;   // board length
pcb_width     = 9.0;    // board width
board_thick   = 1.0;    // bare PCB thickness
lens_protrude = 4.68;   // how far the camera module stands off the front face

// Two plated mounting holes at ONE end (the camera is 32.5 mm from this end):
hole_dia      = 2.2;
hole_spacing  = 4.0;
hole_from_end = 2.0;

// Third mounting hole near the connector end -> 3-point clamp:
third_hole      = true;
third_from_conn = 10.2;
third_y_offset  = -2.0;

/* =========================================================================
   BAR / CLIP PLACEMENT
   ========================================================================= */
clip_hole_pitch = 64.0;  // free parameter - the clips slide along the rod.
                         //   Need NOT match the eye mount's 22.5.
clip_end_margin = 4.5;   // half the clip width + a little
bar_len         = clip_hole_pitch + 2*clip_end_margin;   // -> 73
edge_gap        = 0.50;  // gap between the rod and the plate's bottom edge

/* =========================================================================
   PLATE + RETENTION  (tunable)
   ========================================================================= */
plate_t        = tab_t;  // DERIVED - must equal the clip's tab thickness
frame_far      = 2.5;    // border above the PCB (away from the rod)
pocket_clear   = 0.30;
use_insert     = false;  // false = M2 self-tap into plastic; true = heat-set
screw_pilot_dia= 1.6;
insert_dia     = 3.2;
win_end_inset  = 8.0;
win_edge_inset = 2.0;
conn_relief_w  = 7.0;
lip_reach      = 1.0;
lip_h          = 1.0;
// lip width is DERIVED: the lips only ever hold down the strip of board left
// either side of the cable relief, so they span relief edge -> PCB edge.  A
// fixed 3 mm lip overhung the frame and fouled the clip's upper jaw.
lip_w          = (pcb_width - conn_relief_w)/2;   // -> 1.0

/* =========================================================================
   DERIVED
   ========================================================================= */
// rod center and clip screw, in plate coordinates (plate bottom edge at y=0)
rod_y       = -(rod_w/2 + rod_clear_x + edge_gap);   // -3.20
clip_screw_y= rod_y + screw_offset;                  // +2.20
clip_reach_y= rod_y + x_mouth;                       // +5.00, clip's far face

// bottom border must clear the screw hole AND the clip body
frame_rod = max(frame_far,
                clip_screw_y + screw_dia/2 + 1.2,    // wall around the hole
                clip_reach_y + 0.5);                 // clip must not foul the PCB

plate_h  = frame_rod + pcb_width + frame_far;
pcb_x0   = (bar_len - pcb_len)/2;
pcb_y0   = frame_rod;
pocket_z = plate_t - board_thick;
board_top= plate_t;

// +X end = mounting-hole end;  -X end = connector end
hole_x   = pcb_x0 + pcb_len - hole_from_end;
hole_ys  = [ pcb_y0 + pcb_width/2 - hole_spacing/2,
             pcb_y0 + pcb_width/2 + hole_spacing/2 ];
third_x  = pcb_x0 + third_from_conn;
third_y  = pcb_y0 + pcb_width/2 + third_y_offset;

win_x0 = third_hole ? max(pcb_x0 + win_end_inset, third_x + 4) : pcb_x0 + win_end_inset;
win_x1 = pcb_x0 + pcb_len - win_end_inset;

clip_xs = [ clip_end_margin, clip_end_margin + clip_hole_pitch ];

// datasheet feature positions (from the mounting-hole end), for the mock
cam_x = hole_x - (32.5 - hole_from_end);
led_x = hole_x - (12.5 - hole_from_end);

echo(plate_t = plate_t, plate_h = plate_h, frame_rod = frame_rod,
     tab_t = tab_t, slot_h = slot_h, clip_h = clip_h,
     rod_y = rod_y, clip_screw_y = clip_screw_y);

/* =========================================================================
   MODULES
   ========================================================================= */

// Visual stand-in for the scene PCB (holes + lens + LED + connector)
module pcb_mock() {
    color("DarkSlateGray")
    translate([pcb_x0, pcb_y0, pocket_z])
        difference() {
            cube([pcb_len, pcb_width, board_thick]);
            for (y = hole_ys)
                translate([hole_x - pcb_x0, y - pcb_y0, -1])
                    cylinder(h = board_thick + 2, d = hole_dia);
            if (third_hole)
                translate([third_x - pcb_x0, third_y - pcb_y0, -1])
                    cylinder(h = board_thick + 2, d = hole_dia);
        }
    cy = pcb_y0 + pcb_width/2;
    color("Black") translate([cam_x, cy, board_top]) cylinder(h = lens_protrude, d = 5.68);
    color("Khaki") translate([led_x, cy - 1.75, board_top]) cube([3.5, 3.5, 1.2]);
    color("Gainsboro")
        translate([pcb_x0, cy - 3, board_top]) cube([5, 6, 2.6]);
    color("DimGray")
        translate([pcb_x0 - 7, cy - 2, board_top + 0.4]) cube([8, 4, 1.8]);
}

// The backing mount
module top_mount() {
    cy = pcb_y0 + pcb_width/2;
    difference() {
        cube([bar_len, plate_h, plate_t]);

        // PCB pocket (locates the board)
        translate([pcb_x0 - pocket_clear, pcb_y0 - pocket_clear, pocket_z])
            cube([pcb_len + 2*pocket_clear, pcb_width + 2*pocket_clear, board_thick + 10]);

        // airflow / component-relief window (clear of the screw pilots)
        translate([win_x0, pcb_y0 + win_edge_inset, -1])
            cube([win_x1 - win_x0, pcb_width - 2*win_edge_inset, plate_t + 2]);

        // connector / plug relief at the -X end
        translate([-3, cy - conn_relief_w/2, -1])
            cube([pcb_x0 + 2 + 3, conn_relief_w, plate_t + 2]);

        // CLIP SCREW HOLES - straight through the plate, no ears
        for (x = clip_xs)
            translate([x, clip_screw_y, -1])
                cylinder(h = plate_t + 2, d = screw_dia);

        // PCB mounting-screw holes (self-tap pilot, or insert bore)
        for (y = hole_ys)
            translate([hole_x, y, -1])
                cylinder(h = plate_t + 2, d = use_insert ? insert_dia : screw_pilot_dia);
        if (third_hole)
            translate([third_x, third_y, -1])
                cylinder(h = plate_t + 2, d = use_insert ? insert_dia : screw_pilot_dia);
    }

    // connector-end hold-down lips (flank the cable notch, hold that end flat)
    for (sy = [-1, 1])
        translate([pcb_x0 - pocket_clear - 1.0,
                   sy < 0 ? pcb_y0 : cy + conn_relief_w/2,
                   board_top])
            cube([1.0 + lip_reach, lip_w, lip_h]);
}

// place a clip into plate coordinates:
//   clip local +X -> plate +Y,  clip local Y -> plate X,  clip Z -> plate Z
module place_clip(x) {
    translate([x, rod_y, plate_t/2]) rotate([0, 0, 90]) children();
}

/* =========================================================================
   OUTPUT
   ========================================================================= */
if (show == "assembly") {
    color("LightSteelBlue") top_mount();
    pcb_mock();
    for (x = clip_xs) place_clip(x) color("GhostWhite") rod_clip();
    translate([bar_len/2, rod_y, plate_t/2]) rotate([0,0,90]) rod_mock(bar_len + 20);

} else if (show == "exploded") {
    color("LightSteelBlue") top_mount();
    translate([0, 0, 16]) pcb_mock();
    for (x = clip_xs)
        translate([0, 0, -16]) place_clip(x) color("GhostWhite") rod_clip();
    translate([bar_len/2, rod_y, plate_t/2 - 16]) rotate([0,0,90]) rod_mock(bar_len + 20);

} else if (show == "mount") {
    top_mount();          // export this to STL

} else if (show == "clip") {
    // laid on its side: slot vertical, layers along the rod axis
    rotate([90, 0, 0]) rod_clip();

} else if (show == "pcb") {
    pcb_mock();
}
