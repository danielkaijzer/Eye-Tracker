// ============================================================================
// scene_cam_top_mount_v1.scad
// Replacement backing mount ("blue bar") for the HUAQUE HQ-L103 SCENE camera.
//
// Goal: stop the PCB from shifting by locating it on its OWN mounting holes,
// while keeping the existing white rod clips completely unchanged.
//
// How it locks the board:
//   1. A full-perimeter pocket (board outline + clearance) -> no slide/rotate.
//   2. Two locating posts through the board's two end holes -> precise + no spin.
//   3. Snap post-heads + two connector-end lips -> hold it flat, still removable.
//   4. An open center window -> component clearance + airflow (per NOTES.md).
//
// The clip interface (two Ø3.2 holes at 64 mm pitch) was extracted from the
// existing STEP files, so the current white clips bolt straight on.
// ============================================================================

/* =========================================================================
   VIEW  (pick one, then F5)
   ========================================================================= */
show = "assembly";   // "assembly" | "exploded" | "mount" | "pcb"

/* =========================================================================
   >>> MEASURE THESE on your actual PCB with calipers. <<<
   The values below are ESTIMATES read off your photos - confirm before print.
   ========================================================================= */
pcb_len       = 58.0;   // long dimension of the board
pcb_width     = 8.5;    // short dimension of the board
pcb_thick     = 1.2;    // board thickness

// The two plated mounting holes are near ONE end, stacked across the width:
hole_dia      = 2.1;    // diameter of each mounting hole
hole_from_end = 2.5;    // hole-center distance from the nearest board end (lengthwise)
hole_spacing  = 5.0;    // center-to-center across the width (must be < pcb_width)

/* =========================================================================
   FIXED interface to the existing white clips (extracted from the STEP files).
   Do NOT change clip_hole_dia / clip_hole_pitch or the clips won't line up.
   ========================================================================= */
clip_hole_dia   = 3.2;   // Ø3.2 clearance, matches current bar + clips
clip_hole_pitch = 64.0;  // 64 mm between the two clip screws
clip_end_margin = 3.0;   // clip holes sit 3 mm in from each bar end
bar_len         = clip_hole_pitch + 2*clip_end_margin;   // -> 70 mm

/* =========================================================================
   PLATE + RETENTION (tunable)
   ========================================================================= */
frame         = 2.5;    // border of plate around PCB on the long edges
plate_t       = 3.0;    // backing-plate thickness
pocket_clear  = 0.30;   // gap around PCB in the pocket (per side)
post_clear    = 0.15;   // gap around locating posts in the holes (per side)
post_extra    = 1.4;    // how far the posts stand above the board top
post_snap     = true;   // add a chamfered snap head to the posts
snap_lip      = 0.4;    // how far the snap head overhangs the hole
win_end_inset = 8.0;    // window kept this far from each PCB end (solid seat)
win_edge_inset= 2.0;    // window kept this far from each long edge
lip_reach     = 1.0;    // connector-end hold-down lip overhang
lip_w         = 3.0;    // width of each connector-end lip
lip_h         = 1.0;    // thickness of each lip
tab_gap       = 0.15;   // clearance under snap features (over board top)

$fn = 48;

/* =========================================================================
   DERIVED
   ========================================================================= */
plate_h  = pcb_width + 2*frame;          // overall bar height (~13.5 mm)
pcb_x0   = (bar_len - pcb_len)/2;         // PCB left edge (centered)
pcb_y0   = (plate_h - pcb_width)/2;       // PCB bottom edge (centered)
pocket_z = plate_t - pcb_thick;           // pocket floor height
board_top= plate_t;                       // PCB top is flush with plate top

// mounting-hole centers (near the +X end of the board)
hole_x   = pcb_x0 + pcb_len - hole_from_end;
hole_ys  = [ plate_h/2 - hole_spacing/2, plate_h/2 + hole_spacing/2 ];
post_d   = hole_dia - 2*post_clear;

// clip-hole centers (ends, clear of the PCB)
clip_xs  = [ clip_end_margin, clip_end_margin + clip_hole_pitch ];
clip_y   = plate_h/2;

/* =========================================================================
   MODULES
   ========================================================================= */

// Visual stand-in for the scene PCB (with its two mounting holes + a lens bump)
module pcb_mock() {
    color("DarkSlateGray")
    translate([pcb_x0, pcb_y0, pocket_z])
        difference() {
            cube([pcb_len, pcb_width, pcb_thick]);
            for (y = hole_ys)
                translate([hole_x - pcb_x0, y - pcb_y0, -1])
                    cylinder(h = pcb_thick + 2, d = hole_dia);
        }
    // lens bump on the FRONT face, to show which way the camera looks
    color("Black")
    translate([bar_len/2, plate_h/2, board_top])
        cylinder(h = 2.5, d = 4);
}

// The backing mount itself
module top_mount() {
    union() {
        difference() {
            // ---- main plate ----
            cube([bar_len, plate_h, plate_t]);

            // ---- PCB pocket (recess so the board drops in flush) ----
            translate([pcb_x0 - pocket_clear, pcb_y0 - pocket_clear, pocket_z])
                cube([pcb_len + 2*pocket_clear,
                      pcb_width + 2*pocket_clear,
                      pcb_thick + 10]);

            // ---- airflow / component-relief window (kept clear of posts) ----
            translate([pcb_x0 + win_end_inset, pcb_y0 + win_edge_inset, -1])
                cube([pcb_len - 2*win_end_inset,
                      pcb_width - 2*win_edge_inset,
                      plate_t + 2]);

            // ---- clip screw holes (the interface to the white clips) ----
            for (x = clip_xs)
                translate([x, clip_y, -1])
                    cylinder(h = plate_t + 2, d = clip_hole_dia);
        }

        // ---- locating posts (through the board's own holes) ----
        // start 0.6 mm below the pocket floor so they fuse to the plate.
        for (y = hole_ys)
            translate([hole_x, y, pocket_z - 0.6]) {
                cylinder(h = 0.6 + pcb_thick + post_extra, d = post_d);   // shaft
                if (post_snap)
                    translate([0, 0, 0.6 + pcb_thick + tab_gap])          // snap head
                        cylinder(h = post_extra,
                                 d1 = post_d + 2*snap_lip, d2 = post_d - 0.3);
            }

        // ---- connector-end hold-down lips ----
        // each lip is one cube: the first ~1 mm sits on solid frame (fused),
        // the rest overhangs the board top (which is flush at plate_t).
        for (sy = [-1, 1])
            translate([pcb_x0 - pocket_clear - 1.0,
                       plate_h/2 + sy*(hole_spacing/2 + lip_w/2) - lip_w/2,
                       board_top])
                cube([1.0 + lip_reach, lip_w, lip_h]);
    }
}

/* =========================================================================
   OUTPUT
   ========================================================================= */
if (show == "assembly") {
    color("LightSteelBlue") top_mount();
    pcb_mock();
} else if (show == "exploded") {
    color("LightSteelBlue") top_mount();
    translate([0, 0, 14]) pcb_mock();
} else if (show == "mount") {
    top_mount();          // export this one to STL
} else if (show == "pcb") {
    pcb_mock();
}
