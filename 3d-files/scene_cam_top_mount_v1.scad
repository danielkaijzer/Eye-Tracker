// ============================================================================
// scene_cam_top_mount_v1.scad
// Replacement backing mount ("blue bar") for the HUAQUE HQ-L103 SCENE camera.
//
// Goals:
//   * Lock the PCB down using its OWN mounting holes (no more shifting).
//   * Reuse the existing white rod clips unchanged -> replicate the L-tab ears
//     (a rounded ear at each end with a Ø3.2 clip hole at 64 mm pitch), taken
//     from the current bar's STEP file.
//
// Retention: the board drops into a full-perimeter pocket (locates it), then
// two M2 screws through its two end mounting holes clamp it flat to the plate.
// The lens/LED/components face forward and stay fully exposed; a center window
// gives component clearance + airflow (per NOTES.md).
//
// PCB dimensions are from the manufacturer datasheet (62 x 9 x 1.0 mm board;
// two Ø2.2 holes 4.0 mm apart, 2.0 mm from the end).
// ============================================================================

/* =========================================================================
   VIEW  (pick one, then F5)
   ========================================================================= */
show = "assembly";   // "assembly" | "exploded" | "mount" | "pcb"

/* =========================================================================
   PCB  (from datasheet - measured)
   ========================================================================= */
pcb_len       = 62.0;   // board length
pcb_width     = 9.0;    // board width
board_thick   = 1.0;    // bare PCB thickness ("1.0pcb" on the drawing)
lens_protrude = 4.68;   // how far the camera module stands off the front face
                        // (informational: the front is left open)

// Two plated mounting holes at ONE end (the camera is 32.5 mm from this end):
hole_dia      = 2.2;    // Ø2.2 drilled hole (Ø3.2 is the pad); clears an M2 screw
hole_spacing  = 4.0;    // center-to-center, across the width, symmetric about center
hole_from_end = 2.0;    // hole-center distance from the mounting-hole end edge

// Optional third mounting hole near the connector end (datasheet shows a Ø2.2):
third_hole      = false;
third_from_conn = 10.2; // its center distance from the connector-end edge

/* =========================================================================
   CLIP INTERFACE  (from the current bar's STEP file - fixed)
   The white clips bolt to a rounded "ear" at each end with a Ø3.2 hole.
   ========================================================================= */
clip_hole_dia   = 3.2;   // clip screw clearance hole
clip_hole_pitch = 64.0;  // distance between the two clip screws
clip_end_margin = 3.0;   // ears sit 3 mm in from each bar end
bar_len         = clip_hole_pitch + 2*clip_end_margin;   // -> 70 mm
clip_ear_dia    = 7.0;   // rounded ear around the clip hole
clip_ear_drop   = 2.0;   // how far the ear's hole sits beyond the rod-side edge
                         //   (tune this so the clips reach the rod: test-fit)

/* =========================================================================
   PLATE + RETENTION  (tunable)
   ========================================================================= */
frame          = 2.5;    // border of plate around the PCB on the long edges
plate_t        = 3.0;    // backing-plate thickness
pocket_clear   = 0.30;   // gap around the PCB in the pocket (per side)
use_insert     = false;  // false = M2 self-tap into plastic; true = heat-set insert
screw_pilot_dia= 1.6;    // pilot for an M2 self-tapping screw
insert_dia     = 3.2;    // hole for an M2 brass heat-set insert
win_end_inset  = 8.0;    // airflow window kept this far from each PCB end
win_edge_inset = 2.0;    // airflow window kept this far from each long edge
conn_notch_w   = 6.0;    // cable/connector relief width at the connector end
lip_reach      = 1.0;    // connector-end hold-down lip overhang
lip_w          = 3.0;    // width of each connector-end lip
lip_h          = 1.0;    // thickness of each lip

$fn = 48;

/* =========================================================================
   DERIVED
   ========================================================================= */
plate_h  = pcb_width + 2*frame;         // ~14 mm
pcb_x0   = (bar_len - pcb_len)/2;        // PCB left edge (centered) -> 4
pcb_y0   = (plate_h - pcb_width)/2;      // PCB bottom edge -> 2.5
pocket_z = plate_t - board_thick;        // pocket floor -> 2.0
board_top= plate_t;

// +X end = mounting-hole end;  -X end = connector end
hole_x   = pcb_x0 + pcb_len - hole_from_end;
hole_ys  = [ plate_h/2 - hole_spacing/2, plate_h/2 + hole_spacing/2 ];
third_x  = pcb_x0 + third_from_conn;

// clip ears on the rod-side long edge (y = 0), at each end
clip_xs   = [ clip_end_margin, clip_end_margin + clip_hole_pitch ];
clip_ear_y = -clip_ear_drop;

// datasheet feature positions (from the mounting-hole end), for the mock
cam_x = hole_x - (32.5 - hole_from_end);   // camera 32.5 mm from that end
led_x = hole_x - (12.5 - hole_from_end);   // LED  12.5 mm from that end

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
                translate([third_x - pcb_x0, pcb_width/2, -1])
                    cylinder(h = board_thick + 2, d = hole_dia);
        }
    color("Black")     translate([cam_x, plate_h/2, board_top]) cylinder(h = lens_protrude, d = 5.68);
    color("Khaki")     translate([led_x, plate_h/2, board_top]) cube([3.5,3.5,1.2], center=true);
    color("Gainsboro") translate([pcb_x0-4, plate_h/2, pocket_z]) cube([4, 6, 2.6]); // connector
}

// The backing mount
module top_mount() {
    difference() {
        union() {
            cube([bar_len, plate_h, plate_t]);
            // clip ears (L-tabs) on the rod-side edge at each end
            for (x = clip_xs)
                translate([x, clip_ear_y, 0]) cylinder(h = plate_t, d = clip_ear_dia);
        }

        // PCB pocket (locates the board)
        translate([pcb_x0 - pocket_clear, pcb_y0 - pocket_clear, pocket_z])
            cube([pcb_len + 2*pocket_clear, pcb_width + 2*pocket_clear, board_thick + 10]);

        // airflow / component-relief window (clear of the screw pilots)
        translate([pcb_x0 + win_end_inset, pcb_y0 + win_edge_inset, -1])
            cube([pcb_len - 2*win_end_inset, pcb_width - 2*win_edge_inset, plate_t + 2]);

        // connector / cable relief at the -X end
        translate([-1, plate_h/2 - conn_notch_w/2, pocket_z])
            cube([pcb_x0 + 1.1, conn_notch_w, board_thick + 10]);

        // clip screw holes (through the ears)
        for (x = clip_xs)
            translate([x, clip_ear_y, -1]) cylinder(h = plate_t + 2, d = clip_hole_dia);

        // mounting-screw holes (self-tap pilot, or insert bore)
        for (y = hole_ys)
            translate([hole_x, y, -1])
                cylinder(h = plate_t + 2, d = use_insert ? insert_dia : screw_pilot_dia);
        if (third_hole)
            translate([third_x, plate_h/2, -1])
                cylinder(h = plate_t + 2, d = use_insert ? insert_dia : screw_pilot_dia);
    }

    // connector-end hold-down lips (flank the cable notch, hold that end flat)
    for (sy = [-1, 1])
        translate([pcb_x0 - pocket_clear - 1.0,
                   plate_h/2 + sy*(conn_notch_w/2 + lip_w/2) - lip_w/2,
                   board_top])
            cube([1.0 + lip_reach, lip_w, lip_h]);
}

/* =========================================================================
   OUTPUT
   ========================================================================= */
if (show == "assembly") {
    color("LightSteelBlue") top_mount();
    pcb_mock();
} else if (show == "exploded") {
    color("LightSteelBlue") top_mount();
    translate([0, 0, 16]) pcb_mock();
} else if (show == "mount") {
    top_mount();          // export this to STL
} else if (show == "pcb") {
    pcb_mock();
}
