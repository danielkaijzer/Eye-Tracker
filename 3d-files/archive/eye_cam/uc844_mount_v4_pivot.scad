// ============================================================================
// UC-844 Rev.B IR eye camera mount — v4, rod-hugging clamp + tilt pivot
//
// Why v4 (vs uc844_mount_v3_rod_clamp.scad):
//   The printed v3 aimed the camera too high and held it too far from the eye.
//   Two causes, both fixed here:
//     1. A long rigid spine (24 mm) cantilevered the frame away from the rod and
//        stacked three fixed angles -> overshoot. GONE: the frame now hangs right
//        off the clamp on a short arm.
//     2. The v3 rod clamp bore was cut so undersized the two halves sat a big gap
//        off the rod. FIXED: the bore hugs the rod (halves nearly meet).
//
// New geometry (per the user):
//   * The glasses rod crosses the camera frame at ~1/3 up from the frame's bottom
//     (bottom third of the PCB hangs below the rod).
//   * The frame is TILTED to aim the centered lens up at the pupil.
//   * The aim angle can't be measured well by hand, so the tilt is a friction
//     PIVOT (axis parallel to the rod): wear it, watch the live feed, rotate until
//     the pupil is centered/sharp, tighten one screw to lock. A slot at the pivot
//     also gives a few mm of standoff (lens-to-eye distance) adjustment.
//
//   The square frame + 4-corner PCB screw mount is carried over UNCHANGED from v3.
//
// Printed pieces:
//   "mount"  - clamp base half + arm + pivot pad (one body)
//   "cap"    - clamp upper half (bolts down over the rod)
//   "frame"  - camera frame + pivot ear on its back (rotates against the mount pad)
// Plus screws: 2 clamp, 1 pivot, 4 PCB corners (M2.5 heat-set inserts as in v3).
// ============================================================================

$fn = 60;

// "assembly" (all pieces + rod + tilt applied) | "exploded"
// | "mount" | "cap" | "frame"  (export each printable body to its own STL)
show = "assembly";

// ---------------- PCB / camera board (carried from v3) ------------------
pcb_size        = 38;    // UC-844 Rev.B board is 38x38mm square
pcb_thickness   = 1.6;
hole_dia        = 2.85;

// ---------------- corner mounting holes (carried from v3) ---------------
hole_pitch      = 34;
insert_hole_dia = 3.6;   // pilot for an M2.5 heat-set insert
insert_boss_dia = 7.0;
corner_pad_margin = 1.0;

// ---------------- open picture-frame (carried from v3) ------------------
frame_wall         = 3.0;
register_depth     = 1.0;
register_clearance = 0.10;
lead_in_depth      = max(pcb_thickness - register_depth, 0.6);
lead_in_clearance  = 0.30;
corner_relief_r    = 0.6;

frame_outer  = pcb_size + 2*frame_wall;   // 44
frame_height = register_depth + lead_in_depth;

// ---------------- glasses rod (measured) --------------------------------
rod_x = 5.0;   // fore-aft width (looking straight down)
rod_z = 4.0;   // up-down thickness (looking front-on)

// ---------------- rod clamp (tight two-piece screw clamp) ---------------
// Fix vs v3: squeeze is small, so each half's groove is only slightly shallower
// than the rod half-thickness -> the rod stands barely proud and the halves close
// almost fully (no v3 gap). Width gets a hair of clearance so it still drops in.
clamp_bore_x_clear = 0.30;   // per-side width slop
clamp_bore_squeeze = 0.15;   // per-half groove undersize (v3 was 0.5 -> big gap)
clamp_wall         = 2.5;    // material below the groove
clamp_width        = 14.0;   // grip length along the rod
clamp_insert_hole_dia = 3.6; // base: heat-set insert pilot (cap: clearance)
clamp_insert_boss_dia = 7.0;
clamp_screw_offset    = 7.0; // X from rod center to each clamp screw

channel_r_x = rod_x/2 + clamp_bore_x_clear;
channel_r_z = rod_z/2 - clamp_bore_squeeze;
clamp_half_h = channel_r_z + clamp_wall;
clamp_gap    = max(rod_z - 2*channel_r_z, 0);          // tiny (~0.3)
clamp_len_x  = 2*clamp_screw_offset + clamp_insert_boss_dia + 2;

// ---------------- arm + tilt pivot (SIDE-mounted, back kept open) -------
// To leave the PCB's back fully open (USB connector + heat), the pivot does NOT
// touch the frame's back. Instead a bracket grabs ONE solid side rail of the
// picture frame at ~1/3 up, and ALL the hardware (bracket disc, arm, clamp) sits
// BESIDE the frame along the rod (Y) - nothing behind the board.
//
// The joint is a single-lap friction disc: the frame's bracket disc bears flat
// against the mount's disc, one screw on the rod-parallel axis (Y) locks the tilt.
// The disc hole is a fore/aft slot, so the same screw also sets standoff.
//
// Consequence: the frame hangs to one side of the clamp (~half a frame-width).
// That's fine - clamp toward the temple, camera over the pupil, hardware out of
// your sightline.
tilt_angle = 30;     // NOMINAL aim (deg above horizontal); the pivot overrides this
                     // on your face - it's only where the part is drawn/printed.
pivot_frac = 1/3;    // rod crosses the frame this far up from its bottom edge

pivot_pad_dia    = 14.0;  // friction disc diameter (frame bracket + mount)
disc_th          = 4.0;   // each disc's thickness (Y)
disc_gap         = 1.0;   // clearance between the clamp side and the mount disc
bracket_h        = 12.0;  // vertical bond length of the bracket onto the side rail
pivot_insert_dia = 3.6;   // heat-set insert pilot in the mount disc (M2.5)
pivot_screw_dia  = 2.9;   // screw clearance (frame disc + slot)
pivot_slot_travel = 6.0;  // disc slot length -> +/- standoff adjustment
arm_w  = 8.0;             // arm width (X)
arm_th = 4.0;             // arm thickness (Z)

pivot_x = 0;                                       // pivot axis X (~clamp center)
pivot_z = clamp_half_h/2;                          // Z (mid clamp base, solidly fused)
pivot_y = clamp_width/2 + disc_gap + disc_th;      // Y of the friction interface plane

// ============================================================================
// FRAME MODULES  (verbatim from uc844_mount_v3_rod_clamp.scad)
// ============================================================================
module oval_x_z(rx, rz) { scale([rx, rz, 1]) circle(r = 1); }

module pocket_profile(clearance, relief) {
    union() {
        offset(delta = clearance) square([pcb_size, pcb_size], center = true);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * pcb_size/2, y * pcb_size/2]) circle(r = relief);
    }
}

module pcb_pocket() {
    union() {
        linear_extrude(height = register_depth)
            pocket_profile(register_clearance, corner_relief_r);
        translate([0, 0, register_depth])
            linear_extrude(height = lead_in_depth + 0.02)
                pocket_profile(lead_in_clearance, corner_relief_r + lead_in_clearance);
    }
}

module corner_pad_2d(xs, ys) {
    reach    = hole_pitch/2 - insert_boss_dia/2 - corner_pad_margin;
    pad_span = frame_outer/2 - reach;
    chamfer  = pad_span * 0.6;
    mirror([xs < 0 ? 1 : 0, 0, 0])
    mirror([0, ys < 0 ? 1 : 0, 0])
    union() {
        difference() {
            translate([reach, reach]) square([pad_span, pad_span]);
            translate([reach, reach]) polygon([[0,0],[chamfer,0],[0,chamfer]]);
        }
        translate([hole_pitch/2, hole_pitch/2]) circle(d = insert_boss_dia);
    }
}

module corner_pad(xs, ys) {
    difference() {
        linear_extrude(height = frame_height) corner_pad_2d(xs, ys);
        translate([xs * hole_pitch/2, ys * hole_pitch/2, -0.5])
            cylinder(h = frame_height + 1, d = insert_hole_dia);
    }
}

module frame_body() {
    union() {
        difference() {
            linear_extrude(height = frame_height)
                square([frame_outer, frame_outer], center = true);
            pcb_pocket();
        }
        for (x = [-1, 1], y = [-1, 1]) corner_pad(x, y);
    }
}

// ============================================================================
// ROD CLAMP MODULES  (base half fused to the mount; cap printed separately)
// ============================================================================
module clamp_half(is_cap) {
    difference() {
        union() {
            translate([-clamp_len_x/2, -clamp_width/2, 0])
                cube([clamp_len_x, clamp_width, clamp_half_h]);
            if (!is_cap)
                for (x = [-clamp_screw_offset, clamp_screw_offset])
                    translate([x, 0, 0])
                        cylinder(h = clamp_half_h + 3, d = clamp_insert_boss_dia);
        }
        // half-oval rod groove on the parting face
        translate([0, 0, is_cap ? 0 : clamp_half_h])
            rotate([90, 0, 0])
                linear_extrude(height = clamp_width + 2, center = true)
                    oval_x_z(channel_r_x, channel_r_z);
        // clamp screw holes
        for (x = [-clamp_screw_offset, clamp_screw_offset])
            translate([x, 0, -0.5])
                cylinder(h = clamp_half_h + 4, d = clamp_insert_hole_dia);
    }
}

module rod_mock() {
    color("DimGray")
        translate([0, 0, clamp_half_h])
            rotate([90, 0, 0])
                linear_extrude(height = clamp_width + 40, center = true)
                    oval_x_z(rod_x/2, rod_z/2);
}

// ============================================================================
// FRAME PIECE  (unchanged camera frame + a side bracket on ONE rail)
// Local coords: pivot axis is +Y through the origin, which is the friction face.
// The frame sits at +Y (its near side rail just past the bracket disc), pocket
// facing +X (toward the eye) at tilt = 0, `pivot_frac` up at z = 0. Nothing lands
// on the frame's back. The whole piece is rotated by `tilt_angle` when placed.
// ============================================================================
module frame_piece() {
    off = frame_outer*(0.5 - pivot_frac);      // land the 1/3-up point on the axis (z=0)
    rail_y0 = disc_th;                          // near side-rail outer face (local Y)
    fy      = rail_y0 + frame_outer/2;          // frame center in local Y

    // frame body - byte-for-byte the v3 frame, just positioned beside the pivot
    translate([0, fy, 0])
        rotate([0, 90, 0])
            translate([-off, 0, 0])
                frame_body();

    // side bracket: friction disc at the axis + a neck fused down the near rail
    difference() {
        union() {
            rotate([-90, 0, 0]) cylinder(h = disc_th, d = pivot_pad_dia);   // disc, y 0..disc_th
            // neck bridges the disc to the near rail (x 0..frame_height), bonding
            // over the rail's height - stays outboard of the open window
            translate([-1.0, disc_th - 1.0, -bracket_h/2])
                cube([frame_height + 2.0, rail_y0 + frame_wall - (disc_th - 1.0), bracket_h]);
        }
        // pivot slot (axis Y), elongated fore/aft for standoff adjustment
        hull() for (dx = [-pivot_slot_travel/2, pivot_slot_travel/2])
            translate([dx, -1, 0])
                rotate([-90, 0, 0])
                    cylinder(h = disc_th + 2, d = pivot_screw_dia);
    }
}

// ============================================================================
// MOUNT BODY  (clamp base + side arm + pivot disc)
// ============================================================================
module mount_body() {
    clamp_half(false);

    // arm: from the clamp +Y face out to the mount disc, beside the frame
    hull() {
        translate([pivot_x - arm_w/2, clamp_width/2 - 1, pivot_z - arm_th/2])
            cube([arm_w, 0.1, arm_th]);
        translate([pivot_x - arm_w/2, pivot_y - disc_th, pivot_z - arm_th/2])
            cube([arm_w, 0.1, arm_th]);
    }

    // mount disc: bears against the frame's bracket disc; insert bored from -Y
    translate([pivot_x, pivot_y - disc_th, pivot_z])
        difference() {
            rotate([-90, 0, 0]) cylinder(h = disc_th, d = pivot_pad_dia);
            translate([0, -1, 0])
                rotate([-90, 0, 0]) cylinder(h = disc_th + 1.2, d = pivot_insert_dia);
        }
}

// ============================================================================
// OUTPUT
// ============================================================================
module place_frame() {
    translate([pivot_x, pivot_y, pivot_z]) rotate([0, tilt_angle, 0]) children();
}

if (show == "assembly") {
    color("LightSteelBlue") mount_body();
    color("Silver") translate([0, 0, clamp_gap]) rod_mock();
    color("LightSteelBlue") translate([0, 0, clamp_half_h + clamp_gap]) clamp_half(true);
    color("Khaki") place_frame() frame_piece();
} else if (show == "exploded") {
    ex = 16;
    mount_body();
    translate([0, 0, ex]) rod_mock();
    color("LightSteelBlue") translate([0, 0, clamp_half_h + 2*ex]) clamp_half(true);
    color("Khaki") translate([0, 22, 0]) place_frame() frame_piece();
} else if (show == "mount") {
    mount_body();
} else if (show == "cap") {
    clamp_half(true);
} else if (show == "frame") {
    frame_piece();
}
