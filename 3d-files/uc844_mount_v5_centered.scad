// ============================================================================
// UC-844 Rev.B IR eye camera mount — v5, centered + back-open, thin rod
//
// Why v5 (vs uc844_mount_v4_pivot.scad):
//   v4 side-mounted the frame ~30 mm off the clamp, eating scarce across-the-face
//   space (only ~1 inch of usable rod). v5 keeps everything good about v4 (tight
//   two-piece rod clamp, friction TILT pivot + standoff slot, UNCHANGED frame +
//   4-corner PCB screw mount, fully open PCB back for the USB connector and heat)
//   but the mount is now CENTERED and compact:
//
//     clamp on the rod  ->  short rise arm DOWN  ->  friction tilt pivot at the
//     frame's bottom edge (a small clevis)  ->  frame hangs below the rod, tilted
//     up so the centered lens aims at the pupil.
//
//   Nothing sits behind the PCB; the pivot grabs only the solid bottom rail.
//
// Aim geometry is modeled explicitly (see EYE block): with the pupil ~27 mm above
// the bottom rod and ~19 mm behind the lens plane, the lens tip lands ~44 mm from
// the pupil at ~45 deg up-tilt. Tilt (pivot) + standoff (slot) + M12 refocus tune
// it on the face; `echo()` prints the tip->pupil distance/angle for the drawn model.
//
// Printed pieces:  "mount" (clamp base + rise arm + fork), "cap", "frame".
// ============================================================================

$fn = 60;

// "assembly" (all + eye/lens mocks) | "exploded" | "mount" | "cap" | "frame"
show = "assembly";

// ---------------- PCB / camera board (carried from v3/v4) ---------------
pcb_size        = 38;
pcb_thickness   = 1.6;
hole_dia        = 2.85;
hole_pitch      = 34;
insert_hole_dia = 3.6;
insert_boss_dia = 7.0;
corner_pad_margin = 1.0;

// ---------------- open picture-frame (carried) --------------------------
frame_wall         = 3.0;
register_depth     = 1.0;
register_clearance = 0.10;
lead_in_depth      = max(pcb_thickness - register_depth, 0.6);
lead_in_clearance  = 0.30;
corner_relief_r    = 0.6;
frame_outer  = pcb_size + 2*frame_wall;   // 44
frame_height = register_depth + lead_in_depth;

// ---------------- glasses rod (measured, thin) --------------------------
rod_x = 5.0;    // fore-aft depth
rod_z = 2.75;   // up-down thickness (2.5-3 mm, rounded)

// ---------------- rod clamp (tight two-piece, compact for the thin rod) --
clamp_bore_x_clear = 0.30;
clamp_bore_squeeze = 0.15;
clamp_wall         = 2.0;
clamp_width        = 14.0;   // grip along the rod (fits within ~1 inch usable)
clamp_insert_hole_dia = 3.6;
clamp_insert_boss_dia = 6.0;
clamp_screw_offset    = 5.0; // screws just outside the 5 mm-deep rod
channel_r_x = rod_x/2 + clamp_bore_x_clear;
channel_r_z = rod_z/2 - clamp_bore_squeeze;
clamp_half_h = channel_r_z + clamp_wall;
clamp_gap    = max(rod_z - 2*channel_r_z, 0);
clamp_len_x  = 2*clamp_screw_offset + clamp_insert_boss_dia + 2;   // ~18

// ---------------- eye / aim geometry (MEASURED; drives the nominal) ------
// Origin = clamp center; +X toward the face/eye; +Z up; rod along Y.
eye_depth = 19;   // pupil this far behind the glasses lens plane (0.75")
eye_rise  = 27;   // pupil this far above the bottom rod (2.7 cm)
lens_len  = 13;   // lens protrusion off the PCB front (mock / tip position)
rod_top_gap = 46; // bottom-rod -> top-rod spacing (context only)
eye = [eye_depth, 0, clamp_half_h + eye_rise];

// ---------------- centered tilt pivot (bottom-edge clevis) + rise arm ----
tilt_angle = 45;     // NOMINAL up-tilt; pivot overrides on the face
pivot_x    = -3;     // pivot axis X (world side of the rod)
pivot_z    = -31;    // pivot axis Z (below the rod; sets how low the frame hangs)
tab_reach  = 6;      // frame bottom edge sits this far above the pivot axis

pivot_pad_dia = 13.0;
prong_th   = 3.0;    // fork prong thickness (Y)
prong_clr  = 0.4;
tab_th_y   = 6.0;    // frame bottom tab thickness (Y), between the prongs
pivot_insert_dia = 3.6;   // insert in the far prong (M2.5)
pivot_screw_dia  = 2.9;   // clearance (near prong + tab slot)
pivot_slot_travel = 6.0;  // fore/aft standoff fine-tune
arm_w = 6.0;              // rise-arm / strut width (Y footprint per side)
rail_y = tab_th_y/2 + prong_clr + prong_th/2;

// derived nominal lens-tip position + report -----------------------------
_a = frame_height + lens_len;
_b = tab_reach + frame_outer/2;
lens_tip = [ pivot_x + _a*cos(tilt_angle) - _b*sin(tilt_angle), 0,
             pivot_z + _a*sin(tilt_angle) + _b*cos(tilt_angle) ];
tip_to_eye  = norm(lens_tip - eye);
aim_deg     = atan2(eye[2]-lens_tip[2], eye[0]-lens_tip[0]);  // needed elevation
echo(lens_tip = lens_tip, pupil = eye);
echo(tip_to_eye_mm = tip_to_eye, lens_axis_deg = tilt_angle, aim_needed_deg = aim_deg);

// ============================================================================
// FRAME MODULES  (verbatim from v3/v4)
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
    mirror([xs < 0 ? 1 : 0, 0, 0]) mirror([0, ys < 0 ? 1 : 0, 0])
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
// ROD CLAMP  (base half fused to the mount; cap printed separately)
// ============================================================================
module clamp_half(is_cap) {
    difference() {
        union() {
            translate([-clamp_len_x/2, -clamp_width/2, 0])
                cube([clamp_len_x, clamp_width, clamp_half_h]);
            if (!is_cap)
                for (x = [-clamp_screw_offset, clamp_screw_offset])
                    translate([x, 0, 0]) cylinder(h = clamp_half_h + 3, d = clamp_insert_boss_dia);
        }
        translate([0, 0, is_cap ? 0 : clamp_half_h])
            rotate([90, 0, 0])
                linear_extrude(height = clamp_width + 2, center = true)
                    oval_x_z(channel_r_x, channel_r_z);
        for (x = [-clamp_screw_offset, clamp_screw_offset])
            translate([x, 0, -0.5]) cylinder(h = clamp_half_h + 4, d = clamp_insert_hole_dia);
    }
}
module rod_mock() {
    color("DimGray") translate([0, 0, clamp_half_h]) rotate([90, 0, 0])
        linear_extrude(height = clamp_width + 40, center = true) oval_x_z(rod_x/2, rod_z/2);
}

// ============================================================================
// FRAME PIECE  (unchanged frame + centered bottom tab)
// Local: pivot axis +Y through origin; frame bottom edge at z = tab_reach; frame
// extends +Z; pocket faces +X at tilt = 0.
// ============================================================================
module frame_piece() {
    off = frame_outer/2;   // frame bottom edge -> local z = tab_reach after the shift
    translate([0, 0, tab_reach])
        rotate([0, 90, 0]) translate([-off, 0, 0]) frame_body();
    // centered bottom tab: wraps the solid bottom rail, drops to the pivot axis
    difference() {
        translate([-2, -tab_th_y/2, 0])
            cube([frame_height + 4, tab_th_y, tab_reach + 3]);
        hull() for (dx = [-pivot_slot_travel/2, pivot_slot_travel/2])
            translate([dx, -tab_th_y/2 - 1, 0])
                rotate([-90, 0, 0]) cylinder(h = tab_th_y + 2, d = pivot_screw_dia);
    }
}

// ============================================================================
// MOUNT BODY  (clamp base + fork prongs at the pivot + rise struts)
// ============================================================================
module mount_body() {
    clamp_half(false);
    for (sy = [-1, 1]) {
        ry = sy * rail_y;
        // prong disc at the pivot axis
        translate([pivot_x, ry - prong_th/2, pivot_z])
            difference() {
                rotate([-90, 0, 0]) cylinder(h = prong_th, d = pivot_pad_dia);
                translate([0, -1, 0]) rotate([-90, 0, 0])
                    cylinder(h = prong_th + 2, d = sy > 0 ? pivot_insert_dia : pivot_screw_dia);
            }
        // rise strut: clamp underside -> prong (triangulated for stiffness)
        hull() {
            translate([-clamp_len_x/2 + 1, ry - prong_th/2, 0])
                cube([clamp_len_x - 2, prong_th, 0.1]);
            translate([pivot_x, ry - prong_th/2, pivot_z])
                rotate([-90, 0, 0]) cylinder(h = prong_th, d = pivot_pad_dia);
        }
    }
}

// ============================================================================
// MOCKS  (eye + lens, for aim checking only - not printed)
// ============================================================================
module lens_mock() {
    color("Black")
        translate([frame_height, 0, tab_reach + frame_outer/2])
            rotate([0, 90, 0]) cylinder(h = lens_len, d = 12);
}
module eye_mock() {
    color("Azure",   0.35) translate(eye + [10, 0, 0]) sphere(d = 24);   // eyeball
    color("SteelBlue")     translate(eye) sphere(d = 5);                 // pupil
    // sight line: pupil -> lens tip
    color("Tomato")
        hull() { translate(eye) sphere(0.4); translate(lens_tip) sphere(0.4); }
}

// ============================================================================
module place_frame() { translate([pivot_x, 0, pivot_z]) rotate([0, -tilt_angle, 0]) children(); }

if (show == "assembly") {
    color("LightSteelBlue") mount_body();
    color("Silver") translate([0, 0, clamp_gap]) rod_mock();
    color("LightSteelBlue") translate([0, 0, clamp_half_h + clamp_gap]) clamp_half(true);
    color("Khaki") place_frame() frame_piece();
    place_frame() lens_mock();
    eye_mock();
} else if (show == "exploded") {
    mount_body();
    translate([0, 0, 14]) rod_mock();
    color("LightSteelBlue") translate([0, 0, clamp_half_h + 28]) clamp_half(true);
    color("Khaki") translate([0, 0, -18]) place_frame() frame_piece();
} else if (show == "mount") {
    mount_body();
} else if (show == "cap") {
    clamp_half(true);
} else if (show == "frame") {
    frame_piece();
}
