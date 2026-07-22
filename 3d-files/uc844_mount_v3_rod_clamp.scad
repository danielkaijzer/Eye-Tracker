// ============================================================================
// UC-844 Rev.B IR eye camera mount — v3, integrated rod clamp (OpenSCAD)
//
// Same frame + dipped spine as uc844_mount_v2_dipped_spine.scad (see that
// file's header for the shared background and the dip_angle reasoning), with
// ONE change at the glasses end:
//
//   v2 attached to the glasses with two snap-fit "keyhole" clips that were
//   reverse-engineered from the old mount's STEP file, assuming the glasses
//   present a thin WIRE. The real glasses present a rigid OVAL ROD (measured
//   ~5mm x 4mm with a ruler), so a snap clip is the wrong mechanism. The
//   separate two-piece peg adapter (glasses_rod_adapter.scad) tried to bridge
//   that gap but added a confusing extra interface (pegs into keyhole slots
//   with a loose rotational degree of freedom).
//
//   v3 deletes that whole bridge. The clamp that grips the rod is now part of
//   the mount itself: the near end of the spine IS the lower half of a screw
//   clamp. A separate small CAP (printed alongside) is the upper half. You lay
//   the glasses rod into the channel, set the cap on top, and drive two screws
//   down through the cap into heat-set inserts in the base - the same insert
//   scheme used at the four camera-mounting corners. No pegs, no keyholes, no
//   extra part between the glasses and the camera.
//
//   Trade-off: this means REPRINTING the mount (the v2 print used clips). It's
//   still one solid printed body plus one tiny flat cap.
//
// Set `show` below to "assembly" to see the mount + rod mockup + cap together
// (for checking the parts mate and the screws line up), or "print" to get the
// two printable bodies laid out separately.
//
// Confidence levels:
//   HIGH   - frame/tilt geometry carried over unchanged from v1/v2; rod
//            cross-section (5 x 4mm oval, measured directly with a ruler)
//   MEDIUM - pcb_size, pcb_thickness, hole_dia (from photos/ruler)
//   LOW    - dip_angle (approximate; the rod grip point moved slightly vs the
//            old clip grip point, so re-check lens height on test-fit),
//            clamp screw size/offset - tune-on-fit. NOTE the clamp is
//            intentionally forgiving to rod-size error (see the rod clamp
//            section), so the 5x4mm rod measurement does NOT need to be exact.
// ============================================================================

$fn = 60;

// "assembly" (mount+rod+cap seated) | "exploded" (cap lifted, rod dropping in)
// | "print" (both bodies apart, one STL) | "mount" (just the mount body)
// | "cap" (just the clamp cap). Use "mount" and "cap" to export each piece to
// its own STL so Cura imports them as independent objects.
show = "assembly";

// ---------------- PCB / camera board (MEDIUM confidence) ----------------
pcb_size        = 38;    // UC-844 Rev.B board is 38x38mm square
pcb_thickness   = 1.6;
hole_dia        = 2.85;

// ---------------- corner mounting holes (HIGH confidence, measured) -----
hole_pitch      = 34;    // outer/34mm corner pattern
insert_hole_dia = 3.6;   // pilot hole for an M2.5 heat-set insert
insert_boss_dia = 7.0;
corner_pad_margin = 1.0;

// ---------------- open picture-frame ------------------------------------
frame_wall         = 3.0;
register_depth     = 1.0;
register_clearance = 0.10;
lead_in_depth      = max(pcb_thickness - register_depth, 0.6);
lead_in_clearance  = 0.30;
corner_relief_r    = 0.6;

frame_outer  = pcb_size + 2*frame_wall;
frame_height = register_depth + lead_in_depth;

// ---------------- glasses rod (HIGH confidence - measured with a ruler) --
// The bottom rim of the glasses is a rigid oval-section rod. It runs left-to-
// right across the face (the Y axis here); the mount hangs off it toward the
// eye (+X). "Top-down" you see its fore-aft width (X); "front-on" you see its
// up-down thickness (Z).
rod_x = 5.0;   // fore-aft width, seen looking straight down
rod_z = 4.0;   // up-down thickness, seen looking at the glasses front-on

// ---------------- rod clamp (integrated lower half + separate cap) ------
// The clamp splits horizontally: base = lower half (fused to the spine/mount),
// cap = upper half (separate print). Two screws sit fore and aft of the rod
// and pull the cap down onto the base. Inserts live in the base; cap holes are
// clearance.
//
// FORGIVING BY DESIGN: the bore is cut deliberately SHALLOWER (in the split /
// Z direction) than the measured rod, so the rod always stands proud of the
// parting line and the two halves pinch the ROD - they never bottom out flat
// against each other first. So a measurement error just changes the leftover
// gap between the halves, not whether it grips: lay the rod in, set the cap
// on, tighten the two screws, done. It tolerates the rod being up to ~1mm
// THINNER than measured (down to 2*channel_r_z, where the gap closes to zero)
// and several mm THICKER (limited only by screw length). Width (X) gets a
// little clearance so it always drops in; the real grip is the top/bottom Z
// pinch plus friction.
clamp_bore_x_clear = 0.3;   // per-side width slop - rod always drops into the groove
clamp_bore_squeeze = 0.5;   // each half's groove is this much shallower than the rod's half-thickness -> rod stands proud by this per side and gets pinched
clamp_wall         = 2.5;   // material below the groove (also makes half height == spine_thick)
clamp_insert_hole_dia   = 3.6;    // base: heat-set insert pilot (cap: clearance)
clamp_insert_boss_dia   = 7.0;
clamp_screw_offset      = 7.0;    // X distance from rod center to each screw

channel_r_x = rod_x/2 + clamp_bore_x_clear;    // bore half-width - hugs the rod sides with slight slop
channel_r_z = rod_z/2 - clamp_bore_squeeze;    // bore half-height - UNDERSIZED so the rod stands proud and gets pinched

clamp_half_h = channel_r_z + clamp_wall;                       // height of each half
clamp_gap    = max(rod_z - 2*channel_r_z, 0);                  // leftover gap between the halves at the nominal rod size (the clamp's "give")
clamp_len_x  = 2*clamp_screw_offset + clamp_insert_boss_dia + 2; // fore boss .. aft boss + margin

// ---------------- spine / camera tilt -----------------------------------
tilt_angle   = 60;   // matches the old mount
spine_len_1  = 12;   // clamp-side segment, in-line
spine_len_2  = 12;   // frame-side segment, dipped
dip_angle    = 29;   // droop before the frame (see v2 header; re-tune on fit)
spine_width  = 22.4; // grip length along the rod + spine width (was clip_spacing+clip_wid+2 in v2)
spine_thick  = 4;

spine_z  = clamp_half_h / 2;      // spine centerline height (mid the base block)
face_x   = clamp_len_x / 2;       // +X face of the clamp, where the spine emerges

// ============================================================================
// SHARED FRAME MODULES (unchanged from v1/v2)
// ============================================================================

module oval_x_z(rx, rz) {
    scale([rx, rz, 1]) circle(r = 1);
}

module pocket_profile(clearance, relief) {
    union() {
        offset(delta = clearance) square([pcb_size, pcb_size], center = true);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * pcb_size/2, y * pcb_size/2])
                circle(r = relief);
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
            translate([reach, reach])
                square([pad_span, pad_span]);
            translate([reach, reach])
                polygon([[0, 0], [chamfer, 0], [0, chamfer]]);
        }
        translate([hole_pitch/2, hole_pitch/2])
            circle(d = insert_boss_dia);
    }
}

module corner_pad(xs, ys) {
    difference() {
        linear_extrude(height = frame_height)
            corner_pad_2d(xs, ys);
        translate([xs * hole_pitch/2, ys * hole_pitch/2, -0.5])
            cylinder(h = frame_height + 1, d = insert_hole_dia);
    }
}

tab_len   = 4;
tab_width = spine_width;

module frame_tab() {
    translate([-frame_outer/2 - tab_len, -tab_width/2, 0])
        cube([tab_len, tab_width, frame_height]);
}

module frame_body() {
    union() {
        difference() {
            linear_extrude(height = frame_height)
                square([frame_outer, frame_outer], center = true);
            pcb_pocket();
        }
        for (x = [-1, 1], y = [-1, 1])
            corner_pad(x, y);
        frame_tab();
    }
}

// ============================================================================
// ROD CLAMP MODULES (new in v3)
// ============================================================================

// One half of the clamp block. is_cap=false -> base (lower half, with proud
// insert bosses and the channel cut into its TOP face). is_cap=true -> cap
// (upper half, channel cut into its BOTTOM face, plain clearance holes).
module clamp_half(is_cap) {
    difference() {
        union() {
            translate([-clamp_len_x/2, -spine_width/2, 0])
                cube([clamp_len_x, spine_width, clamp_half_h]);
            if (!is_cap)
                for (x = [-clamp_screw_offset, clamp_screw_offset])
                    translate([x, 0, 0])
                        cylinder(h = clamp_half_h + 3, d = clamp_insert_boss_dia);
        }
        // rod channel: full oval centered on the parting face, so only the
        // half inside the block is actually removed -> a half-oval groove that
        // meets the other half to wrap the rod. Runs the full width along Y.
        translate([0, 0, is_cap ? 0 : clamp_half_h])
            rotate([90, 0, 0])
                linear_extrude(height = spine_width + 2, center = true)
                    oval_x_z(channel_r_x, channel_r_z);
        // screw holes
        for (x = [-clamp_screw_offset, clamp_screw_offset])
            translate([x, 0, -0.5])
                cylinder(h = clamp_half_h + 4, d = clamp_insert_hole_dia);
    }
}

// The rod itself (mockup, for fit-checking only - not printed). Sits in the
// channel at the parting plane and runs past both clamp faces to show it's a
// continuous rim.
module rod_mock() {
    color("DimGray")
        translate([0, 0, clamp_half_h])
            rotate([90, 0, 0])
                linear_extrude(height = spine_width + 40, center = true)
                    oval_x_z(rod_x/2, rod_z/2);
}

// ============================================================================
// MOUNT (base clamp half + dipped spine + tilted frame)
// ============================================================================

module mount_body() {
    overlap = 2;

    // lower half of the rod clamp, fused to everything downstream
    clamp_half(false);

    // segment 1: horizontal, straight off the clamp's +X face
    translate([face_x - overlap, -spine_width/2, spine_z - spine_thick/2])
        cube([spine_len_1 + overlap, spine_width, spine_thick]);

    seg1_end = [face_x + spine_len_1, 0, spine_z];

    // Joint 1 (dip bend): clean hull between the two true mating faces.
    hull() {
        translate(seg1_end + [-0.01, -spine_width/2, -spine_thick/2])
            cube([0.01, spine_width, spine_thick]);
        translate(seg1_end)
            rotate([0, dip_angle, 0])
                translate([0, -spine_width/2, -spine_thick/2])
                    cube([0.01, spine_width, spine_thick]);
    }

    // segment 2 + frame, dipped down together
    translate(seg1_end)
        rotate([0, dip_angle, 0]) {
            translate([0, -spine_width/2, -spine_thick/2])
                cube([spine_len_2, spine_width, spine_thick]);

            // Joint 2 (tilt bend): hull seg2 end face against the frame tab tip.
            hull() {
                translate([spine_len_2 - 0.01, -spine_width/2, -spine_thick/2])
                    cube([0.01, spine_width, spine_thick]);
                translate([spine_len_2, 0, 0])
                    rotate([0, -tilt_angle, 0])
                        translate([frame_outer/2 + tab_len/2, 0, -frame_height/2])
                            translate([-frame_outer/2 - tab_len, -tab_width/2, 0])
                                cube([0.01, tab_width, frame_height]);
            }

            // tilt negative -> pocket opening faces back toward the eye
            translate([spine_len_2, 0, 0])
                rotate([0, -tilt_angle, 0])
                    translate([frame_outer/2 + tab_len/2, 0, -frame_height/2])
                        frame_body();
        }
}

// ============================================================================
// OUTPUT
// ============================================================================

if (show == "assembly") {
    mount_body();
    // rod centered between the two groove faces; because the bore is
    // undersized the halves stay clamp_gap apart, pinching the rod.
    translate([0, 0, clamp_gap/2]) rod_mock();
    color("LightSteelBlue")
        translate([0, 0, clamp_half_h + clamp_gap])
            clamp_half(true);
} else if (show == "exploded") {
    // Same as assembly but the cap is lifted straight up and the rod floats
    // above the OPEN groove in the base - showing that the rod drops in from
    // the top (the only way to capture a continuous, no-free-end rim) and the
    // cap then bolts down over it. No threading through a closed hole.
    ex = 16;   // explode gap
    mount_body();
    translate([0, 0, ex]) rod_mock();
    color("LightSteelBlue")
        translate([0, 0, clamp_half_h + 2*ex])
            clamp_half(true);
} else if (show == "mount") {
    // just the mount body - export this to its own STL
    mount_body();
} else if (show == "cap") {
    // just the clamp cap - export this to its own STL
    clamp_half(true);
} else {
    // print layout: the two printable bodies, apart, in ONE STL. Handy for a
    // quick look, but Cura imports it as a single object - use "mount" and
    // "cap" above to get two independent STLs instead. (The mount still prints
    // rotated ~31 deg about Y to lay the frame flat, exactly as v2 did - do
    // that in the slicer, same as before.)
    mount_body();
    translate([0, spine_width + 15, 0])
        clamp_half(true);
}
