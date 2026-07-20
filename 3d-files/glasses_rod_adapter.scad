// ============================================================================
// Glasses rod adapter - v1 (parametric, OpenSCAD)
//
// Bridges the already-printed uc844_mount_v2_dipped_spine.scad to the real
// attachment point on the headset: a rigid oval-cross-section rod (measured
// directly with a ruler), not the thin wire the mount's keyhole clips were
// originally reverse-engineered for from the old STEP file. Rather than
// reprint the validated mount, this is a small separate piece:
//   - one end has two round pegs that plug into the mount's existing
//     keyhole slots (same spacing as the printed clips)
//   - the other end is a 2-piece screw clamp that grips the rod, mirroring
//     the clamp mechanism visible in photos of the old mount, sized to the
//     actual rod
//
// PRINT BOTH HALVES (base and cap) - two separate pieces, not fused. Sandwich
// the rod between them and draw the two screws down to clamp it; the pegs
// (on the base) then push into the mount's two keyhole slots.
//
// This doesn't change the lens-height calibration already solved for the
// main mount (dip_angle=29) - that was based on WHERE the old mount attached
// to the rod, not the attachment mechanism, so as long as this clamp grips
// the rod at roughly the same spot the old one did, the height should carry
// over.
//
// Confidence levels:
//   HIGH   - rod cross-section (5mm x 4mm oval), measured directly with a
//            ruler on the actual glasses
//   LOW    - peg diameter/length, clamp screw size, clamp proportions -
//            first-pass guesses sized to fit the mount's existing keyhole
//            slot and a plausible small screw; expect to tune after a
//            test-fit, same as every other tolerance in this project
// ============================================================================

$fn = 60;

// ---------------- the mount's existing clip geometry (must match!) ------
// These MUST match uc844_mount_v2_dipped_spine.scad - the pegs below have to
// plug into that already-printed part's keyhole slots.
clip_spacing     = 15;    // distance between the two clips on the printed mount
clip_mouth       = 3.1;
clip_neck        = 2.1;
clip_pocket      = 3.4;

// ---------------- rod (HIGH confidence - measured with a ruler) ---------
rod_dia_a = 5.0;   // oval rod, top-down measurement - this is the clamp's WIDTH axis
rod_dia_b = 4.0;   // oval rod, front-on measurement - this is the axis the clamp splits along

// ---------------- peg: plugs into the mount's keyhole pocket (LOW) ------
// Sized between the slot's 2.1mm pinch and 3.4mm pocket width, like the
// wire the slot was originally designed around - tune on test-fit.
peg_dia    = 2.8;
peg_length = 5.0;   // reaches past the pinch into the pocket

// ---------------- clamp (LOW confidence - no reference for the old screw size)
clamp_insert_hole_dia = 3.6;   // pilot hole for an M2.5 heat-set insert - match your insert's spec
clamp_insert_boss_dia = 7.0;
clamp_wall            = 2.0;   // material around the rod channel, and above/below the screw bosses
clamp_length          = clip_spacing + 10;      // spans both peg positions with margin
clamp_screw_offset    = clamp_length/2 - 4;      // screw positions from center, along the clamp's length

channel_r_a = rod_dia_a/2;   // channel half-width (unsplit axis)
channel_r_b = rod_dia_b/2;   // channel half-depth (split axis)

half_height  = channel_r_b + clamp_wall;
clamp_width  = rod_dia_a + 2*clamp_insert_boss_dia;

// ============================================================================
// MODULES
// ============================================================================

module oval_profile(ra, rb) {
    scale([ra, rb, 1])
        circle(r = 1);
}

// One half of the clamp: a block with a half-oval notch in its top face
// (running the full length) and 2 vertical screw holes. is_cap flips the
// notch to the bottom face and uses clearance holes instead of insert bosses.
module clamp_half(is_cap) {
    difference() {
        union() {
            translate([-clamp_width/2, -clamp_length/2, 0])
                cube([clamp_width, clamp_length, half_height]);
            // insert bosses on the base only, proud of the block so there's
            // real depth for the heat-set insert
            if (!is_cap)
                for (y = [-clamp_screw_offset, clamp_screw_offset])
                    translate([0, y, 0])
                        cylinder(h = half_height + 3, d = clamp_insert_boss_dia);
        }
        // rod channel: half oval, full length, sitting exactly at the
        // parting face so the two halves' notches meet to form a full oval
        translate([0, 0, is_cap ? 0 : half_height])
            rotate([90, 0, 0])
                linear_extrude(height = clamp_length + 1, center = true)
                    oval_profile(channel_r_a, channel_r_b);
        // screw holes: through-clearance in the cap, pilot hole in the base
        for (y = [-clamp_screw_offset, clamp_screw_offset])
            translate([0, y, -0.5])
                cylinder(h = half_height + 4, d = clamp_insert_hole_dia);
    }
}

module peg(y) {
    translate([clamp_width/2, y, half_height/2])
        rotate([0, 90, 0])
            cylinder(h = peg_length, d = peg_dia);
}

module clamp_base() {
    union() {
        clamp_half(false);
        for (y = [-clip_spacing/2, clip_spacing/2])
            peg(y);
    }
}

module clamp_cap() {
    clamp_half(true);
}

// lay both pieces out separately, ready to print side by side
translate([-clamp_width/2 - 5, 0, 0])
    clamp_base();

translate([clamp_width/2 + 5, 0, 0])
    clamp_cap();
