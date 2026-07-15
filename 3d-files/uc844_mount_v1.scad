// ============================================================================
// UC-844 Rev.B IR eye camera mount — v1 (parametric, OpenSCAD)
//
// Replaces the old internal-eye-cam mount for a new camera (ArduCam UC-844
// Rev.B, 38x38mm PCB). Two things were deliberately carried over from the old
// mount's real CAD file (3d-files/bottom mount/v1 various internal eye cams.step):
//   - the 60 degree camera tilt (also confirmed in 3d-files/NOTES.md)
//   - the glasses-attachment clip geometry (measured directly off that STEP
//     file's two clip solids)
// Everything else here is new, sized for the UC-844 board.
//
// Frame shape: an open picture-frame (thin rim hugging the PCB edges, hollow
// in the middle - no solid back) with a diagonal braced pad at each corner
// carrying the mounting screw. The camera is tilted so the pocket opening
// faces back toward the glasses clips (inward, toward the eye) rather than
// away from them.
//
// Confidence levels on the numbers below, so you know what to double check
// before printing:
//   HIGH   - measured directly from the STEP file geometry (tilt angle, clip
//            mouth/neck/pocket widths, clip block size, clip bolt hole dia)
//   MEDIUM - measured from photos/ruler on the actual board (pcb_size,
//            pcb_thickness, hole_dia) - re-check with calipers if you have them
//   LOW    - designer's-choice defaults with no hard measurement behind them
//            yet (hole_pitch, clip_spacing, spine_length, tolerances) - these
//            are the ones you'll most likely tune after a first test fit
// ============================================================================

$fn = 60;

// ---------------- PCB / camera board (MEDIUM confidence) ----------------
pcb_size        = 38;    // UC-844 Rev.B board is 38x38mm square
pcb_thickness   = 1.6;   // measured ~1/16" at the bare edges
hole_dia        = 3.0;   // corner hole diameter, measured "just under 1/8in"

// ---------------- corner mounting holes (LOW confidence - verify!) ------
// Board silkscreen mentions hole patterns compatible with 34x34 and 28x28mm
// pitches. Only 4 real plated corner holes exist (confirmed from photos —
// the other 6 small holes are via-stitching on the ground plane, not for
// screws). Measure your actual board before printing; swap the value below.
hole_pitch      = 34;
insert_hole_dia = 3.6;   // pilot hole for an M2.5 heat-set insert - match your insert's spec
insert_boss_dia = 7.0;   // outer diameter of the printed boss around each insert
corner_pad_margin = 1.0; // extra solid margin kept around each boss inside its corner pad

// ---------------- open picture-frame (addresses PCB slop from the old mount) --
// Just a rim hugging the 4 edges + 4 corner pads - no solid back, so airflow
// behind the PCB (per NOTES.md) is inherent rather than needing perforation.
frame_wall         = 3.0;   // rim material around the PCB on the straight edges
register_depth     = 1.0;   // height of the snug "registration" zone, measured from the bottom
register_clearance = 0.10;  // per-side clearance in the registration zone - tight, kills slide/rotation
lead_in_depth      = max(pcb_thickness - register_depth, 0.6); // looser zone above it, eases drop-in
lead_in_clearance  = 0.30;  // per-side clearance in the lead-in zone
corner_relief_r    = 0.6;   // extra rounding at the 4 inside pocket corners (clears PCB corners/printed fillets)

frame_outer  = pcb_size + 2*frame_wall;
frame_height = register_depth + lead_in_depth;

// ---------------- glasses attachment clip --------------------------------
// Measured from 3d-files/bottom mount/v1 various internal eye cams.step,
// solids 5 & 6 (the two small clip blocks, bbox 11.60 x 5.40 x 6.00mm).
// It's a snap-fit "keyhole" slot, not a screw-cinched clamp: the wire pops
// through a narrow pinch and is captured in a wider pocket behind it. The
// bolt hole is unrelated to wire retention - it just bolts the clip to the
// spine (matches the screw+nut visible in the reference photos).
clip_len   = 11.6;  // HIGH - overall block length (insertion axis)
clip_wid   = 5.4;   // HIGH - overall block width
clip_height= 6.0;   // HIGH - overall block height (extrusion depth in the STEP file)
clip_mouth = 3.1;   // HIGH - entry opening width, where the wire is pushed in
clip_neck  = 2.1;   // HIGH - pinch/retention throat width (undersized on purpose - flexes to admit the wire, springs back to hold it)
clip_pocket= 3.4;   // HIGH - wire capture pocket width, behind the pinch
clip_bolt_dia = 3.2; // HIGH - hole that bolts the clip to the spine
// Segment lengths along the insertion axis are a reconstruction (the STEP
// vertex dump confirmed the widths above and their order, but not each
// segment's exact length) - tune these against your own glasses wire if the
// fit feels off.
clip_mouth_len  = 3.0;
clip_neck_len   = 3.0;
clip_back_wall  = 1.0;   // solid material left behind the pocket, at the closed end
clip_pocket_len = clip_len - clip_mouth_len - clip_neck_len - clip_back_wall;
clip_spacing    = 15;    // LOW - distance between the two clips; verify against your glasses' rim spacing

// ---------------- spine / camera tilt ------------------------------------
tilt_angle   = 60;   // degrees - matches the old mount (NOTES.md + STEP file DIRECTION vectors)
spine_length = 20;   // frame-to-clip distance - tune during test-fit, no hard reference for this
spine_width  = clip_spacing + clip_wid + 2;   // spans both clips fully (plus margin) so the spine fuses into one solid part, not two loose ones
spine_thick  = 4;

// ============================================================================
// MODULES
// ============================================================================

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

// One corner's 2D footprint, built for the +x/+y corner then mirrored into
// place: a square pad reaching in from the frame's outer corner, chamfered
// diagonally on its inner corner (the brace shape from the sketch), plus an
// explicit boss disk so the insert hole always has full material around it
// regardless of the chamfer.
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

// Solid tab sticking out from the frame's near edge, toward the spine. The
// frame's own middle is hollow (open frame), so without this the spine would
// only ever meet the frame at its hollow center - this tab is what the spine
// actually fuses into.
tab_len   = 4;
tab_width = spine_width;

module frame_tab() {
    translate([-frame_outer/2 - tab_len, -tab_width/2, 0])
        cube([tab_len, tab_width, frame_height]);
}

module frame_body() {
    union() {
        // open rim: hugs the PCB edges, hollow in the middle, no solid back
        difference() {
            linear_extrude(height = frame_height)
                square([frame_outer, frame_outer], center = true);
            pcb_pocket();
        }
        // 4 diagonal-braced corner pads carrying the mounting screws
        for (x = [-1, 1], y = [-1, 1])
            corner_pad(x, y);
        frame_tab();
    }
}

module clip_slot_profile() {
    union() {
        translate([-1, -clip_mouth/2])
            square([clip_mouth_len + 1, clip_mouth]);
        translate([clip_mouth_len - 0.5, -clip_neck/2])
            square([clip_neck_len + 0.5, clip_neck]);
        translate([clip_mouth_len + clip_neck_len - 0.5, -clip_pocket/2])
            square([clip_pocket_len + 0.5, clip_pocket]);
    }
}

module glasses_clip() {
    bolt_x = clip_mouth_len - 0.4; // near the mouth end, matching the STEP measurement
    difference() {
        translate([0, -clip_wid/2, 0])
            cube([clip_len, clip_wid, clip_height]);
        translate([0, 0, -0.5])
            linear_extrude(height = clip_height + 1)
                clip_slot_profile();
        translate([bolt_x, -clip_wid/2 - 0.5, clip_height/2])
            rotate([-90, 0, 0])
                cylinder(h = clip_wid + 1, d = clip_bolt_dia);
    }
}

module mount_assembly() {
    // extra overlap so the spine solidly fuses into the clips and the frame
    // instead of merely touching them at a zero-width boundary
    overlap = 2;

    for (dy = [-clip_spacing/2, clip_spacing/2])
        translate([0, dy, 0])
            glasses_clip();

    translate([clip_len - overlap, -spine_width/2, (clip_height - spine_thick)/2])
        cube([spine_length + 2 * overlap, spine_width, spine_thick]);

    // tilt is negative so the pocket opening (where the PCB/lens faces) tilts
    // back toward the spine and glasses clips - i.e. inward toward the eye -
    // instead of outward away from the face.
    // Rotating/translating is anchored at the tab's midpoint (not the frame's
    // own hollow center) so the tab lands solidly inside the spine regardless
    // of tilt angle/sign.
    translate([clip_len + spine_length, 0, clip_height/2])
        rotate([0, -tilt_angle, 0])
            translate([frame_outer/2 + tab_len/2, 0, -frame_height/2])
                frame_body();
}

mount_assembly();
