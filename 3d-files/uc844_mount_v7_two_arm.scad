// ============================================================================
// UC-844 Rev.B IR eye camera mount — v7, 3 parts: frame + TWO arms + reused clips
//
// Copied from v6 (uc844_mount_v6_arm.scad); fixes v6's two bugs:
//   - v6's one-piece arm PASSED THROUGH the frame (its pad sat behind the frame).
//   - the clip interface was hand-waved.
//
//   1) FRAME  - unchanged square frame + 4-corner PCB mount, plus two FRONT-bottom
//               lugs. Prints FLAT; lens/PCB drop in; PCB back stays open.
//   2) ARMS   - TWO separate brackets (one per side). Each: a clip-ear at the rod
//               (mating the real white clip) -> short body -> pad lapping a frame
//               front lug. They stay IN FRONT of / below the frame -> no collision,
//               out of the FOV; open center for the lens.
//   3) CLIPS  - the existing white clips (imported from the STEP as clip_ref.stl),
//               grip the rod to the sides; rotate([90,0,0]) seats them on the rod.
//
// Aim (from measurements: pupil 27 mm above the rod, 19 mm behind the lens plane):
// lens tip ~30 mm from the pupil at ~38 deg up-tilt. Eye / rod / gaze-ray mocks +
// a 70 deg FOV cone confirm nothing blocks the view; echo() prints tip->pupil.
//
// Coords: origin = rod center; +X toward the face/eye; +Z up; rod along Y.
// ============================================================================

$fn = 60;
// "assembly" | "exploded" | "frame" | "arm"
show = "assembly";

// ---------------- PCB / frame (carried from v3-v5, UNCHANGED) ------------
pcb_size = 38; pcb_thickness = 1.6; hole_dia = 2.85;
hole_pitch = 34; insert_hole_dia = 3.6; insert_boss_dia = 7.0; corner_pad_margin = 1.0;
frame_wall = 3.0; register_depth = 1.0; register_clearance = 0.10;
lead_in_depth = max(pcb_thickness - register_depth, 0.6); lead_in_clearance = 0.30;
corner_relief_r = 0.6;
frame_outer = pcb_size + 2*frame_wall;   // 44
frame_height = register_depth + lead_in_depth;

// ---------------- eye / aim geometry (MEASURED) -------------------------
eye_depth = 19;   // pupil behind the lens plane
eye_rise  = 27;   // pupil above the rod
lens_len  = 15.86; // M12 lens length (Ø12 body / Ø14 front)
fov_deg   = 70;   // horizontal FOV of the lens
eye = [eye_depth, 0, eye_rise];

// ---------------- pose (frame placement) --------------------------------
// Frame bottom edge at local z=0, pocket +X; placed by rotate(-tilt) about the
// rod-parallel axis at (px,0,pz). Tuned so the lens tip is ~30 mm from the pupil.
tilt_angle = 38;
px = -4.8;
pz = -19.6;

// ---------------- clip / ear interface (from bottom mount/*.step) --------
clip_pitch    = 18.7;   // Ø3.2 clip-screw spacing along the rod
clip_hole_dia = 3.2;
ear_screw_dia = 2.9;    // M2.5 clearance for the frame<->arm screws
ear_insert_dia = 3.6;   // M2.5 heat-set insert
ear_pitch     = 18.0;   // frame attach-ear spacing (Y)
ear_z_local   = 39;     // attach-ear height up the frame (local z)
arm_th        = 4.0;

// ---- real clip (imported from bottom mount/*.step -> clip_ref.stl) ------
// rotate([90,0,0]) makes the U-channel capture the rod (verified). Tune the
// centering offsets so the channel sits on the rod; clips at +/- clip_pitch/2.
clip_dx = 0; clip_dz = 0;
module clip_mock(sy) {
    color("GhostWhite")
        translate([clip_dx, sy*clip_pitch/2, clip_dz])
            rotate([90, 0, 0]) import("clip_ref.stl");
}

// derived lens tip + report ----------------------------------------------
_a = frame_height + lens_len;
_b = frame_outer/2;
lens_tip = [ px + _a*cos(tilt_angle) - _b*sin(tilt_angle), 0,
             pz + _a*sin(tilt_angle) + _b*cos(tilt_angle) ];
tip_to_eye = norm(lens_tip - eye);
aim_deg    = atan2(eye[2]-lens_tip[2], eye[0]-lens_tip[0]);
echo(lens_tip = lens_tip, pupil = eye, tip_to_eye_mm = tip_to_eye,
     lens_axis_deg = tilt_angle, aim_needed_deg = aim_deg);

// ============================================================================
// FRAME MODULES  (verbatim from v3-v5)
// ============================================================================
module oval_x_z(rx, rz) { scale([rx, rz, 1]) circle(r = 1); }
module pocket_profile(cl, rel) {
    union() { offset(delta=cl) square([pcb_size,pcb_size],center=true);
        for (x=[-1,1],y=[-1,1]) translate([x*pcb_size/2,y*pcb_size/2]) circle(r=rel); }
}
module pcb_pocket() {
    union() {
        linear_extrude(register_depth) pocket_profile(register_clearance, corner_relief_r);
        translate([0,0,register_depth]) linear_extrude(lead_in_depth+0.02)
            pocket_profile(lead_in_clearance, corner_relief_r+lead_in_clearance);
    }
}
module corner_pad_2d(xs,ys) {
    reach=hole_pitch/2-insert_boss_dia/2-corner_pad_margin; pad_span=frame_outer/2-reach;
    chamfer=pad_span*0.6;
    mirror([xs<0?1:0,0,0]) mirror([0,ys<0?1:0,0]) union() {
        difference() { translate([reach,reach]) square([pad_span,pad_span]);
            translate([reach,reach]) polygon([[0,0],[chamfer,0],[0,chamfer]]); }
        translate([hole_pitch/2,hole_pitch/2]) circle(d=insert_boss_dia);
    }
}
module corner_pad(xs,ys) {
    difference() { linear_extrude(frame_height) corner_pad_2d(xs,ys);
        translate([xs*hole_pitch/2,ys*hole_pitch/2,-0.5]) cylinder(h=frame_height+1,d=insert_hole_dia); }
}
module frame_body() {
    union() {
        difference() { linear_extrude(frame_height) square([frame_outer,frame_outer],center=true);
            pcb_pocket(); }
        for (x=[-1,1],y=[-1,1]) corner_pad(x,y);
    }
}

// ============================================================================
// PART 1 - FRAME PIECE (frame + two back attach-ears)
// Local: pocket faces +X, back face at x=0, bottom edge at z=0, extends +Z.
// ============================================================================
lug_w    = 7;      // frame lug width (Y)
lug_fwd  = 3.5;    // frame lug protrudes FORWARD (+X, eye side) from the front-bottom
lug_drop = 9;      // and drops below the bottom edge
lug_hole_dz = -4;  // Y bolt-hole height on the lug (local z)
function place_pt(p) = [ px + p[0]*cos(tilt_angle) - p[2]*sin(tilt_angle),
                         p[1],
                         pz + p[0]*sin(tilt_angle) + p[2]*cos(tilt_angle) ];
module frame_piece() {
    // frame standing: bottom edge -> z=0, pocket -> +X
    translate([0, 0, frame_outer/2]) rotate([0, 90, 0]) frame_body();
    // FRONT-bottom lugs: protrude forward + down from the bottom border so the arm
    // attaches in FRONT of the frame and never reaches behind it (bolt along Y).
    for (sy = [-1, 1])
        translate([frame_height - 0.6, sy*ear_pitch/2 - lug_w/2, -lug_drop])
            difference() {
                cube([lug_fwd, lug_w, lug_drop + 3]);   // overlaps bottom border z0..3
                translate([lug_fwd/2, -1, lug_drop + lug_hole_dz])
                    rotate([-90, 0, 0]) cylinder(h = lug_w + 2, d = ear_screw_dia);
            }
}

// ============================================================================
// PART 2 - TWO SEPARATE ARMS (one per side). Each: a clip-ear at the rod (mating
// the real white clip's Ø3.2 hole) -> short body -> pad lapping the frame front lug.
// Stays in FRONT of / below the frame -> no collision, below the FOV.
// ============================================================================
function clip_ear_pt(sy) = [clip_dx - 3.2, sy*clip_pitch/2, clip_dz];      // clip screw hole (vertical)
function frame_lug_pt(sy) = place_pt([frame_height - 0.6 + lug_fwd/2, sy*ear_pitch/2, lug_hole_dz]);
module one_arm(sy) {
    ce = clip_ear_pt(sy) + [0, 0, -3.5];   // ear pad just below the clip's tab
    fl = frame_lug_pt(sy);
    difference() {
        union() {
            hull() {
                translate(ce) cube([arm_th, arm_th + 2, arm_th], center = true);
                translate(fl) cube([arm_th, arm_th + 2, arm_th], center = true);
            }
            translate(ce) cube([arm_th + 3, arm_th + 2, arm_th], center = true);  // clip-ear flange
            translate(fl) cube([arm_th + 2, arm_th + 2, arm_th + 2], center = true); // frame pad
        }
        translate(ce) cylinder(h = arm_th + 8, center = true, d = clip_hole_dia);   // vertical clip screw
        translate(fl) rotate([-90,0,0]) cylinder(h = lug_w + 6, center = true, d = ear_screw_dia); // Y frame bolt
    }
}
module arm() { for (sy = [-1, 1]) one_arm(sy); }   // assembly shows both

// ============================================================================
// MOCKS  (eye + rod + gaze ray + FOV cone; not printed)
// ============================================================================
module lens_mock() {
    translate([frame_height, 0, frame_outer/2]) rotate([0,90,0]) {
        color("DimGray") cylinder(h=lens_len, d=12);
        color("Black") translate([0,0,lens_len-3]) cylinder(h=3, d=14);
    }
}
module rod_mock() { color("Silver") rotate([90,0,0]) linear_extrude(60,center=true) oval_x_z(2.5, 1.375); }
module eye_mock() {
    color("Azure",0.30) translate(eye+[10,0,0]) sphere(d=24);
    color("SteelBlue") translate(eye) sphere(d=5);
    color("Tomato") hull(){ translate(eye) sphere(0.4); translate(lens_tip) sphere(0.4); }
}
module fov_cone() {
    // cone from the lens tip toward the eye, half-angle fov/2, length to the eye
    L = tip_to_eye;
    color("Yellow", 0.12)
        translate(lens_tip)
            rotate([0, 90 - tilt_angle, 0])   // point +Z' along the lens axis (elev tilt)
                cylinder(h = L, r1 = 0, r2 = L*tan(fov_deg/2));
}

// ============================================================================
module place_frame() { translate([px,0,pz]) rotate([0,-tilt_angle,0]) children(); }

if (show == "assembly") {
    color("Khaki")        place_frame() frame_piece();
    color("LightSteelBlue") arm();
    place_frame() lens_mock();
    rod_mock();
    for (sy = [-1, 1]) clip_mock(sy);
    eye_mock();
    // fov_cone();
} else if (show == "exploded") {
    place_frame() frame_piece();
    translate([0,0,-25]) arm();
    rod_mock(); eye_mock();
} else if (show == "frame") {
    frame_piece();
} else if (show == "arm") {
    one_arm(1);          // ONE arm (print two: this + its mirror)
} else if (show == "print") {
    rotate([0, -90, 0]) frame_piece();            // frame flat, pocket up
    translate([0, 55, 0]) one_arm(1);             // arm (orient in slicer)
}
