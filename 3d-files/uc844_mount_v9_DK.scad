// ============================================================================
// UC-844 Rev.B IR eye camera mount — v8
//
//   1) FRAME  - square frame + 4-corner PCB mount, lens/PCB drop in; PCB back stays open.

//   2) ARMS   - TWO separate brackets (one per side). Each: a clip-ear at the rod
//               (mating the real white clip) -> short body -> TODO: Find a way to attach to frame.
//               Stay out of the FOV; open center for the lens.

//   3) CLIPS  - the existing white clips (imported from the STEP as clip_ref.stl),
//               grip the rod to the sides; rotate([90,0,0]) seats them on the rod.
//
// Aim (from measurement estimates: pupil 27 mm above the rod, 19 mm behind the lens plane):
// lens tip ~30 mm from the pupil at ~38 deg up-tilt. Eye / rod / gaze-ray mocks +
// a 70 deg FOV cone confirm nothing blocks the view; echo() prints tip->pupil.
//
// Coords: origin = rod center; +X toward the face/eye; +Z up; rod along Y.
// ============================================================================

$fn = 60;
// "assembly" | "exploded" | "frame" | "arm" | "print"
show = "print";

// ---------------- PCB / frame ------------
pcb_size = 38; pcb_thickness = 1.6; hole_dia = 2.85;
hole_pitch = 34; insert_hole_dia = 3.6; insert_boss_dia = 7.0; corner_pad_margin = 1.0;
frame_wall = 3.0; register_depth = 1.0; register_clearance = 0.10;
lead_in_depth = max(pcb_thickness - register_depth, 0.6); lead_in_clearance = 0.30;
corner_relief_r = 0.6;
frame_outer = pcb_size + 2*frame_wall;   // 44
frame_height = register_depth + lead_in_depth; // == pcb_thickness, 1.6

// ---------------- eye / aim geometry (MEASURED) -------------------------
eye_depth = 19;   // pupil behind the lens plane (estimate)
eye_rise  = 27;   // pupil above the rod (estimate)
lens_len  = 15.86; // M12 lens length (Ø12 body / Ø14 front)
fov_deg   = 70;   // horizontal FOV of the lens
eye = [eye_depth, 0, eye_rise];

// ---------------- pose (frame placement) --------------------------------
// Frame bottom edge at local z=0, pocket +X; placed by rotate(-tilt) about the
// rod-parallel axis at (px,0,pz). Tuned so the lens tip is ~30 mm from the pupil.
tilt_angle = 38.8; // 38
px = -5; // -4.8;
pz = -20.5; // -19.6;

// lens seating face, frame-local X. Rear-mount: PCB sits behind the frame.
//   front drop-in : frame_height
//   rear, PCB front face on frame back : 0
//   rear, PCB back  face on frame back : -pcb_thickness
lens_base_x = 0;

// ---------------- clip / ear interface (from bottom mount/*.step) --------
// clip_pitch    = 18.7;   // Ø3.2 clip-screw spacing along the rod
clip_pitch = 22.5; // 39
clip_hole_dia = 3.2;
ear_screw_dia = 2.9;    // M2.5 clearance for the frame<->arm screws
ear_insert_dia = 3.6;   // M2.5 heat-set insert
ear_pitch     = 18.0;   // frame attach-ear spacing (Y)
ear_z_local   = 39;     // attach-ear height up the frame (local z)
arm_th        = 3;

// derived lens tip + report ----------------------------------------------
_a = lens_base_x + lens_len;
_b = frame_outer/2;
lens_tip = [ px + _a*cos(tilt_angle) - _b*sin(tilt_angle), 0,
             pz + _a*sin(tilt_angle) + _b*cos(tilt_angle) ];
tip_to_eye = norm(lens_tip - eye);
aim_deg    = atan2(eye[2]-lens_tip[2], eye[0]-lens_tip[0]);
echo(lens_tip = lens_tip, pupil = eye, tip_to_eye_mm = tip_to_eye,
     lens_axis_deg = tilt_angle, aim_needed_deg = aim_deg);

// ============================================================================
// FRAME MODULES
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
// PART 1 - FRAME PIECE 
// Local: pocket faces +X, back face at x=0, bottom edge at z=0, extends +Z.
// ============================================================================

function place_pt(p) = [ px + p[0]*cos(tilt_angle) - p[2]*sin(tilt_angle),
                         p[1],
                         pz + p[0]*sin(tilt_angle) + p[2]*cos(tilt_angle) ];
module frame_piece() {
    // frame standing: bottom edge -> z=0, pocket -> +X
    translate([0, 0, frame_outer/2]) rotate([0, 90, 0]) frame_body();

}

// ============================================================================
// PART 2 - TWO SEPARATE ARMS (one per side). Each: a clip-ear at the rod (mating
// the real white clip's Ø3.2 hole) -> short body -> ___
// ============================================================================
function clip_ear_pt(sy) = [clip_dx - 3.2, sy*clip_pitch/2, clip_dz];      // clip screw hole (vertical)

module one_arm(sy) {
    ce = clip_ear_pt(sy);   // ear pad center at clip screw hole
    
    // Calculate local Z height on frame corresponding to clip Z height
    local_z = (ce[2] - pz) / cos(tilt_angle);

    // Project straight horizontally from ce (-X towards frame)
    frame_x = px - local_z * sin(tilt_angle) - 5;
    
    frame_y = sy * frame_outer / 2;
    
    // frame_target = [frame_x, ce[1], ce[2]];
    frame_target = [frame_x, frame_y, ce[2]];
    
    // Total thickness and width of the arm extension
    arm_height = arm_th; // Matches your 5mm depth
    arm_width  = arm_th + 2; 

    difference() {
        union() {
            // Hull between two cubes that match the full 5mm height and width
            hull() {
                translate(ce) 
                    cube([arm_width, arm_width, arm_height], center = true);
                
                translate(frame_target) 
                    cube([arm_width, arm_width, arm_height], center = true);
            }
        }

        // Vertical clip screw clearance hole
        translate(ce) 
            cylinder(h = arm_height + 8, center = true, d = clip_hole_dia);
        
        // Trim everything past the frame's back surface plane
        place_frame() {
            translate([-100, -100, -100])
                cube([100, 200, 200]); 
        }
        
        // Trim outer Y overhang flush with frame walls
        if (sy > 0) {
            translate([-100, frame_outer/2, -100])
                cube([200, 100, 200]);
        } else {
            translate([-100, -100 - frame_outer/2, -100])
                cube([200, 100, 200]);
        }
        
    }
}

module arm() { for (sy = [-1, 1]) one_arm(sy); }   // assembly shows both

module frame_and_arms() {
    union() {
        frame_piece();
        
        // Transform the arms back into local frame coordinates
        rotate([0, tilt_angle, 0]) 
            translate([-px, 0, -pz]) 
                arm();
    }
}

// ============================================================================
// MOCKS  (eye + rod + gaze ray + lens FOV cone + eye FOV cone; not printed)
//         clips can be printed maybe...
// ============================================================================
module lens_mock() {
    translate([lens_base_x, 0, frame_outer/2]) rotate([0,90,0]) {
        color("DimGray") cylinder(h=lens_len, d=12);
        color("Black") translate([0,0,lens_len-3]) cylinder(h=3, d=14);
    }
}

// Rod is the bottom glasses frame
module rod_mock() { color("Silver") rotate([90,0,0]) linear_extrude(60,center=true) oval_x_z(2.5, 1.375); }


module eye_mock() {
    eye_dia   = 24;  // Average adult eyeball diameter (mm)
    pupil_dia = 4;   // Typical pupil diameter in IR/indoor conditions (mm)
    iris_dia  = 12;  // Average human iris diameter (mm)
    
    // 1. Eyeball (Sclera) - Centered 12 mm (+X) behind the pupil plane
    color("White", 0.85) 
        translate(eye + [eye_dia/2, 0, 0]) 
            sphere(d = eye_dia);

    // 2. Iris - A flat disc flush at the pupil plane (facing -X toward the camera)
    color("LightBlue") 
        translate(eye + [0.1, 0, 0]) 
            rotate([0, -90, 0]) 
                linear_extrude(0.2) 
                    circle(d = iris_dia);

    // 3. Pupil - A flat dark aperture at the exact target location
    color("Black") 
        translate(eye) 
            rotate([0, -90, 0]) 
                linear_extrude(0.3) 
                    circle(d = pupil_dia);

    // 4. Center-of-pupil target marker (keeps your original visual focal point)
    color("Red") 
        translate(eye) 
            sphere(d = 1.0);

    // Gaze and Lens Rays
    color("Tomato") hull(){ translate(eye) sphere(0.4); translate(lens_tip) sphere(0.4); }
    lens_dir = [ cos(tilt_angle), 0, sin(tilt_angle) ];
    lens_end = lens_tip + lens_dir * 50;
    color("Lime") hull(){ translate(lens_tip) sphere(0.4); translate(lens_end) sphere(0.4); }
}

module lens_fov_cone() {
    // cone from the lens tip toward the eye, half-angle fov/2, length to the eye
    L = tip_to_eye;
    color("Yellow", 0.12)
        translate(lens_tip)
            rotate([0, 90 - tilt_angle, 0])   // point +Z' along the lens axis (elev tilt)
                cylinder(h = L, r1 = 0, r2 = L*tan(fov_deg/2));
}

// ----------------------------------------------------------------------------
// EYE FOV CONE MODULE
// ----------------------------------------------------------------------------
// Models monocular field of view originating from the pupil looking straight ahead (-X).
// Default parameters reflect single-eye biological limits (~150° H x ~130° V).
// ----------------------------------------------------------------------------
module eye_fov_cone(){
    fov_h = 30;     // Total horizontal field of view (degrees)
    fov_v = 30;     // Total vertical field of view (degrees)
    range = 100;     // How far out to project the cone (mm)
    gaze_dir = [-1, 0, 0]; // Straight ahead vector (-X toward forward gaze)
    
    // Semi-angles (half-fov)
    rx = range * tan(fov_h / 2);
    rz = range * tan(fov_v / 2);

    color("Cyan", 0.08) // Soft semi-transparent blue tint
        translate(eye)
            // Orient the cone along the gaze direction (-X)
            rotate([0, -90, 0])
                scale([1, rz / rx, 1]) // Scale Z to match elliptical vertical FOV
                    cylinder(h = range, r1 = 0, r2 = rx);
}

// ---- real clip (imported from bottom mount/*.step -> clip_ref.stl) ------
// rotate([90,0,0]) makes the U-channel capture the rod (verified). Tune the
// centering offsets so the channel sits on the rod; clips at +/- clip_pitch/2.
clip_dx = -2.2; clip_dz = 0;
module clip_mock(sy) {
    color("GhostWhite")
        translate([clip_dx, sy*clip_pitch/2, clip_dz])
            rotate([90, 0, 0]) import("clip_ref.stl");
}

// ============================================================================
module place_frame() { translate([px,0,pz]) rotate([0,-tilt_angle,0]) children(); }

if (show == "assembly") {
    color("Khaki")        place_frame() frame_and_arms();
    
    // color("LightSteelBlue") arm();
    
    place_frame() lens_mock();
    rod_mock();
    for (sy = [-1, 1]) clip_mock(sy);
    eye_mock();
    lens_fov_cone();
    eye_fov_cone();
    
} else if (show == "exploded") {
    place_frame() frame_piece();
    place_frame() lens_mock();
    
    translate([0,0,-25]) arm();
    rod_mock(); 
    eye_mock();
    for (sy = [-1, 1]) clip_mock(sy);
    
} else if (show == "frame") {
    frame_piece();
} else if (show == "arm") {
    one_arm(1);          // ONE arm (print two: this + its mirror)
} else if (show == "print") {
    // Whole part (frame + angled arms) lying flat, pocket up
    rotate([0, -90, 0]) 
        frame_and_arms();
}
