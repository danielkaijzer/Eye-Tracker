workspace "Eye Tracker" "Wearable head-mounted eye tracker for high-precision gaze estimation." {

    model {
        wearer = person "Wearer / Researcher" "Wears the headset and reviews live gaze on the dashboard."

        rig = softwareSystem "Headset Rig" "IR eye camera + RGB scene camera mounted on a 3D-printed frame; appears to the host OS as two USB UVC video devices." {
            tags "Hardware"
        }

        supabase = softwareSystem "Supabase" "Hosted authentication and per-user profile/session state." {
            tags "External"
        }

        tracker = softwareSystem "Eye Tracker" "Captures pupil + scene video, calibrates the gaze mapping, streams annotated frames and gaze coordinates to a dashboard." {

            backend = container "Python Backend" "scripts.eyetracker package: capture, pupil detection, calibration, polynomial gaze fit, MJPEG/HTTP bridge." "Python 3.12, OpenCV, Flask" {
                pupil       = component "Pupil Detector"           "Wraps Pupil Labs Detector2D + pye3d 3D eye model."                                              "scripts/eyetracker/pupil/pupil_labs.py"
                gates       = component "Confidence + Jump Gates"  "Reject low-confidence detections and saccade-overshoot outliers."                              "scripts/eyetracker/pupil/gating.py"
                aruco       = component "ArUco Homography"         "Detects screen-corner markers; builds screen-to-scene homography for calibration labels."     "scripts/eyetracker/scene/aruco_homography.py"
                calibration = component "Calibration Routine"      "State machine that walks the fixation grid, collects pupil/scene pairs, fits the mapper."     "scripts/eyetracker/calibration/routine.py"
                mapper      = component "Polynomial Gaze Mapper"   "Bivariate polynomial pupil-px -> scene-cam-px (degree 2 quick / degree 3 detailed)."          "scripts/eyetracker/gaze/polynomial.py"
                smoother    = component "1-Euro Smoother"          "Speed-adaptive low-pass filter on gaze output (Casiez et al. 2012)."                          "scripts/eyetracker/gaze/smoothing.py"
                display     = component "Display / MJPEG Bridge"   "Renders annotated frames via cv2 windows, or streams MJPEG + gaze JSON over HTTP."           "scripts/eyetracker/display/*.py"
            }

            frontend = container "Next.js Dashboard" "Login, live gaze overlay on scene video, real-time heatmap, profile/games/ml-analytics pages." "Next.js 16, React 19, TypeScript" {
                tags "Web"
            }
        }

        // Context relationships
        wearer   -> rig      "Wears"
        wearer   -> frontend "Views live gaze + heatmap"
        rig      -> backend  "USB UVC video (eye + scene)"
        backend  -> frontend "MJPEG frames + gaze JSON over HTTP"
        frontend -> supabase "Auth + session state"

        // Component-level flow inside the backend
        pupil       -> gates       "Raw pupil samples + confidence"
        gates       -> calibration "Accepted pupil centers (during calibration)"
        gates       -> mapper      "Accepted pupil centers (runtime)"
        aruco       -> calibration "Per-frame screen-to-scene homography"
        calibration -> mapper      "Fitted polynomial coefficients"
        mapper      -> smoother    "Predicted scene-cam pixel"
        smoother    -> display     "Smoothed gaze coordinates"
        rig         -> pupil       "Eye-cam frames"
        rig         -> aruco       "Scene-cam frames"
        rig         -> display     "Scene-cam frames (passthrough for overlay)"
        display     -> frontend    "MJPEG frames + gaze JSON over HTTP"
    }

    views {
        systemContext tracker "C1-Context" {
            include *
            autoLayout lr
        }

        container tracker "C2-Container" {
            include *
            autoLayout lr
        }

        component backend "C3-Component" {
            include *
            autoLayout lr
        }

        styles {
            element "Person" {
                shape Person
                background #2c3e50
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Hardware" {
                shape Box
                background #444444
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Web" {
                shape WebBrowser
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
        }
    }
}
