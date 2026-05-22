# Data I want to Collect during calibration sequence and clicking game

- Camera intrinsics inluding
    - focal + principal point
    - distortion coefficients
- Camera extrinsics (via custom jig before the session recording)
- Depth/vector from eye to eye-camera optical center
- Estimated eye center
- 2D pupil estimates; raw ellipse params (center, axes, angle) + detector confidence per frame
- Estimated 3D vector from eye center to pupil center (via pye3d); gaze direction
    - Also normalized unit vector
- scene gaze point x,y
- Eye camera image with timestamp (original and normalized)
- Scene camera image with timestamp (original and normalized)
- Ground truth pixel location
- Vector from ground truth pixel to scene camera optical center
- ArUcO locations
- phase tag (calibration vs clicking game vs optical flow)
- subject ID
- glasses enum (glasses vs contacts vs nothing)
- headset model version
- per-subject kappa angle
- py3d version
- pupil_detector version

For clicking game:
- click_pixel_x
- click_pixel_y
- click_timestamp

Maybe (if I add these sensors):
- Depth sensor data (for distance) timestamped
- IMU sensor (for pose) timestamped
