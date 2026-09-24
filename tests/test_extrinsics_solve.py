"""The eye<->scene extrinsics solve recovers known transforms from synthetic
board poses, and the saved record maps eye-cam coords to scene-cam coords."""
import cv2
import numpy as np

from scripts.extras.calibrate_extrinsics import (
    inv_T, rig_calibration_record, rotation_angle_deg, solve_extrinsics, to_T,
)


def _rot(axis, deg):
    axis = np.asarray(axis, float)
    return cv2.Rodrigues(axis / np.linalg.norm(axis) * np.radians(deg))[0]


# Roughly the real rig: cameras ~50/13/50 mm apart, looking in very different
# directions; boards on two jig sheets ~90° apart.
T_EYE_SCENE = to_T(_rot([1, 0.2, 0.1], 150), [50.0, -13.0, 50.0])
T_A_B = to_T(_rot([0, 1, 0], 90), [250.0, 0.0, 250.0])


def _pairs(rng, n, rot_noise_deg=0.0, trans_noise_mm=0.0):
    """Board poses each camera would measure at n varied headset poses."""
    T_B_A = inv_T(T_A_B)
    pairs = []
    for _ in range(n):
        T_scene_B = to_T(_rot(rng.normal(size=3), rng.uniform(5, 30)),
                         [rng.uniform(-40, 40), rng.uniform(-40, 40), rng.uniform(250, 350)])
        T_eye_A = T_EYE_SCENE @ T_scene_B @ T_B_A
        pair = {}
        for name, T in (("eye", T_eye_A), ("scene", T_scene_B)):
            R = _rot(rng.normal(size=3), rng.normal(0, rot_noise_deg)) @ T[:3, :3]
            t = T[:3, 3] + rng.normal(0, trans_noise_mm, 3)
            pair[f"{name}_rvec"] = cv2.Rodrigues(R)[0]
            pair[f"{name}_tvec"] = t.reshape(3, 1)
        pairs.append(pair)
    return pairs


def _errors(T_est, T_true):
    return (rotation_angle_deg(T_est[:3, :3].T @ T_true[:3, :3]),
            float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])))


def test_recovers_exact_transforms():
    result = solve_extrinsics(_pairs(np.random.default_rng(0), 15))
    rot, trans = _errors(result["T_eye_scene"], T_EYE_SCENE)
    assert rot < 0.01 and trans < 0.01
    rot, trans = _errors(result["T_A_B"], T_A_B)
    assert rot < 0.01 and trans < 0.01


def test_tolerates_measurement_noise():
    result = solve_extrinsics(_pairs(np.random.default_rng(1), 20,
                                     rot_noise_deg=0.2, trans_noise_mm=0.5))
    rot, trans = _errors(result["T_eye_scene"], T_EYE_SCENE)
    assert rot < 1.0 and trans < 5.0


def test_record_maps_eye_coords_to_scene_coords():
    result = solve_extrinsics(_pairs(np.random.default_rng(2), 12))
    intr = dict(K=np.eye(3), dist=np.zeros(5), image_size=(640, 480),
                reproj_rms=None, source="test.json")
    record = rig_calibration_record(result, intr, intr, "small", "large", "test")
    ext = record["extrinsics_eye_to_scene"]
    R, t = np.array(ext["R"]), np.array(ext["t"])

    p_scene = np.array([10.0, -20.0, 300.0])
    p_eye = (T_EYE_SCENE @ np.append(p_scene, 1.0))[:3]
    assert np.allclose(R @ p_eye + t, p_scene, atol=1e-3)
