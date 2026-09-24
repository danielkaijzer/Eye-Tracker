"""The distortion-aware homography maps screen points to the right raw
scene-cam pixels.

With intrinsics loaded, H is solved in undistorted, normalized camera
coordinates, and every public projection must convert back to raw (distorted)
pixels. These tests use the real lens model from scene_intrinsics.json:
- the pixel <-> undistorted conversions round-trip, and
- with a synthetic screen seen through that lens, project_via_homography()
  lands exactly on each target's true pixel, while the old raw-pixel solve
  misses by pixels (so the test can tell the difference).
"""
import json

import cv2
import numpy as np

from scripts.eyetracker.calibration.paths import scene_intrinsics_path
from scripts.eyetracker.scene.aruco_homography import ArucoHomography

SCREEN_W, SCREEN_H = 1920, 1080


def _lens():
    with open(scene_intrinsics_path()) as f:
        data = json.load(f)
    return np.array(data["K"], float), np.array(data["dist"], float).reshape(-1)


def _mapper(use_intrinsics=True):
    mapper = ArucoHomography()
    K, dist = _lens()
    mapper._K, mapper._dist, mapper._has_intrinsics = K, dist, use_intrinsics
    mapper.set_screen_size(SCREEN_W, SCREEN_H)
    return mapper


def _true_scene_px(screen_pts, G_norm, K, dist):
    """Where screen points really land in the raw image: screen -> undistorted
    normalized coords (the homography G_norm) -> lens distortion -> pixels."""
    pts = np.hstack([np.asarray(screen_pts, float), np.ones((len(screen_pts), 1))])
    proj = (G_norm @ pts.T).T
    norm = np.hstack([proj[:, :2] / proj[:, 2:3], np.ones((len(proj), 1))])
    px, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1, 2)


def _synthetic_screen(K):
    """A screen filling most of the frame with mild keystone, so the corner
    markers sit near the image edges where distortion is largest."""
    screen_quad = np.float32([[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]])
    image_quad = np.float32([[70, 50], [1860, 80], [1830, 1040], [90, 1010]])
    G_px = cv2.getPerspectiveTransform(screen_quad, image_quad)
    return np.linalg.inv(K) @ G_px   # screen px -> undistorted normalized coords


def _solve(mapper, G_norm, K, dist):
    anchors = mapper.screen_anchor_points()
    ids = np.array(mapper.marker_ids, dtype=np.int32).reshape(-1, 1)
    centers = _true_scene_px([anchors[i] for i in mapper.marker_ids], G_norm, K, dist)
    # Each marker's 4 corners collapse onto its center; the solver averages them.
    corners = [np.tile(c, (1, 4, 1)).astype(np.float32) for c in centers]
    H, reproj_err = mapper._homography_from(corners, ids)
    assert H is not None
    return H, reproj_err


def _target_grid():
    xs = np.linspace(200, SCREEN_W - 200, 7)
    ys = np.linspace(200, SCREEN_H - 200, 5)
    return np.array([(x, y) for y in ys for x in xs])


def test_pixel_undistorted_round_trip():
    mapper = _mapper()
    xs, ys = np.meshgrid(np.linspace(0, 1919, 9), np.linspace(0, 1079, 7))
    px = np.column_stack([xs.ravel(), ys.ravel()])
    back = np.array([mapper._hspace_to_pixel(p) for p in mapper._pixel_to_hspace(px)])
    assert np.abs(back - px).max() < 0.05


def test_projection_matches_true_pixels_through_the_lens():
    K, dist = _lens()
    G_norm = _synthetic_screen(K)
    targets = _target_grid()
    truth = _true_scene_px(targets, G_norm, K, dist)

    mapper = _mapper(use_intrinsics=True)
    H, reproj_err = _solve(mapper, G_norm, K, dist)
    pred = np.array([mapper.project_via_homography(tuple(t), H) for t in targets])
    assert reproj_err < 0.05
    assert np.linalg.norm(pred - truth, axis=1).max() < 0.1

    # The old raw-pixel solve on the same data misses by pixels, so this test
    # would catch the distortion handling being dropped or bypassed.
    raw = _mapper(use_intrinsics=False)
    H_raw, _ = _solve(raw, G_norm, K, dist)
    pred_raw = np.array([raw.project_via_homography(tuple(t), H_raw) for t in targets])
    assert np.linalg.norm(pred_raw - truth, axis=1).max() > 1.0
