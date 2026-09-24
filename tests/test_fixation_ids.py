"""Regression guard: fixation ids must be unique within a calibration session.

current_idx restarts at 0 for each multi-pose grid and for the pass-2
recapture subset. It used to double as labels.csv `fixation_id` and the image
filename, so later poses / recaptures reused ids 0..k and overwrote earlier
images (found on session_20260923_175740). These tests drive the routine end
to end with a fake ArUco mapper (identity homography: scene == screen coords)
and check ids, filenames, and snapshot alignment.

Runs standalone: python -m tests.test_fixation_ids (from the repo root).
"""
import csv
import os
import tempfile
from unittest import mock

import numpy as np

from scripts.eyetracker.calibration import routine as routine_mod
from scripts.eyetracker.calibration.collector import SampleCollector
from scripts.eyetracker.calibration.persistence import LABELS_CSV_HEADER
from scripts.eyetracker.calibration.routine import CalibrationRoutine
from scripts.eyetracker.calibration.targets import GridPattern
from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper

SCREEN = (1920, 1080)


class _FakeTargetMapper:
    def cached_homography(self):
        return np.eye(3), 0.0

    def screen_anchor_points(self):
        return None


def _pupil_for(target, rng):
    """Smooth screen->pupil map plus a fixed per-fixation offset, so the fit
    has nonzero LOO residuals (and pass 2 has 'worst' points to redo)."""
    tx, ty = target
    return np.array([300 - tx / 12, 200 + ty / 15]) + rng.normal(0, 3, 2)


def _run(routine, rng, skip_first=False):
    snapshots = []
    clock = [1000.0]   # fake monotonic frame timestamps (s)
    with tempfile.TemporaryDirectory() as tmp:
        labels = os.path.join(tmp, "labels.csv")
        with open(labels, "w", newline="") as f:
            csv.writer(f).writerow(LABELS_CSV_HEADER)
        with mock.patch.object(routine_mod, "begin_session", return_value=(tmp, labels)), \
             mock.patch.object(routine_mod, "save_calibration",
                               side_effect=lambda snap, *a, **k: snapshots.append(snap)), \
             mock.patch.object(routine_mod, "write_session_metadata",
                               side_effect=lambda d, snap, *a, **k: snapshots.append(snap)):
            routine.start(*SCREEN)
            frame = np.zeros((4, 4, 3), np.uint8)
            if skip_first:
                routine.skip()
            while routine.is_active:
                if routine._awaiting_pose:
                    routine.begin_next_pose()
                    continue
                pupil = _pupil_for(routine.targets[routine.current_idx], rng)
                routine.begin_capture()
                while routine.is_collecting:
                    clock[0] += 0.01
                    routine.tick(pupil_center=pupil + rng.normal(0, 0.2, 2),
                                 eye_frame=frame, scene_frame=frame, confidence=1.0,
                                 eye_ts=clock[0], scene_ts=clock[0] - 0.02)
            with open(labels) as f:
                rows = list(csv.DictReader(f))
            images = sorted(os.listdir(tmp))
    return rows, images, snapshots[-1]


def _routine(rows, cols, degree, recapture=0, poses=1):
    return CalibrationRoutine(
        pattern=GridPattern(rows=rows, cols=cols, margin=180),
        collector=SampleCollector(samples=5, inliers=4, pupil_std_thresh=12.0,
                                  scene_std_thresh=10.0, warmup=0),
        target_mapper=_FakeTargetMapper(),
        mapper=PolynomialGazeMapper(degree=degree),
        mapper_degree=degree,
        recapture_worst_n=recapture,
        num_poses=poses,
    )


def _check_unique(rows, images, n_fixations):
    ids = sorted({int(r["fixation_id"]) for r in rows})
    assert len(ids) == n_fixations, ids
    # Each id's rows share one screen target.
    for fid in ids:
        targets = {(r["x_screen"], r["y_screen"]) for r in rows if int(r["fixation_id"]) == fid}
        assert len(targets) == 1, (fid, targets)
    # Every labelled image survives (nothing overwritten).
    eye_imgs = [i for i in images if i.endswith(".png") and not i.endswith("_scene.png")]
    assert len(eye_imgs) == len(rows), (len(eye_imgs), len(rows))
    assert {r["image_path"] for r in rows} == set(eye_imgs)


def _check_snapshot_aligned(snap):
    # Identity homography => scene label == screen target for every fixation.
    assert len(snap.screen_points) == len(snap.pupil_vectors) == len(snap.scene_points)
    np.testing.assert_allclose(snap.scene_points, snap.screen_points, atol=1e-6)


def test_recapture_gets_new_fixation_ids():
    rng = np.random.default_rng(0)
    rows, images, snap = _run(_routine(4, 5, degree=3, recapture=5), rng)
    _check_unique(rows, images, n_fixations=25)          # 20 pass-1 + 5 recaptures
    assert len(snap.superseded_fixation_ids) == 5
    assert all(0 <= i < 20 for i in snap.superseded_fixation_ids)
    assert len(snap.pupil_vectors) == 20
    _check_snapshot_aligned(snap)


def test_recapture_with_skip_redoes_the_right_targets():
    rng = np.random.default_rng(1)
    rows, images, snap = _run(_routine(4, 5, degree=3, recapture=5), rng, skip_first=True)
    _check_unique(rows, images, n_fixations=24)          # 19 pass-1 + 5 recaptures
    assert len(snap.pupil_vectors) == 19
    _check_snapshot_aligned(snap)


def test_frame_timestamps_logged():
    rng = np.random.default_rng(3)
    rows, _, _ = _run(_routine(3, 3, degree=2), rng)
    eye = np.array([float(r["eye_frame_ts"]) for r in rows])
    scene = np.array([float(r["scene_frame_ts"]) for r in rows])
    assert np.all(np.diff(eye) > 0)                    # per-sample, increasing
    np.testing.assert_allclose(eye - scene, 0.02, atol=1e-6)   # paired per row


def test_multipose_fixation_ids_span_poses():
    rng = np.random.default_rng(2)
    rows, images, snap = _run(_routine(3, 4, degree=2, poses=3), rng)
    _check_unique(rows, images, n_fixations=36)          # 3 poses x 12 points
    assert snap.superseded_fixation_ids == []
    _check_snapshot_aligned(snap)


if __name__ == "__main__":
    test_recapture_gets_new_fixation_ids()
    test_recapture_with_skip_redoes_the_right_targets()
    test_frame_timestamps_logged()
    test_multipose_fixation_ids_span_poses()
    print("ok")
