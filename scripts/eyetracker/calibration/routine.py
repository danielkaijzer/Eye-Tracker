"""CalibrationRoutine — the calibration state machine.

Owns:
- the active target list (from a TargetPattern)
- the current fixation index, captured-vector buffer, skipped-index list
- a SampleCollector for per-fixation variance gating
- the per-session image dump dir + labels.csv path
- a queue of "pending" CSV rows + image paths that are flushed on Accept
  or thrown away on Reject / Skip

Does NOT own the Tk overlay — the overlay reads routine state and the App
loop drives it. The routine signals completion by setting `is_active` to
False; the App is responsible for calling overlay.close() in response.

Two construction-time modes layer on top of the basic grid:
- `num_poses > 1` (multi-pose): run the same grid at several head poses in
  one session, pausing in a "pose break" between them, and aggregate every
  captured point into a single fit. See docs/multipose_calibration.md.
- `fit_on_finish=False` (validation): capture identically but fit nothing and
  dump the medians to validation_*.npz instead, leaving the live calibration
  untouched. See docs/calibration_coverage.md.
"""
import math
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from scripts.eyetracker.calibration.collector import (
    CollectorAccept,
    CollectorReject,
    SampleCollector,
)
from scripts.eyetracker.calibration.persistence import (
    CalibrationSnapshot,
    append_label_rows,
    begin_session,
    save_calibration,
    save_validation,
)
from scripts.eyetracker.calibration.targets import TargetPattern
from scripts.eyetracker.config import ARUCO_IDS, CALIB_POSE_GUIDANCE
from scripts.eyetracker.gaze.base import FitReport, GazeMapper
from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper
from scripts.eyetracker.pupil.gating import JumpGate
from scripts.eyetracker.scene.aruco_homography import ArucoHomography


_LOG_THROTTLE_S = 1.0


class CalibrationRoutine:
    def __init__(self,
                 pattern: TargetPattern,
                 collector: SampleCollector,
                 target_mapper: ArucoHomography,
                 mapper: GazeMapper,
                 jump_gate: Optional[JumpGate] = None,
                 mapper_degree: int = 2,
                 recapture_worst_n: int = 0,
                 num_poses: int = 1,
                 fit_on_finish: bool = True,
                 label: str = "calibration"):
        self.pattern = pattern
        self.collector = collector
        self.target_mapper = target_mapper
        self.mapper = mapper
        self.jump_gate = jump_gate
        # Polynomial degree to apply when fitting via this routine. The
        # mapper is shared between routines, so we set its degree just
        # before fitting rather than at construction time.
        self.mapper_degree = mapper_degree
        # 0 = single pass. >0 = after pass-1 fit, drop the N highest-LOO-
        # residual fixations and re-prompt those targets before refitting.
        self.recapture_worst_n = recapture_worst_n
        # >1 runs the same grid at several head poses in one session and
        # aggregates all captured points into a single fit, extending pupil/
        # scene coverage toward the full oculomotor range. Each pose is still
        # captured static (the per-fixation gate rejects head motion); the user
        # repositions only between poses.
        self.num_poses = num_poses
        # False = collect-only "validation" run: skip fitting / saving the live
        # calibration and instead dump the captured medians to validation_*.npz
        # for held-out accuracy measurement.
        self.fit_on_finish = fit_on_finish
        # Used in console output to distinguish quick vs detailed sessions.
        self.label = label

        # State machine fields. is_active is the public "are we calibrating
        # right now" flag; -1 sentinel for current_idx means inactive.
        self.is_active: bool = False
        self.is_collecting: bool = False
        self.current_idx: int = -1
        self.targets: List[Tuple[int, int]] = []
        self.skipped_indices: List[int] = []
        self.captured_pupil: List[np.ndarray] = []
        self.captured_scene: List[np.ndarray] = []
        # Target index (into self.targets) for each captured fixation.
        # Used by the two-pass recapture path to remap "worst sample i"
        # back to "redo target k".
        self.captured_target_indices: List[int] = []
        # True iff we're currently re-prompting pass-2 (recapture) targets.
        self._in_pass_two: bool = False
        # Multi-pose state. current_pose is 1-based; _awaiting_pose is the
        # "grid done, waiting for the user to reposition + press 'c'" break
        # between poses.
        self.current_pose: int = 1
        self._awaiting_pose: bool = False
        # Original (pass-1) targets list, preserved so the saved snapshot
        # records the full screen-point set even after self.targets is
        # narrowed to the recapture subset in pass 2.
        self._original_targets: List[Tuple[int, int]] = []
        self.screen_width: Optional[int] = None
        self.screen_height: Optional[int] = None
        # Set externally (by App / wiring) once the scene camera reports its
        # actual frame size; used at save time to record scene_width/height.
        self.scene_size: Optional[Tuple[int, int]] = None
        self.session_dir: Optional[str] = None
        self.labels_path: Optional[str] = None

        # Pending state cleared on every reject / advance / skip.
        self._pending_rows: list = []
        self._pending_image_paths: List[str] = []

        # Throttle timestamps for spammy frame-rate-driven logs.
        self._last_no_scene_log_ts = 0.0
        self._last_aruco_log_ts = 0.0

    # ---- View accessors used by the overlay ----------------------------------

    @property
    def total_points(self) -> int:
        return len(self.targets)

    @property
    def collecting_sample_count(self) -> int:
        return self.collector.sample_count()

    @property
    def awaiting_pose(self) -> bool:
        """True while paused between poses, waiting for the user to reposition
        their head and press 'c'. Read by the overlay to show guidance."""
        return self._awaiting_pose

    def next_pose_guidance(self, pose: Optional[int] = None) -> str:
        """Human-readable instruction for the given 1-based pose (defaults to
        the pose we're about to start). Falls back gracefully if there are more
        poses than guidance strings."""
        idx = (pose if pose is not None else self.current_pose + 1) - 1
        if 0 <= idx < len(CALIB_POSE_GUIDANCE):
            return CALIB_POSE_GUIDANCE[idx]
        return "reposition your head to a new orientation"

    # ---- Lifecycle -----------------------------------------------------------

    def start(self, screen_width: int, screen_height: int) -> None:
        """Initialize for a new calibration session. Call after the screen
        size has been determined (e.g. after the Tk overlay sized its root)."""
        self.is_active = True
        self.is_collecting = False
        self.current_idx = 0
        self.targets = self.pattern.generate(screen_width, screen_height)
        self.skipped_indices = []
        self.captured_pupil = []
        self.captured_scene = []
        self.captured_target_indices = []
        self._in_pass_two = False
        self._original_targets = []
        self.current_pose = 1
        self._awaiting_pose = False
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._pending_rows = []
        self._pending_image_paths = []
        self.collector.reset()
        if self.jump_gate is not None:
            self.jump_gate.reset()

        self.session_dir, self.labels_path = begin_session()

        print(f"{self.label.capitalize()} started ({self.total_points} points, "
              f"degree-{self.mapper_degree} polynomial) on "
              f"{screen_width}x{screen_height} screen.")
        if self.recapture_worst_n > 0:
            print(f"  Two-pass mode: worst {self.recapture_worst_n} fixations "
                  "will be re-prompted after pass 1.")
        if self.num_poses > 1:
            print(f"  Multi-pose mode: {self.num_poses} head poses, "
                  f"{self.total_points} points each (do not remove the headset).")
            print(f"  Pose 1/{self.num_poses}: {self.next_pose_guidance(1)}")
        if not self.fit_on_finish:
            print("  Validation mode: collect-only, the live calibration "
                  "is left untouched.")
        print(f"  Dataset: {self.session_dir}")
        print("  Look at the RED dot and press 'c'.")

    def begin_capture(self) -> None:
        """User pressed 'c' on an idle target. Start the per-fixation buffer."""
        if not self.is_active or self.is_collecting or self._awaiting_pose:
            return
        self.is_collecting = True
        self.collector.begin()

    def begin_next_pose(self) -> None:
        """User pressed 'c' during a pose break. Re-arm the same grid for the
        next pose, keeping the points captured so far so they all feed one fit."""
        if not (self.is_active and self._awaiting_pose):
            return
        self.current_pose += 1
        self._awaiting_pose = False
        self.current_idx = 0
        self.targets = self.pattern.generate(self.screen_width, self.screen_height)
        self.skipped_indices = []
        self.is_collecting = False
        self.collector.reset()
        if self.jump_gate is not None:
            self.jump_gate.reset()
        print(f"Pose {self.current_pose}/{self.num_poses}: "
              f"{self.next_pose_guidance(self.current_pose)}")
        print("  Look at the RED dot and press 'c'.")

    def skip(self) -> None:
        """User pressed 's'. Mark current target skipped, advance or finish."""
        if not self.is_active:
            return
        if self.current_idx < 0 or self.current_idx >= self.total_points:
            return

        if self.is_collecting:
            self._discard_pending()
            self.is_collecting = False
            self.collector.reset()

        self.skipped_indices.append(self.current_idx)
        print(f"Skipped point {self.current_idx + 1}/{self.total_points}.")
        self._advance_or_finish()

    # ---- Per-frame drive -----------------------------------------------------

    def tick(self, *,
             pupil_center: Optional[np.ndarray],
             eye_frame: Optional[np.ndarray],
             scene_frame: Optional[np.ndarray],
             confidence: float) -> None:
        """Called every frame from the App loop. Only does work when
        `is_collecting` is True. The pupil_center/eye_frame are produced by
        the pupil pipeline; pass them straight in."""
        if not (self.is_active and self.is_collecting):
            return
        if pupil_center is None or eye_frame is None:
            return
        if self.collector.consume_warmup_frame():
            return

        if scene_frame is None:
            self._throttled("_last_no_scene_log_ts",
                            "  No scene camera — cannot calibrate in scene-cam mode.")
            return

        H, _ = self.target_mapper.compute_homography(scene_frame)
        if H is None:
            self._throttled("_last_aruco_log_ts",
                            "  [aruco] not all 4 markers visible")
            return

        tx, ty = self.targets[self.current_idx]
        proj = H @ np.array([tx, ty, 1.0], dtype=float)
        if abs(proj[2]) < 1e-9:
            self._throttled("_last_aruco_log_ts",
                            "  [aruco] degenerate homography projection")
            return
        target_u = float(proj[0] / proj[2])
        target_v = float(proj[1] / proj[2])

        sample_idx = self.collector.sample_count()
        img_name = f"fix{self.current_idx:02d}_sample{sample_idx:02d}.png"
        scene_img_name = f"fix{self.current_idx:02d}_sample{sample_idx:02d}_scene.png"
        img_path = os.path.join(self.session_dir, img_name)
        scene_img_path = os.path.join(self.session_dir, scene_img_name)
        cv2.imwrite(img_path, eye_frame)
        cv2.imwrite(scene_img_path, scene_frame)

        px = float(pupil_center[0])
        py = float(pupil_center[1])
        self._pending_rows.append([
            img_name, self.current_idx, tx, ty,
            f"{px:.3f}", f"{py:.3f}", f"{confidence:.4f}", f"{time.time():.3f}",
            f"{target_u:.3f}", f"{target_v:.3f}",
        ])
        self._pending_image_paths.append(img_path)
        self._pending_image_paths.append(scene_img_path)

        result = self.collector.add(np.array(pupil_center, dtype=float),
                                    np.array([target_u, target_v], dtype=float))
        if result is None:
            return
        if isinstance(result, CollectorReject):
            print(f"  {result.reason}")
            self._discard_pending()
            self.is_collecting = False
            self.collector.reset()
            return
        assert isinstance(result, CollectorAccept)
        self._accept(result, scene_frame, target_u, target_v)

    # ---- Internals -----------------------------------------------------------

    def _accept(self,
                result: CollectorAccept,
                scene_frame: np.ndarray,
                target_u: float,
                target_v: float) -> None:
        self.captured_pupil.append(result.pupil_median)
        self.captured_scene.append(result.scene_median)
        self.captured_target_indices.append(self.current_idx)
        self._log_aruco_check(scene_frame, result.scene_median)
        append_label_rows(self.labels_path, self._pending_rows)
        self._pending_rows = []
        self._pending_image_paths = []
        self.is_collecting = False
        self.collector.reset()
        self._advance_or_finish()

    def _log_aruco_check(self, scene_frame: np.ndarray,
                          scene_median: np.ndarray) -> None:
        try:
            H_chk, reproj_err = self.target_mapper.compute_homography(scene_frame)
        except Exception as e:
            print(f"  ArUco detection error for fixation {self.current_idx}: {e}")
            return
        if H_chk is None:
            print(f"  ArUco: not all 4 markers visible for fixation {self.current_idx}")
            return
        tx, ty = self.targets[self.current_idx]
        v = np.array([tx, ty, 1.0], dtype=float)
        proj_chk = H_chk @ v
        if abs(proj_chk[2]) <= 1e-9:
            return
        u = float(proj_chk[0] / proj_chk[2])
        vp = float(proj_chk[1] / proj_chk[2])
        err = math.sqrt((u - scene_median[0]) ** 2 + (vp - scene_median[1]) ** 2)
        print(f"  ArUco check: fixation {self.current_idx} predicted "
              f"({u:.1f},{vp:.1f}) vs label-median "
              f"({scene_median[0]:.1f},{scene_median[1]:.1f}), "
              f"err={err:.1f}px (reproj={reproj_err:.1f}px)")

    def _advance_or_finish(self) -> None:
        if self.current_idx + 1 >= self.total_points:
            if self.current_pose < self.num_poses and not self._in_pass_two:
                self._begin_pose_break()
            else:
                self._finish()
        else:
            self.current_idx += 1
            captured = len(self.captured_pupil)
            if not self.is_collecting:
                # Only print after a successful capture, not after a skip
                if captured > 0 and (captured + len(self.skipped_indices)
                                     == self.current_idx):
                    print(f"  Captured {captured}/{self.total_points}. "
                          "Look at next dot, press 'c'.")

    def _begin_pose_break(self) -> None:
        """Grid for the current pose is done but more poses remain — pause and
        wait for the user to reposition. Buffers are intentionally NOT cleared
        so points accumulate across poses into one fit."""
        self._awaiting_pose = True
        self.is_collecting = False
        self.collector.reset()
        captured = len(self.captured_pupil)
        next_pose = self.current_pose + 1
        print(f"Pose {self.current_pose}/{self.num_poses} done "
              f"({captured} points so far).")
        print(f"  Next, {self.next_pose_guidance(next_pose)}, then press 'c' "
              f"to start pose {next_pose}/{self.num_poses}.")

    def _finish(self) -> None:
        n = len(self.captured_pupil)
        skipped = len(self.skipped_indices)
        if not self.fit_on_finish:
            self._save_validation(n)
            self._end()
            return
        if isinstance(self.mapper, PolynomialGazeMapper):
            self.mapper.set_degree(self.mapper_degree)
        min_pts = self._min_fit_points()
        if n < min_pts:
            print(f"Not enough non-skipped points to fit "
                  f"({n} captured, {skipped} skipped). "
                  f"Need at least {min_pts}.")
            self._end()
            return

        try:
            report = self.mapper.fit(np.array(self.captured_pupil),
                                     np.array(self.captured_scene))
        except ValueError as e:
            print(str(e))
            self._end()
            return

        if self._should_run_pass_two(report):
            self._begin_pass_two(report)
            return

        self._report_and_save(report)
        self._end()

    def _save_validation(self, n: int) -> None:
        if n < 1:
            print("No points captured — nothing to save as validation.")
            return
        snapshot = self._build_snapshot()
        save_validation(snapshot)

    def _min_fit_points(self) -> int:
        if isinstance(self.mapper, PolynomialGazeMapper):
            return {2: 6, 3: 10}[self.mapper.degree]
        return 6

    def _should_run_pass_two(self, report: FitReport) -> bool:
        if self.recapture_worst_n <= 0 or self._in_pass_two:
            return False
        if report.per_point_errs is None:
            return False
        # Need enough surviving samples after drop to keep the fit valid
        # with breathing room above the minimum.
        keep = len(self.captured_pupil) - self.recapture_worst_n
        return keep >= self._min_fit_points() + 1

    def _begin_pass_two(self, report: FitReport) -> None:
        errs = np.asarray(report.per_point_errs)
        worst = sorted(np.argsort(errs)[-self.recapture_worst_n:].tolist())
        print(f"Pass 1 LOO: avg={report.loo_avg_err:.1f}px, "
              f"max={report.loo_max_err:.1f}px.")
        print(f"  Re-prompting {len(worst)} worst fixations "
              f"(per-point LOO errs: "
              f"{[f'{errs[i]:.1f}px' for i in worst]}).")

        # Capture the on-screen positions of the targets we're redoing
        # before we mutate self.targets, and preserve the pass-1 target
        # list for the saved snapshot.
        recapture_target_pts = [self.targets[self.captured_target_indices[i]]
                                for i in worst]
        self._original_targets = list(self.targets)

        keep_idx = [i for i in range(len(self.captured_pupil)) if i not in worst]
        self.captured_pupil = [self.captured_pupil[i] for i in keep_idx]
        self.captured_scene = [self.captured_scene[i] for i in keep_idx]
        # Targets list is reset to the pass-2 set; index mapping inside
        # captured_target_indices is no longer meaningful after this and
        # we won't read it again (recapture only happens once).
        self.captured_target_indices = []

        self.targets = recapture_target_pts
        self.skipped_indices = []
        self.current_idx = 0
        self.is_collecting = False
        self.collector.reset()
        if self.jump_gate is not None:
            self.jump_gate.reset()
        self._in_pass_two = True
        print("  Look at the RED dot and press 'c'.")

    def _report_and_save(self, report: FitReport) -> None:
        skipped = len(self.skipped_indices)
        pass_label = " (pass 2)" if self._in_pass_two else ""
        print(f"Polynomial calibration fitted ({report.n_points} points, "
              f"{skipped} skipped, degree {self.mapper_degree}){pass_label}.")
        print(f"  LOO error: avg={report.loo_avg_err:.1f}px, "
              f"max={report.loo_max_err:.1f}px")
        snapshot = self._build_snapshot()
        scene_w = snapshot.scene_size[0] if snapshot.scene_size is not None else 640
        # 4% of scene-cam width is a rough "is the fit usable" threshold;
        # caller can override by inspecting report directly.
        err_threshold = 0.04 * scene_w
        if report.loo_avg_err > err_threshold:
            print(f"  WARNING: High error (>{err_threshold:.0f}px at this "
                  "scene-cam resolution) — consider recalibrating.")
        save_calibration(snapshot, self.mapper)
        print("Calibration Complete!")

    def _end(self) -> None:
        self.is_active = False
        self.is_collecting = False
        self.current_idx = -1
        self._in_pass_two = False

    def _build_snapshot(self) -> CalibrationSnapshot:
        anchors = self.target_mapper.screen_anchor_points()
        aruco_screen_centers = (
            np.array([anchors[i] for i in ARUCO_IDS], dtype=float)
            if anchors else np.zeros((0, 2), dtype=float)
        )
        screen_size = ((self.screen_width, self.screen_height)
                       if self.screen_width is not None and self.screen_height is not None
                       else None)
        # In pass 2 the captured_target_indices mapping is no longer valid
        # (self.targets was narrowed to the recapture subset), so fall back to
        # the preserved full pass-1 grid. Otherwise map each captured fixation
        # back to its on-screen target so screen_points stays length-matched to
        # pupil_vectors — across skips and across repeated grids in multi-pose.
        if self._in_pass_two and self._original_targets:
            screen_targets = self._original_targets
        else:
            screen_targets = [self.targets[idx]
                              for idx in self.captured_target_indices]
        return CalibrationSnapshot(
            pupil_vectors=np.array(self.captured_pupil),
            scene_points=np.array(self.captured_scene),
            screen_points=np.array(screen_targets),
            aruco_screen_centers=aruco_screen_centers,
            scene_size=self.scene_size,
            screen_size=screen_size,
        )

    def _discard_pending(self) -> None:
        for p in self._pending_image_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        self._pending_rows = []
        self._pending_image_paths = []

    def _throttled(self, ts_field: str, msg: str) -> None:
        now = time.time()
        last = getattr(self, ts_field)
        if now - last >= _LOG_THROTTLE_S:
            print(msg)
            setattr(self, ts_field, now)
