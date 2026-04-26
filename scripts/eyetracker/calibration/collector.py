"""Per-fixation sample collection + variance gating.

The CalibrationRoutine streams `(pupil_xy, scene_target_uv)` samples into a
`SampleCollector` while the user holds their gaze on a target. After
CALIB_SAMPLES samples, the collector picks the CALIB_INLIERS closest to the
median, checks both pupil-pixel and scene-pixel inlier std-dev against
thresholds, and returns either an Accept (with median pupil + median scene
labels) or a Reject (with a human-readable reason)."""
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class CollectorAccept:
    pupil_median: np.ndarray
    scene_median: np.ndarray
    inlier_pupil_std: float
    inlier_scene_std: float
    raw_pupil_std: float


@dataclass
class CollectorReject:
    reason: str
    raw_pupil_std: float
    inlier_pupil_std: float
    inlier_scene_std: float


# `None` here means "not full yet — keep adding samples".
CollectorResult = Union[None, CollectorAccept, CollectorReject]


class SampleCollector:
    def __init__(self,
                 samples: int,
                 inliers: int,
                 pupil_std_thresh: float,
                 scene_std_thresh: float,
                 warmup: int):
        self.samples = samples
        self.inliers = inliers
        self.pupil_std_thresh = pupil_std_thresh
        self.scene_std_thresh = scene_std_thresh
        self.warmup = warmup
        self.warmup_remaining = 0
        self._pupil: list = []
        self._scene: list = []

    def begin(self) -> None:
        self.warmup_remaining = self.warmup
        self._pupil = []
        self._scene = []

    def reset(self) -> None:
        self.warmup_remaining = 0
        self._pupil = []
        self._scene = []

    def consume_warmup_frame(self) -> bool:
        """Returns True if a warmup frame was consumed (caller should skip
        adding this frame's sample). Returns False once warmup is exhausted."""
        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
            return True
        return False

    def sample_count(self) -> int:
        return len(self._pupil)

    def add(self, pupil_xy: np.ndarray, scene_uv: np.ndarray) -> CollectorResult:
        """Append a sample. Returns None if more samples are needed, or an
        Accept/Reject once the buffer is full."""
        self._pupil.append(np.asarray(pupil_xy, dtype=float).copy())
        self._scene.append(np.asarray(scene_uv, dtype=float).copy())
        if len(self._pupil) < self.samples:
            return None
        return self._evaluate()

    def _evaluate(self) -> CollectorResult:
        pupil = np.array(self._pupil)
        scene = np.array(self._scene)
        raw_pupil_std = float(np.max(np.std(pupil, axis=0)))

        median = np.median(pupil, axis=0)
        deviations = np.linalg.norm(pupil - median, axis=1)
        keep_idx = np.argsort(deviations)[: self.inliers]
        inliers = pupil[keep_idx]
        inlier_pupil_std = float(np.max(np.std(inliers, axis=0)))

        scene_inliers = scene[keep_idx]
        inlier_scene_std = float(np.max(np.std(scene_inliers, axis=0)))

        if inlier_pupil_std > self.pupil_std_thresh:
            return CollectorReject(
                reason=(f"High variance (inlier std={inlier_pupil_std:.2f}px, "
                        f"raw std={raw_pupil_std:.2f}px). Retrying — "
                        "hold still and press 'c'."),
                raw_pupil_std=raw_pupil_std,
                inlier_pupil_std=inlier_pupil_std,
                inlier_scene_std=inlier_scene_std,
            )

        if inlier_scene_std > self.scene_std_thresh:
            return CollectorReject(
                reason=(f"High label variance (std={inlier_scene_std:.2f}px). "
                        "Head may have moved. Retrying — hold still and press 'c'."),
                raw_pupil_std=raw_pupil_std,
                inlier_pupil_std=inlier_pupil_std,
                inlier_scene_std=inlier_scene_std,
            )

        return CollectorAccept(
            pupil_median=np.median(inliers, axis=0),
            scene_median=np.median(scene_inliers, axis=0),
            inlier_pupil_std=inlier_pupil_std,
            inlier_scene_std=inlier_scene_std,
            raw_pupil_std=raw_pupil_std,
        )
