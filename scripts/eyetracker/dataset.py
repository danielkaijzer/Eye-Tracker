"""Read collected calibration sessions for analysis.

A session is a directory `session_<ts>/` under `dataset_root()` containing:
- `labels.csv` — one row per accepted sample (the ground-truth pairs),
- the per-sample eye/scene frames, and
- `metadata.json` — camera intrinsics/extrinsics + provenance (sessions captured
  before the JSON-metadata change lack this; they're treated as legacy).

This is the read side of the dataset (the capture pipeline writes it). Today the
only consumer is `scripts/extras/measure_gaze_accuracy.py`, which refits the
polynomial from one session.

TODO (CNN training work — build when the training loop actually needs it):
- `load_all(root)`: glob every session's labels.csv into one DataFrame tagged
  with `session_id`, and cache a typed `combined.parquet` (rebuilt when any
  session changes) for fast whole-dataset loads. Add `pyarrow` to requirements
  when this lands — CSV stays the crash-safe capture format, Parquet is the
  consumption format.
- An image-loading layer: this module only returns the labels table; a CNN
  Dataset/DataLoader still needs to read each row's `image_path` (the eye-frame
  PNG) and join it to the label. That belongs with the training code.
- Revisit the access pattern then (per-frame vs per-fixation rows, train/val
  splits, lazy vs eager) — `load_session` may need to grow options.
"""
import glob
import json
import os
from typing import List, Optional, Tuple

import pandas as pd

from scripts.eyetracker.calibration.paths import dataset_root


def session_dirs(root: Optional[str] = None) -> List[str]:
    """Every `session_<ts>/` under `root` (default `dataset_root()`), sorted."""
    base = root if root is not None else dataset_root()
    return sorted(
        d for d in glob.glob(os.path.join(base, "session_*"))
        if os.path.isdir(d)
    )


def load_session(session_dir: str) -> Tuple[Optional[dict], pd.DataFrame]:
    """Return (metadata, labels). `metadata` is None for legacy sessions with no
    metadata.json. `labels` gains a `session_id` column for joins/concats."""
    df = pd.read_csv(os.path.join(session_dir, "labels.csv"))
    df["session_id"] = os.path.basename(os.path.normpath(session_dir))
    meta_path = os.path.join(session_dir, "metadata.json")
    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return metadata, df
