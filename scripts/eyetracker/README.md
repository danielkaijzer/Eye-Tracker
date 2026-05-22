# Eye Tracker — Backend

Python pipeline that detects the pupil, calibrates a polynomial mapping from pupil pixels to scene-camera pixels, and either renders the gaze locally (OpenCV) or streams annotated frames over HTTP for the Next.js dashboard to overlay.

See the [root README](../../README.md) for the full system overview.

## Install

Python 3.11+ recommended.

```
brew install eigen opencv                       # macOS system deps for pupil-detectors
git clone https://github.com/pupil-labs/pupil-detectors.git ../../../pupil-detectors
pip install -r ../../requirements.txt           # installs the local pupil-detectors clone
```

`requirements.txt` references `../pupil-detectors` as a local path relative to the repo root; adjust the clone location or edit the path if your layout differs. `pye3d` ships from PyPI.

**Hardware** — head-mounted rig with an IR eye camera and a forward-facing scene camera (USB UVC). Calibration draws four ArUco markers (`DICT_4X4_50`, IDs 0/1/2/3) directly onto the laptop screen — nothing to print or mount.

## Run

From the repo root:

```
py -m scripts.eyetracker            # local cv2 windows
py -m scripts.eyetracker --web      # serves MJPEG endpoints for the dashboard
```

In web mode, run the Next.js frontend in a separate terminal (see [`frontend/README.md`](../../frontend/README.md)).

## In-app controls

| Key | Action |
| --- | --- |
| `c` | Quick calibration (4×3 grid, degree-2 polynomial) |
| `d` | Detailed calibration (5×4 grid, degree-3 polynomial, with worst-point recapture) |
| `l` | Load most recent saved calibration |
| `r` | Reset the pye3d 3D pupil model (give it ~30 s to reconverge) |
| `space` | Pause |
| `q` | Quit |

Calibration displays four ArUco markers (IDs 0/1/2/3, `DICT_4X4_50`) at the corners of the laptop screen — they let the routine project each target's screen pixel into the scene camera to manufacture training labels.

## Layout

```
__main__.py             # Composition root: wires concrete classes into App
app.py                  # Main loop, frame routing, key dispatch
config.py               # All tunables (camera, calibration grid, smoother, ArUco)
cameras/                # OpenCV camera sources + discovery
pupil/                  # Pupil Labs detector + confidence/jump gates
scene/                  # ArUco detection and screen→scene homography
gaze/                   # Polynomial mapper, 1€ smoother
calibration/            # State machine, sample collector, persistence
display/                # Tk calibration overlay, cv2 windows, Flask MJPEG server
```

See [`docs/polynomial_gaze_mapping.md`](../../docs/polynomial_gaze_mapping.md) for the math behind the pupil→scene fit and why the homography only shows up during calibration.

## Code style

[PEP 8](https://peps.python.org/pep-0008/), enforced by `flake8` in CI (`.github/workflows/linter.yml`): blocking on `E9`/`F63`/`F7`/`F82` (syntax errors, undefined names), with line length 127 and McCabe complexity 10 as non-blocking warnings. Public-facing modules, classes, and functions carry docstrings.
