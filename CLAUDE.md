# Claude Guidelines

## Git & Version Control
- Use your judgment on committing and pushing to feature branches once a change is complete and verified; ask if unsure.
- Always ask before pushing to `main`, force-pushing, rewriting pushed history, or deleting branches. Merging PRs is mine.
- **Never add attribution to commits or PRs**: no `Co-Authored-By: Claude` trailer and no "Generated with Claude Code" line. I direct the work, so it's credited to me.

## Environment
- Python env: conda env `et`. Tests: `python -m pytest tests`.
- Linux (Jetson Orin, V4L2) is the research platform: data collection and
  anything timing-sensitive. macOS is a best-effort live-demo mode (quick
  calibration + live gaze). Research features may be Linux-only; keep macOS
  code inside the existing `IS_LINUX` branches and `cameras/uvc_util.py`, and
  don't add new macOS-specific features.

## Project context
- Gaze mapping today is a 2D polynomial fit. The roadmap goes straight to a CNN
  (eye image in, gaze point out); there is no 3D-model step. Don't suggest
  polynomial tuning (degree, grid size), since it has been shown not to help.
  Current focus is data collection whose sessions stay usable across rig changes.
- Eye cam: Arducam OV9281 (`0x0c45:0x6366`, serial UC762; that VID:PID is shared
  with the retired Sonix cam). Scene cam: `0x0bda:0xd565`.
- `TODO.md` is the running backlog. Where things are documented:
  `docs/dataset_format.md` (on-disk format), `3d-files/README.md` (mounts + jig),
  `3d-files/MEASUREMENTS.md` (hardware dimensions).
