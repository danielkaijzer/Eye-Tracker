# Claude Guidelines

## Git & Version Control
- Always check with me before committing or pushing. Proposing it is fine (suggest the commit and message, then ask), but don't run it until I say yes.
- **Never add attribution to commits or PRs**: no `Co-Authored-By: Claude` trailer and no "Generated with Claude Code" line. I direct the work, so it's credited to me.

## Environment
- Python env: conda env `et`. Tests: `python -m pytest tests`.
- The rig runs on a Jetson Orin (Linux, V4L2). macOS still works for development;
  macOS-only paths (uvc-util exposure control) are marked as such.

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
