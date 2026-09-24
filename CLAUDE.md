# Claude Guidelines

## Git & Version Control
- Use your judgment on committing and pushing to feature branches once a change is complete and verified; ask if unsure.
- Always ask before pushing to `main`, force-pushing, rewriting pushed history, or deleting branches. Merging PRs is mine.
- After pushing a new feature branch, open its PR with `gh pr create` (concise title, description of what changed and why, how it was verified). As the branch grows, update the description with `gh pr edit`. Never merge.
- **Never add attribution to commits or PRs**: no `Co-Authored-By: Claude` trailer and no "Generated with Claude Code" line. I direct the work, so it's credited to me.

## Working with me
- The physical rig sometimes runs ahead of the repo (e.g., the calibration jig and
  the eye cam's LED both changed without doc updates). When I've recently changed
  or plan to change some hardware, confirm related assumptions from docs or code
  with me before building on them. Otherwise use your judgment; don't ask every time.
- Keep the repo lean: prefer deleting dead code and stale docs (git keeps the
  history) to keeping them around, but don't make changes for their own sake.
- Process and organization suggestions are welcome when they'd clearly pay off:
  a sentence or two, not a tangent.

## TODO tracking
- `TODO.md` is the single backlog (big features get a `TODO_<topic>.md` spec).
  Read it at the start of a session.
- Update it at milestones: after a commit that finishes or changes tracked work,
  and after a PR merges. Delete items whose PR merged, fix items the work made
  stale, add follow-ups discovered along the way, and keep its short "Next up"
  list (the first thing I read after a break) matching reality.
- Before deleting an item, move any finding worth keeping into the relevant doc.
- `TODO.md` changes on several branches at once: keep edits to the sections your
  work touches, and check open branches for conflicts before pushing.
- If unsure whether an item is still open, ask.

## Findings
- `docs/findings.md` is the cross-session, cross-machine record of what we've
  measured, decided, or learned the hard way. Before working in an area, read its
  section there.
- At milestones (same as TODO), add new findings. If a change contradicts one,
  mark it Superseded (with the date) in the same commit rather than deleting it.
- Project knowledge goes in the repo (findings, topic docs), not in Claude's
  local memory, which doesn't carry across machines.

## Environment
- Python env: conda env `et`. Tests: `python -m pytest tests`.
- Linux (Jetson Orin, V4L2) is the research platform: data collection and
  anything timing-sensitive. macOS is a best-effort live-demo mode (quick
  calibration + live gaze). Research features may be Linux-only; keep macOS
  code inside the existing `IS_LINUX` branches and `cameras/uvc_util.py`, and
  don't add new macOS-specific features.

## Project context
- Roadmap: 2D polynomial fit → CNN (eye image in, gaze point out); there is no
  3D-model step. Don't suggest polynomial tuning (degree, grid size), since it has
  been shown not to help.
- Recorded sessions must stay usable across rig changes (cameras, mounts, frame
  rates), so they carry their own calibration metadata.
- Eye cam: Arducam OV9281 (`0x0c45:0x6366`, serial UC762; that VID:PID is shared
  with the retired Sonix cam). Scene cam: `0x0bda:0xd565`.
- Solo project since 2026-05-23; earlier commits may be from the prior
  contributors listed in the README.
- Where things are documented: `docs/dataset_format.md` (on-disk format),
  `3d-files/README.md` (mounts + jig), `3d-files/MEASUREMENTS.md` (hardware
  dimensions).
