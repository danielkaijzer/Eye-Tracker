# TODO: Multi-pose calibration coverage + guidance clarity

Two follow-up ideas for why multi-pose ('m') calibration gets hard at extreme
head angles. Not started yet: this is a working spec to pick up later. It
builds on the calibration UX work (merged in PR #28) and on the undistorted
homography (`use-intrinsics-for-homography`, see `TODO.md`).

## Root cause for this doc's two items

Scene cam horizontal FOV is genuinely narrow: `fx=1894.7px` at 1920px width →
`2·atan(960/1894.7) ≈ 54°` (computed from `scene_intrinsics.json`). Neon's
scene cam is 130°+ specifically to avoid this class of problem. Multi-pose
calibration (`'m'`) asks the user to turn their head while keeping all 4
corner ArUco markers in frame simultaneously — past some turn angle that's a
hard physical constraint, not a tuning problem.

Also worth remembering: `'d'` (detailed) is single-pose, eyes-only — no head
movement is supposed to happen there at all. If "what angle do I look at" is
being asked about `'d'`, the actual answer is "don't turn your head"; these
two items are about `'m'` specifically.

## 1. Tie multi-pose guidance to the live marker HUD (cheap)

**Problem:** `CALIB_POSE_GUIDANCE` (`config.py`) gives blind qualitative
instructions — `"turn your head LEFT, keep looking at the screen"` — with no
feedback for how far is far enough, or too far. The user has to guess, then
find out only when a fixation stalls/times out that they overshot.

**Fix:** `ArucoHomography.last_found_ids` and `last_marker_count` (added in
`calibration-update`) already drive the overlay's per-corner status rings.
The pose-break screen (`TkCalibrationOverlay._render_pose_break`, and
`CalibrationRoutine.next_pose_guidance` in `routine.py`) could reuse that
same live signal instead of static text — e.g. append "(currently Y/4
markers visible)" to the guidance string during the pose break, or phrase the
guidance itself around the feedback: "turn left until a marker just starts to
drop, then ease back." This makes the existing rings double as the angle
guide instead of adding new UI.

Small, self-contained, no architecture change — good first PR for this file.

## 2. Don't require the 4 specific corner markers (bigger lift)

**Problem:** `compute_homography`/`_homography_from` hard-require all 4 of
`ARUCO_IDS` (`aruco_homography.py`, `_homography_from`: `if any(i not in
id_to_center for i in self.marker_ids): return None, None`). At a real head
turn, one corner marker physically leaves the narrow-FOV frame before the
others — so multi-pose's actual usable turn range is capped by whichever
corner marker clips first, not by the lens' overall FOV.

**Fix direction:** add markers at edge midpoints (8 total anchors around the
screen border, in `screen_anchor_points()`/`quiet_zone_origins()`), and let
the homography solve from any 4+ visible, well-spread markers rather than a
fixed set of 4. This raises the actual usable head-turn range independent of
the lens — the constraint becomes "4 of 8 markers visible" instead of "4 of
4," which is a much easier bar to clear at a steep angle.

**Scope/complexity, be honest about this before starting:**
- Marker layout changes touch `config.py` (`ARUCO_IDS`, quiet zone
  geometry), `screen_anchor_points()`/`quiet_zone_origins()`, the Tk
  overlay's corner-drawing code (`_draw_aruco_corners`), and probably the
  ring HUD's quadrant-arc mapping (`_draw_active_indicators` in
  `tk_overlay.py` currently hardcodes 4 quadrants from `ARUCO_IDS`).
- `cv2.findHomography` wants ≥4 points but doesn't care which corners they
  come from — the real work is: (a) picking a "well-spread enough" subset
  policy (4 markers clustered in one half of the screen give a much worse
  extrapolated fit than 4 spread around the perimeter — this is the same
  class of edge-bias problem the undistortion branch just fixed, so don't
  reintroduce it here), and (b) deciding what "good enough spread" means
  quantitatively.
- Alternative considered and rejected for now: solvePnP from a single
  marker's 4 corners (no need for multiple markers) — noisier, since one
  small marker's corners are tightly clustered in a small image region and
  any pixel noise blows up over the full extrapolated homography. Multiple
  spread-out markers is the more robust direction.

Not scoped into a branch yet — needs its own design pass on the "which
markers, how many, how to pick a well-spread subset" question before writing
code.
