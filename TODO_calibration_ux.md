# TODO: Calibration UX feedback (this branch: calibration-update)

Usability fixes for the live calibration overlay — they don't change the fit
or the data format. Being implemented on the calibration-update branch.

Status: ALL items implemented (2+3 subsumed by 5). Keys: 'c' capture,
's' skip, Esc abort, 'p' preview, '['/']'/'{'/'}' exposure, '-'/'=' marker
white level, 'q' quit.

## Context: how a fixation is captured today

`CalibrationRoutine.update()` (`scripts/eyetracker/calibration/routine.py`, ~L243)
runs every frame while `is_collecting` is True and **silently early-returns**
when it can't use the frame:

- `pupil_center is None or eye_frame is None` → no pupil this frame.
- `scene_frame is None` → no scene cam.
- `compute_homography(...)` returns `None` → fewer than 4 ArUco markers visible.

Samples only accumulate when none of those fire, until `SampleCollector` reaches
`CALIB_SAMPLES`. So if pupil detection (or markers) drops out, the point sits
"collecting" forever with no feedback — the bug below.

The overlay (`scripts/eyetracker/display/tk_overlay.py`, `render()`) currently
shows marker state only as bottom-of-screen text:
`color = "#00ff00" if marker_count == 4 else "#ff6060"` (L138–142). The active
dot is always `fill="red"` (L107–114) regardless of readiness.

## 1. Sample-stall timeout — DONE

**Problem:** if no new samples land for a while (pupil lost, markers dropped),
the routine gets stuck on the active point indefinitely and the user ends up
staring at the dot. Holding a fixation that long also corrupts the reading.

**Want:** if no *accepted* sample has been added for **> 5 s** while collecting,
abort this fixation, reset the collector, drop back to the idle "press 'c'"
state for the same point, and tell the user to adjust and re-press 'c'.

**Sketch:**
- Track a `last_sample_ts` (or "collecting started" ts), updated whenever a
  sample is actually added in `update()`. Reset it in `begin_capture()`.
- In `update()` (or a per-frame tick), if `is_collecting` and
  `now - last_sample_ts > CALIB_SAMPLE_TIMEOUT_S`, call the same teardown as a
  collector reject (`_discard_pending()`, `is_collecting = False`,
  `collector.reset()`) and print/flag a "timed out — adjust and press 'c'".
- New config: `CALIB_SAMPLE_TIMEOUT_S = 5.0`.
- Surface the timeout in the overlay status line so it's visible, not console-only.
- Decide whether warmup frames count toward the clock (they shouldn't — start the
  timer after `consume_warmup_frame()` stops returning True).

## 2. Active dot turns green when 4/4 markers detected — DONE via item 5

**Problem:** the user has to glance at the bottom "4/4" text to know it's safe to
hold still, which itself moves the eyes and spoils the capture.

**Want:** color the **active dot** green when 4/4 markers are detected (ready),
red otherwise — so readiness is centered on where the eyes already are.

**Sketch:** in `render()`, the active-dot branch (L107–114) chooses `fill` from
`self.target_mapper.last_marker_count == 4`. Keep the orange "collecting" ring.
Keep the existing bottom-of-screen marker text too (still useful at a glance).

## 3. Pupil-not-detected warning + dot stays non-green without a pupil — DONE via item 5

**Problem:** during calibration there's no indication when pupil detection has
dropped out — only markers are surfaced.

**Want:**
- Warning text (e.g. "Pupil not detected") above the existing "4/4 markers"
  line whenever the pupil isn't being detected.
- The active dot must **not** turn green unless *both* 4/4 markers **and** a
  pupil are present — readiness = markers AND pupil. (Combine with item 2.)

**Sketch / plumbing note:** the overlay currently has no pupil signal — it only
reads `target_mapper.last_marker_count`. A "pupil detected this frame" flag needs
to reach the overlay. Cleanest is probably to have the routine/App expose the
latest pupil-detection state (the same `pupil_center is None` check `update()`
already does) and have `render()` read it alongside `last_marker_count`. Confirm
where the pupil pipeline result lives in the App loop before wiring it.

## 4. Bind exposure hotkeys in the Tk overlay — DONE

**Problem:** scene-cam exposure cannot be adjusted during calibration at all.
`App._handle_key()` (app.py ~L264) handles `[` `]` `{` `}`, but the fullscreen
Tk overlay has keyboard focus and `tk_overlay.py` L51 only binds `c`/`s`/`q` —
the bracket presses never reach the app. So "markers blown out mid-calibration"
is currently unfixable without quitting the overlay.

**Fix:** bind `bracketleft`/`bracketright`/`braceleft`/`braceright` in
`overlay.open()`, pushing the same chars onto the key queue. `_handle_key`
already does the rest. Show `exposure_status()` in the overlay status line so
the nudge is visible without the eye-cam window.

## 5. Two status rings around the active dot (extends items 2+3) — DONE

Supersedes item 2's "dot turns green" rendering (readiness logic unchanged:
`ready = markers_ok and pupil_ok`).

- **Inner ring** (~r=30): pupil status. Signal = `App.last_pupil_center is not
  None` (already post conf-gate + jump-gate, so it covers glare/no-IR-filter
  pupil dropouts). Needs the item-3 plumbing: App stamps a flag onto the
  routine (or passes a status object) before `render()`.
- **Outer ring** (~r=40): marker status as **four quadrant arcs**
  (`canvas.create_arc`), one per corner marker (IDs 0=TL, 1=TR, 2=BR, 3=BL
  map directly to quadrants). Requires `ArucoHomography.update_marker_count()`
  to also cache `last_found_ids` (it already computes the set at L74 and
  discards it).
- **Peripheral-vision note:** red-vs-green is hard to see at 20°+ eccentricity.
  Make ready = quiet (dim/absent), failure = loud (thick bright arc) — detect
  presence/absence, not hue.
- **Progress arc:** replace the orange collecting ring with an arc,
  `extent = 360 * samples / CALIB_SAMPLES`, so capture progress is peripheral
  too.
- Bottom text can name missing markers ("missing: TL, BR") once
  `last_found_ids` exists.

## 6. Manual abort key (pairs with item 1's timeout) — DONE (Esc)

Key: Escape (or `a`), bound in the overlay. If `is_collecting`, run the same
teardown as the CollectorReject branch (routine.py ~L232): `_discard_pending()`,
`is_collecting = False`, `collector.reset()` — but do NOT advance
`current_idx`. Back to idle "press 'c'" on the same point. (`s` mid-collect
skips the point entirely; abort retries it.)

## 7. Camera preview toggle inside the overlay — DONE ('p')

Hotkey `v` toggles a downscaled (~500px wide) scene-cam preview centered on
the Tk canvas, redrawn each frame while visible.

- Encode as **PPM** (raw, no compression) → `tk.PhotoImage`; PNG per frame is
  needlessly slow. Keep a reference or Tk garbage-collects the image.
- Annotate, don't show raw: `cv2.aruco.drawDetectedMarkers` + tint pixels
  ≥250 red so blown-out regions are obvious. This is the only feedback that
  diagnoses *specular glare* (exposure/marker-level knobs don't fix glare).
- Optional eye-frame preview beside it. NB: `last_eye_frame` is deliberately
  clean for the dataset (app.py copies before the ellipse draw) — keep a
  separate annotated copy for preview.
- Showing a bright image mid-fixation ruins that capture; acceptable, this is
  a between-attempts troubleshooting mode (abort first via item 6).

## 8. ArUco marker white level — DONE ('-'/'=')

Secondary knob for marker blowout (exposure via item 4 is the primary fix,
but a display-side level persists across sessions instead of being re-tuned
each time). NB: the IR filter lives on the EYE camera — removing it affects
Pupil Labs pupil detection only, never scene exposure or marker rendering.

- Config `ARUCO_WHITE_LEVEL` (default 255). In `generate_marker_png()`:
  `img[img > 127] = white_level`.
- The quiet-zone rectangle fill (tk_overlay.py L166) must be computed to the
  SAME grey — a grey marker inside a white quiet zone is worse than either.
- ArUco detection thresholds adaptively; grey-on-black should detect down to
  ~100, but verify on this camera.
- If live-adjustable via hotkey: invalidate the `_photo_images` cache on
  change.

## Perf note — DONE

ArUco detection used to run twice per 1080p frame during collection. Now
`ArucoHomography.process_frame()` (App loop) detects once and caches
corners/ids + `last_found_ids`; the routine solves via `cached_homography()`.

## Note for item 7 (preview key choice)

`v` now starts a validation capture at the top level, but that branch is
guarded by `not routine.is_active`, so `v` is free *inside* calibration —
App can route it to the preview toggle when `routine.is_active`. Still,
consider a different key to avoid user confusion.

## Notes
- Items 2 and 3 share the "is the dot allowed to be green" logic — implement
  together: `ready = markers_ok and pupil_ok`. (Item 5 is the rendering of
  that logic.)
- All of these are overlay/feedback changes; none should touch the saved data
  format or the polynomial fit.
- Suggested build order: 4 (trivial) → 5 → 1+6 (shared teardown) → 7 → 8.
