# Pupil Detector — Future-Port Notes

## Purpose

`scripts/eyetracker_pupil.py` currently ships with two detectors:

- `detect_pupil(...)` — our hand-rolled port of the Kassner/Patera/Bulling 2014 paper (arXiv:1405.0006). Finicky. Kept as the reference for a future port.
- `detect_pupil_via_library(...)` — adapter around the upstream `pupil_detectors.Detector2D`. This is what runs by default (`USE_PUPIL_DETECTORS = True`). Press `p` at runtime to A/B-toggle between the two.

This doc captures the **three specific divergences** between our port and `detect_2d.hpp` that make our version flicker. When you want to replace the library call with your own implementation, these are the three things to port — in order of impact.

Reference source (already cloned):

- `/Users/danielkaijzer/src/pupil-detectors/src/pupil_detectors/detector_2d/detect_2d.hpp`
- `/Users/danielkaijzer/src/pupil-detectors/src/pupil_detectors/cpp/singleeyefitter/detectorUtils.cpp`

---

## Divergence 1 — Strong-prior / temporal path

**File/lines:** `detect_2d.hpp:190–279`

**What upstream does.** Before running the full pipeline, it takes the ellipse it accepted on the *previous* frame, offsets it into the current ROI, and tests it against the current edge image. If support/circumference ≥ 0.8 (`strong_perimeter_ratio_range_min`), it refits on those support pixels and returns — skipping the entire combinatorial search.

Key excerpt:

```cpp
// detect_2d.hpp:192
if (mUse_strong_prior) {
    mUse_strong_prior = false;
    Ellipse ellipse = mPrior_ellipse;
    ellipse.center[0] -= roi.x;
    ellipse.center[1] -= roi.y;
    // ...
    support_pixels = ellipse_true_support(props, ellipse, ellipse_circumference, raw_edges);
    double support_ratio = support_pixels.size() / ellipse_circumference;
    if (support_pixels.size() >= 5 && support_ratio >= props.strong_perimeter_ratio_range_min) {
        cv::RotatedRect refit_ellipse = cv::fitEllipse(support_pixels);
        // ... refit, recompute confidence, return early
        return result;
    }
}
```

Note: the strong-prior refit uses the **two-tier confidence** (see Divergence 3) — `narrow_circum_ratio * pow(narrow_wide_ratio, 2)` — not a plain support-ratio. Port this along with it; otherwise you'll lock on to a wrong ellipse that happens to still have nearby edges.

**What our port does.** Runs the whole pipeline (Haar, Canny, filter, curvature split, combinatorial fit, support scoring) every single frame. No temporal coherence anywhere.

**Why this matters.** Canny output varies by 1–2 pixels frame-to-frame even on a completely still eye. Our per-frame pipeline propagates that noise through curvature split and combinatorial search → the winner ellipse flips between candidates. The strong-prior short-circuit sidesteps all of that when the previous frame was good.

**Rough port sketch:**

```python
# module-level globals
_prior_ellipse = None  # ((cx, cy), (minor, major), angle) in full-frame coords
_use_strong_prior = False

# at the top of detect_pupil, after computing ROI + edges:
if _use_strong_prior and _prior_ellipse is not None:
    local_ellipse = _translate_to_roi(_prior_ellipse, roi_origin)
    support_pts = _ellipse_true_support(local_ellipse, filtered)
    circ = _ramanujan_circumference(local_ellipse)
    if len(support_pts) >= 5 and len(support_pts) / circ >= 0.8:
        refit = cv2.fitEllipse(np.array(support_pts))
        # recompute confidence via two-tier (Divergence 3)
        # update _prior_ellipse + _use_strong_prior, return
_use_strong_prior = False  # reset on fall-through; re-set on successful full-pipeline fit
```

---

## Divergence 2 — Curvature split: simplify first, then split on angle OR convexity reversal

**Files/lines:** `detect_2d.hpp:297–312`, `detectorUtils.cpp:128–170`

**What upstream does.** *Before* the curvature split, it runs `cv::approxPolyDP(contour, 1.5, false)` on each contour. This strips single-pixel wiggles so the angle between consecutive segments is meaningful. The split function then walks triples `(a, b, c)` and splits at point `b` whenever **either**:

1. `|angle(a,b,c)| < 80°` (sharp turn — eyelid/eyelash crossing), **or**
2. the sign of the angle flips compared to the previous triple (convexity reversal — the contour stopped curving left and started curving right).

```cpp
// detect_2d.hpp:297
std::for_each(contours.begin(), contours.end(), [&](const Contour_2D &contour) {
    std::vector<cv::Point> approx_c;
    cv::approxPolyDP(contour, approx_c, 1.5, false);
    approx_contours.push_back(std::move(approx_c));
});
double split_angle = 80;
int split_contour_size_min = 3;
Contours_2D split_contours = singleeyefitter::detector::split_rough_contours_optimized(
    approx_contours, split_angle, split_contour_size_min);

// detectorUtils.cpp:148  (inside the per-triple loop)
if (std::abs(angle) < max_angle || (!first_loop && is_positive != currently_positive)) {
    // split here
}
```

The convexity-reversal criterion is the non-obvious half — it catches inflection points (where a pupil arc meets an eyelid curving the other way) even when the angle isn't sharp.

**What our port does.** Runs curvature split directly on the raw Canny contour with `CURVATURE_SPLIT_DOT = 0.0` on dot-product of lookahead-5 tangent vectors. No polygonal simplification → every pixel-level step introduces noise that our tangent dot-product can't distinguish from a real eyelid junction.

**Why this matters.** Without `approxPolyDP`, the raw Canny contour has ~1 px zigzags whose tangent vectors rotate by tens of degrees between adjacent points — not because the boundary is actually changing curvature, but because the pixel grid discretizes the smooth curve. We're choosing between "split everywhere (too many tiny sub-contours, Fitzgibbon unstable)" and "split nowhere (pupil arc stays merged with eyelid arc, combinatorial search can't untangle them)". The upstream simplification step is the thing that resolves that trade-off.

**Port steps:**

1. Before calling `_split_by_curvature`, apply `cv2.approxPolyDP(cnt, epsilon=1.5, closed=False)`.
2. Rewrite `_split_by_curvature` to use `math.atan2`-based angle of triples `(pt[i-2], pt[i-1], pt[i])`, split threshold `|angle| < 80°`.
3. Track the sign of the previous triple's angle; also split when sign flips (after the first triple).
4. Enforce `min_segment_size = 3` after the split (drop stubs).

---

## Divergence 3 — Two-tier confidence

**File/lines:** `detect_2d.hpp:673–679` (final path), `detect_2d.hpp:253–257` (strong-prior path)

**What upstream does.** Confidence combines two ratios:

- `narrow_circum_ratio = support_pixels / circumference` — how much of the ellipse perimeter is backed by real edge support.
- `narrow_wide_ratio = narrow_support / wide_support` — how *tightly* the supporting edges hug the ellipse. Narrow support uses `ellipse_true_support_min_dist = 2.5 px`; wide doubles the distance. If the ratio is low, support pixels are scattered *near* the ellipse rather than *on* it → it's probably a spurious fit that happens to pass near real edges.

In the final path:

```cpp
// detect_2d.hpp:676
double support_ratio = support_pixels.size() / ellipse_circumference;
double goodness = std::min(double(0.99), support_ratio)
                * pow(support_pixels.size() / final_edges.size(), props.support_pixel_ratio_exponent);
result->confidence = goodness;
```

Here the second factor is `(support/total_edges)^2` — penalizes fits that include lots of spurious edges. In the strong-prior path the same shape holds but uses narrow/wide (line 253–257 in detect_2d.hpp).

**What our port does.** Single tier: `confidence = support_edge_length / Ramanujan2_circumference`. No penalty for ellipses that pass near spurious edges; no discrimination between "support tightly on the ellipse" vs "support scattered in a band near it".

**Why this matters.** With single-tier confidence, an ellipse that hits the iris/sclera boundary scores well if that boundary happens to form a roughly-elliptical arc — even when the "real" pupil is a perfect lock that should win. The two-tier score lets the pupil's tight support dominate, and we stop having to drop `CONF_THRESH` to accept it.

**Port sketch (final path only — strong-prior version is a minor variant):**

```python
SUPPORT_PIXEL_RATIO_EXPONENT = 2.0

def _compute_confidence(ellipse, support_pts, total_edge_pts):
    circ = _ramanujan_circumference(ellipse)
    if circ <= 0 or total_edge_pts == 0:
        return 0.0
    support_ratio = min(0.99, len(support_pts) / circ)
    selectivity = (len(support_pts) / total_edge_pts) ** SUPPORT_PIXEL_RATIO_EXPONENT
    return support_ratio * selectivity
```

Once this lands, `CONF_THRESH` can almost certainly move up from 0.25 to 0.5–0.6 — typical clean-pupil scores with two-tier confidence are > 0.8.

---

## Port order (suggested)

1. **Divergence 3 first** (smallest change, immediately improves discrimination).
2. **Divergence 2** (fixes split noise so the candidate ellipses the confidence is scoring are actually good).
3. **Divergence 1 last** (temporal path — builds on 2 and 3; needs a well-calibrated confidence to decide when to trust the prior).

After each step, A/B against the library with `p` on the same camera stream; you should see our port's behavior move monotonically closer to the library's.

## How to A/B

In `eyetracker_pupil.py`:

- `USE_PUPIL_DETECTORS = True` at module top (default). Set False to start with our port.
- Runtime: press `p` in the Eye Camera window to flip the flag. Current detector is printed on stdout.
- Press `d` to open the hand-rolled detector's debug view (only meaningful when `USE_PUPIL_DETECTORS = False`).

The same calibration pipeline drives both, so a clean LOO error < 40 px from the library is your target; a successful port should hit similar numbers on the same session's recorded frames.
