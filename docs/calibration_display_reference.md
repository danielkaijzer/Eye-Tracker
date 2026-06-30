# Calibration Display Reference

Physical dimensions of the screen used to present calibration targets. Needed
to convert on-screen marker positions into **metric** object points so the
scene camera can recover the target plane's 3D pose (and therefore fixation
**depth**) via `solvePnP`, not just the 2D screen→scene homography.

> Not used yet. The current pipeline only runs a 2D screen→scene homography,
> which carries no depth. This is recorded for the future data-collection
> pipeline (see `docs/data_collection.md`), where depth lets us place each
> target as a 3D point in the eye-camera frame for device-agnostic labels.

## Rig display — 15-inch MacBook Air (M3, 2024)

| Property | Value |
| --- | --- |
| Panel | 15.3″ Liquid Retina |
| Native resolution | 2880 × 1864 px |
| Pixel density | 224 ppi |
| **Active area** | **≈ 326.6 × 211.4 mm** (12.86 × 8.32 in) |

Active area = native pixels ÷ 224 ppi. Verify against Apple's tech specs, or
measure the lit area directly with calipers if you want sub-mm accuracy.

## ⚠️ Logical points vs. native pixels (read before using these numbers)

The active-area mm above map to the **native** 2880 × 1864 grid. But calibration
markers are drawn in whatever coordinate space the windowing layer reports — on
a Retina Mac that is typically **logical points**, not native pixels. The 15″
Air's default "looks like" scaled mode is **1512 × 982**, i.e. a scale factor of
~1.90 (2880 / 1512), and it is *not* exactly 2×.

So the metric scale must be computed in the **same coordinate space the
homography anchor points live in**:

```
mm_per_unit = active_area_mm / (screen size in the space markers are drawn in)
```

Whoever wires up PnP depth must either:
- read the screen size in the same units the markers are placed in
  (`ArucoHomography.set_screen_size` / `screen_anchor_points`), or
- apply the Retina backing-scale factor explicitly.

Confirm the actual scaled resolution in use (System Settings → Displays) before
trusting any fixed scale factor — it's user-configurable.
