"""Regression guard for the polynomial mapper's leave-one-out (LOO) error.

LOO has silently regressed to a `np.zeros(n)` stub twice during degree-2/3
refactors (see docs/loo_error_notes.md). The fit itself stayed correct, so live
tracking worked — the breakage only showed up in the quality machinery
(recapture targeting, the high-error warning, and measure_gaze_accuracy, which
reports the eccentricity-binned LOO table). These asserts fail loudly if the
refit loop ever disappears again.

No pytest dependency: this uses plain asserts and runs standalone with
    python -m tests.test_polynomial_loo
(from the repo root). The functions are named test_* so pytest can also collect
them if the project adopts it later.
"""
import numpy as np

from scripts.eyetracker.gaze.polynomial import PolynomialGazeMapper


def _noisy_quadratic(rng, n):
    """n pupil points and a known degree-2 scene surface + Gaussian noise, so a
    correct LOO must be strictly positive (the noise is not fittable)."""
    pupil = rng.uniform(-1, 1, (n, 2))
    scene = np.column_stack([
        3 + 2 * pupil[:, 0] - pupil[:, 1] + 0.5 * pupil[:, 0] ** 2,
        1 - pupil[:, 0] + 2 * pupil[:, 1] + 0.3 * pupil[:, 1] ** 2,
    ]) + rng.normal(0, 0.05, (n, 2))
    return pupil, scene


def test_loo_nonzero_on_noisy_data_degree2():
    """The core guard: LOO must not be identically zero on noisy data."""
    rng = np.random.default_rng(0)
    pupil, scene = _noisy_quadratic(rng, 12)
    report = PolynomialGazeMapper(degree=2).fit(pupil, scene)
    assert report.loo_avg_err > 0, "degree-2 LOO avg is zero — LOO is stubbed"
    assert report.loo_max_err > 0
    assert np.any(report.per_point_errs > 0)
    assert report.per_point_errs.shape == (12,)


def test_loo_nonzero_on_noisy_data_degree3():
    """Same guard at degree 3 (10 coefficients, so use more points)."""
    rng = np.random.default_rng(1)
    pupil, scene = _noisy_quadratic(rng, 20)
    report = PolynomialGazeMapper(degree=3).fit(pupil, scene)
    assert report.loo_avg_err > 0, "degree-3 LOO avg is zero — LOO is stubbed"
    assert np.any(report.per_point_errs > 0)
    assert report.per_point_errs.shape == (20,)


def test_loo_near_zero_on_clean_polynomial():
    """Positive control: noise-free data lying exactly on a degree-2 surface is
    perfectly fit, so every held-out prediction is near-exact and LOO ~ 0.
    (Guards against a fix that just returns a nonzero constant.)"""
    rng = np.random.default_rng(2)
    pupil = rng.uniform(-1, 1, (12, 2))
    scene = np.column_stack([
        3 + 2 * pupil[:, 0] - pupil[:, 1] + 0.5 * pupil[:, 0] ** 2,
        1 - pupil[:, 0] + 2 * pupil[:, 1] + 0.3 * pupil[:, 1] ** 2,
    ])
    report = PolynomialGazeMapper(degree=2).fit(pupil, scene)
    assert report.loo_max_err < 1e-6, (
        f"clean degree-2 data should fit near-exactly, got LOO max "
        f"{report.loo_max_err}")


def _run() -> None:
    tests = [
        test_loo_nonzero_on_noisy_data_degree2,
        test_loo_nonzero_on_noisy_data_degree3,
        test_loo_near_zero_on_clean_polynomial,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} passed.")


if __name__ == "__main__":
    _run()
