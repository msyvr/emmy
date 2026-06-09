"""Known-answer tests for the fragility phantom and curvature estimator.

Run directly (``uv run python phase0/test_fragility.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from curvature_estimators import estimate_curvature
from fragility_source import response, sample, true_curvature

SIGMA = tuple(np.linspace(0.0, 1.0, 9))


def test_true_curvature_identity():
    assert true_curvature(-1.5) == -1.5


def test_response_at_operating_point_is_r0():
    assert abs(response(0.5, kappa=2.0, r0=0.5) - 0.5) < 1e-12


def test_estimator_exact_on_noise_free_quadratic():
    sigma = np.array(SIGMA)
    for kappa in (-2.0, 0.0, 1.5):
        y = response(sigma, kappa)
        assert abs(estimate_curvature(sigma, y) - kappa) < 1e-9


def test_estimator_nan_with_too_few_levels():
    assert np.isnan(estimate_curvature([0.0, 1.0], [0.1, 0.2]))


def test_estimator_recovers_kappa_at_large_budget():
    rng = np.random.default_rng(1)
    sigma, y = sample(-1.0, SIGMA, replicates=500, noise=0.05, rng=rng)
    assert abs(estimate_curvature(sigma, y) - (-1.0)) < 0.05


def test_sign_recovered_for_fragile_and_antifragile():
    rng = np.random.default_rng(2)
    s_f, y_f = sample(-1.5, SIGMA, 200, 0.05, rng)
    s_a, y_a = sample(1.5, SIGMA, 200, 0.05, rng)
    assert estimate_curvature(s_f, y_f) < 0  # fragile
    assert estimate_curvature(s_a, y_a) > 0  # antifragile


def test_floor_shrinks_with_budget():
    rng = np.random.default_rng(3)

    def spread(replicates: int) -> float:
        ests = [estimate_curvature(*sample(-1.0, SIGMA, replicates, 0.05, rng)) for _ in range(200)]
        return float(np.std(ests))

    assert spread(50) < spread(3)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
