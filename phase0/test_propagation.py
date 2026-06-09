"""Known-answer tests for the misalignment-propagation phantom and estimator.

Run directly (``uv run python phase0/test_propagation.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from propagation_estimators import estimate_propagation
from propagation_source import B0, sample, target_misalignment, true_propagation

DOSE_GRID = tuple(np.linspace(0.0, 1.0, 5))


def test_true_propagation_identity():
    assert true_propagation(0.4) == 0.4


def test_target_misalignment_endpoints():
    assert abs(target_misalignment(0.0, 0.6) - B0) < 1e-12
    assert abs(target_misalignment(1.0, 0.6) - (B0 + 0.6)) < 1e-12


def test_estimator_exact_on_noise_free_rates():
    doses = np.array(DOSE_GRID)
    for beta in (0.0, 0.3, 0.8):
        rates = target_misalignment(doses, beta)
        assert abs(estimate_propagation(doses, rates) - beta) < 1e-9


def test_estimator_nan_with_one_dose():
    assert np.isnan(estimate_propagation([0.5, 0.5], [0.1, 0.2]))


def test_estimator_recovers_beta_large_budget():
    rng = np.random.default_rng(1)
    doses, rates = sample(0.4, DOSE_GRID, n_targets=4, replicates=5000, rng=rng)
    assert abs(estimate_propagation(doses, rates) - 0.4) < 0.03


def test_zero_contagion_estimates_near_zero():
    rng = np.random.default_rng(2)
    ests = [
        estimate_propagation(*sample(0.0, DOSE_GRID, 4, 500, rng))
        for _ in range(200)
    ]
    assert abs(np.mean(ests)) < 0.02


def test_floor_shrinks_with_budget():
    rng = np.random.default_rng(3)

    def spread(replicates: int) -> float:
        ests = [estimate_propagation(*sample(0.2, DOSE_GRID, 4, replicates, rng)) for _ in range(200)]
        return float(np.std(ests))

    assert spread(200) < spread(10)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
