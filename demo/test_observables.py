"""Known-answer tests for the demo observables.

Run directly (``uv run python demo/test_observables.py``) or via pytest.
These guard the load-bearing measurement computations: if the estimators are
wrong, the invariance claim is meaningless regardless of the experiment.
"""

from __future__ import annotations

import numpy as np

from observables import (
    action_autocorrelation,
    joint_return,
    pairwise_mutual_information,
    per_step_reward_variance,
)

RNG = np.random.default_rng(0)
TOL = 0.02


def test_mi_independent_is_zero():
    # Two independent fair coins -> I = 0 (up to sampling noise).
    a1 = RNG.integers(0, 2, size=200_000)
    a2 = RNG.integers(0, 2, size=200_000)
    assert abs(pairwise_mutual_information(a1, a2)) < TOL


def test_mi_identical_is_one_bit():
    # a2 == a1, fair coin -> I = H(a1) = 1 bit.
    a1 = RNG.integers(0, 2, size=200_000)
    assert abs(pairwise_mutual_information(a1, a1.copy()) - 1.0) < TOL


def test_mi_anticorrelated_is_one_bit():
    # a2 = 1 - a1 is perfectly dependent -> I = 1 bit.
    a1 = RNG.integers(0, 2, size=200_000)
    assert abs(pairwise_mutual_information(a1, 1 - a1) - 1.0) < TOL


def test_autocorr_alternating_is_minus_one():
    seq = np.tile([0, 1], 500)
    assert abs(action_autocorrelation(seq) - (-1.0)) < TOL


def test_autocorr_constant_is_one():
    assert action_autocorrelation(np.ones(500)) == 1.0


def test_autocorr_iid_is_near_zero():
    seq = RNG.integers(0, 2, size=200_000)
    assert abs(action_autocorrelation(seq)) < TOL


def test_reward_quantities_scale_as_expected():
    # Rescaling rewards by c: mean scales by c, variance by c**2.
    base = RNG.normal(3.0, 1.5, size=100_000)
    c = 2.0
    assert abs(joint_return(c * base) - c * joint_return(base)) < 0.05
    ratio = per_step_reward_variance(c * base) / per_step_reward_variance(base)
    assert abs(ratio - c**2) < 0.05


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
