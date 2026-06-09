"""Known-answer tests for the Phase 0 phantom and estimators.

Run directly (``uv run python phase0/test_phase0.py``) or via pytest. These
guard the load-bearing computations: if the closed-form true value or the
estimator is wrong, the whole calibration is meaningless.
"""

from __future__ import annotations

import numpy as np

from coupled_source import (
    conditional_joint,
    mi_of_joint,
    sample,
    true_conditional_mi,
)
from mi_estimators import conditional_mi

P_S = (0.5, 0.5)
K = 4
LOG2_K = 2.0  # log2(4)
TOL = 0.02


def test_true_cmi_zero_at_no_coupling():
    assert abs(true_conditional_mi((0.0, 0.0), P_S, K)) < 1e-12


def test_true_cmi_max_at_full_coupling():
    # kappa = 1 -> a2 == a1 -> I(a1; a2 | s) = log2(K).
    assert abs(true_conditional_mi((1.0, 1.0), P_S, K) - LOG2_K) < 1e-12


def test_true_cmi_monotone_in_coupling():
    vals = [true_conditional_mi((k, k), P_S, K) for k in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_true_cmi_averages_over_context():
    # Heterogeneous coupling: conditional MI is the P(s)-weighted per-context MI.
    expected = 0.5 * 0.0 + 0.5 * LOG2_K
    assert abs(true_conditional_mi((0.0, 1.0), P_S, K) - expected) < 1e-12


def test_conditional_joint_is_normalized():
    joints = conditional_joint((0.3, 0.8), K)
    assert np.allclose(joints.sum(axis=(1, 2)), 1.0)


def test_mi_of_joint_independent_is_zero():
    assert abs(mi_of_joint(np.full((K, K), 1.0 / K**2))) < 1e-12


def test_mi_of_joint_identity_is_log_k():
    assert abs(mi_of_joint(np.eye(K) / K) - LOG2_K) < 1e-12


def test_estimators_converge_to_true():
    rng = np.random.default_rng(1)
    kappa_s = np.full(2, 0.6)
    true = true_conditional_mi(kappa_s, P_S, K)
    s, a1, a2 = sample(kappa_s, P_S, K, 200_000, rng)
    assert abs(conditional_mi(s, a1, a2, K, 2, corrected=True) - true) < TOL
    assert abs(conditional_mi(s, a1, a2, K, 2, corrected=False) - true) < TOL


def test_plugin_overestimates_at_independence():
    # The classic finite-sample MI bias is positive: at true CMI = 0 the plug-in
    # estimate is reliably above zero at small N.
    rng = np.random.default_rng(3)
    vals = np.array([
        conditional_mi(*sample(np.zeros(2), P_S, K, 100, rng), K, 2, corrected=False)
        for _ in range(300)
    ])
    assert vals.mean() > 0.05


def test_miller_madow_reduces_bias_small_n():
    rng = np.random.default_rng(2)
    kappa_s = np.full(2, 0.3)
    true = true_conditional_mi(kappa_s, P_S, K)
    plug = np.empty(300)
    mm = np.empty(300)
    for r in range(300):
        s, a1, a2 = sample(kappa_s, P_S, K, 100, rng)
        plug[r] = conditional_mi(s, a1, a2, K, 2, corrected=False)
        mm[r] = conditional_mi(s, a1, a2, K, 2, corrected=True)
    assert abs(mm.mean() - true) < abs(plug.mean() - true)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
