"""Known-answer tests for the discriminant-validity (confound) phantom.

These guard the closed-form ground truth and the discriminant behaviour the
phantom exists to demonstrate: conditional MI rejects a pure common cause, plain
MI does not. Run directly (``uv run python phase0/test_confound.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from confound_source import (
    conditional_joint,
    context_marginal,
    sample,
    true_conditional_mi,
    true_unconditional_mi,
)
from coupled_source import true_conditional_mi as coupled_true_cmi
from mi_estimators import conditional_mi, miller_madow_mi

P_S = (0.5, 0.5)
PREFERRED = (0, 1)          # distinct -> the context is a common cause
K = 4
TOL = 0.02


def test_marginal_normalized():
    mu = context_marginal(0.7, PREFERRED, K)
    assert np.allclose(mu.sum(axis=1), 1.0)


def test_conditional_joint_normalized():
    joints = conditional_joint(0.6, 0.4, PREFERRED, K)
    assert np.allclose(joints.sum(axis=(1, 2)), 1.0)


def test_joint_marginals_match_context_marginal():
    # Both a1 and a2 must carry the same context marginal mu_s.
    kappa_cc, kappa_link = 0.6, 0.4
    mu = context_marginal(kappa_cc, PREFERRED, K)
    joints = conditional_joint(kappa_cc, kappa_link, PREFERRED, K)
    assert np.allclose(joints.sum(axis=2), mu)   # a1 marginal per context
    assert np.allclose(joints.sum(axis=1), mu)   # a2 marginal per context


def test_cmi_zero_whenever_no_link():
    # The core specificity property: no genuine link -> conditional MI is exactly
    # zero, for ANY common-cause strength.
    for kappa_cc in (0.0, 0.3, 0.6, 0.9, 1.0):
        assert abs(true_conditional_mi(kappa_cc, 0.0, PREFERRED, P_S, K)) < 1e-12


def test_unconditional_mi_positive_under_pure_common_cause():
    # kappa_link = 0 but kappa_cc > 0 with distinct preferred actions: plain MI is
    # fooled (positive) while conditional MI (above) is zero. This is the confound.
    mi = true_unconditional_mi(0.9, 0.0, PREFERRED, P_S, K)
    assert mi > 0.1
    assert true_conditional_mi(0.9, 0.0, PREFERRED, P_S, K) < 1e-12


def test_nothing_in_nothing_out():
    assert abs(true_unconditional_mi(0.0, 0.0, PREFERRED, P_S, K)) < 1e-12
    assert abs(true_conditional_mi(0.0, 0.0, PREFERRED, P_S, K)) < 1e-12


def test_no_common_cause_means_mi_equals_cmi():
    # kappa_cc = 0 -> uniform marginals -> no common cause -> I(a1;a2) == I(a1;a2|s).
    for kappa_link in (0.0, 0.3, 0.7, 1.0):
        mi = true_unconditional_mi(0.0, kappa_link, PREFERRED, P_S, K)
        cmi = true_conditional_mi(0.0, kappa_link, PREFERRED, P_S, K)
        assert abs(mi - cmi) < 1e-12


def test_reduces_to_coupled_source_at_no_common_cause():
    # At kappa_cc = 0 the phantom is exactly coupled_source with kappa = kappa_link.
    for kappa_link in (0.0, 0.4, 0.8, 1.0):
        here = true_conditional_mi(0.0, kappa_link, PREFERRED, P_S, K)
        there = coupled_true_cmi((kappa_link, kappa_link), P_S, K)
        assert abs(here - there) < 1e-12


def test_estimators_converge_to_true():
    rng = np.random.default_rng(1)
    kappa_cc, kappa_link = 0.6, 0.5
    true_cmi = true_conditional_mi(kappa_cc, kappa_link, PREFERRED, P_S, K)
    true_mi = true_unconditional_mi(kappa_cc, kappa_link, PREFERRED, P_S, K)
    s, a1, a2 = sample(kappa_cc, kappa_link, PREFERRED, P_S, K, 200_000, rng)
    assert abs(conditional_mi(s, a1, a2, K, 2, corrected=True) - true_cmi) < TOL
    assert abs(miller_madow_mi(a1, a2, K) - true_mi) < TOL


def test_discriminant_behaviour_at_finite_n():
    # The headline behavioural check: under a strong pure common cause, the plain-MI
    # estimate fires while the conditional-MI estimate sits near zero -- across
    # independent re-samples, not just in the closed form.
    rng = np.random.default_rng(7)
    kappa_cc, kappa_link, n = 0.9, 0.0, 2000
    mi = np.empty(200)
    cmi = np.empty(200)
    for r in range(200):
        s, a1, a2 = sample(kappa_cc, kappa_link, PREFERRED, P_S, K, n, rng)
        mi[r] = miller_madow_mi(a1, a2, K)
        cmi[r] = conditional_mi(s, a1, a2, K, 2, corrected=True)
    assert mi.mean() > 0.2          # plain MI clearly fooled by the common cause
    assert abs(cmi.mean()) < 0.05   # conditional MI rejects it (near the floor)
    assert mi.mean() > 10 * abs(cmi.mean())


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
