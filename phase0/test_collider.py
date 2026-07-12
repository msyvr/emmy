"""Known-answer tests for the collider (common-effect) phantom.

These guard the closed-form ground truth and the negative-control behaviour the
phantom exists to demonstrate: plain MI correctly reads zero (no coupling
exists), while conditioning on the jointly-produced state manufactures
dependence. Run directly (``uv run python phase0/test_collider.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from collider_source import (
    channel_flip,
    conditional_joint,
    sample,
    true_conditional_mi,
    true_unconditional_mi,
    unconditional_joint,
)
from coupled_source import mi_of_joint
from mi_estimators import conditional_mi, miller_madow_mi

TOL = 0.02
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def test_conditional_joint_normalized():
    for rho in GRID:
        joints = conditional_joint(rho, 0.8, 0.6)
        assert np.allclose(joints.sum(axis=(1, 2)), 1.0)


def test_conditional_marginals_uniform():
    # Given s, each action alone is a fair coin -- the collider biases nothing
    # marginally; it only correlates the pair.
    joints = conditional_joint(0.9, 0.7, 0.8)
    assert np.allclose(joints.sum(axis=2), 0.5)   # a1 marginal per sigma
    assert np.allclose(joints.sum(axis=1), 0.5)   # a2 marginal per sigma


def test_closed_form_matches_joint():
    # 1 - H_b(phi) must equal the exact MI of the per-sigma joints, averaged
    # over P(s) = (1/2, 1/2).
    for rho1 in GRID:
        for rho2 in (0.3, 0.75, 1.0):
            for lam in GRID:
                joints = conditional_joint(rho1, rho2, lam)
                from_joint = 0.5 * (mi_of_joint(joints[0]) + mi_of_joint(joints[1]))
                assert abs(true_conditional_mi(rho1, rho2, lam) - from_joint) < 1e-12


def test_unconditional_mi_exactly_zero():
    # The truth of the phantom: no coupling exists. The marginalized joint is
    # uniform in every cell, whatever the knobs.
    for rho in GRID:
        for lam in GRID:
            joint = unconditional_joint(rho, 0.8, lam)
            assert np.allclose(joint, 0.25)
            assert abs(mi_of_joint(joint)) < 1e-12
            assert true_unconditional_mi(rho, 0.8, lam) == 0.0


def test_no_leak_without_memory():
    # Memoryless behavior (rho = 0 for either agent) leaks nothing, at any
    # state fidelity: the collider path runs THROUGH the agents' memories.
    for lam in GRID:
        assert abs(true_conditional_mi(0.0, 0.9, lam)) < 1e-12
        assert abs(true_conditional_mi(0.9, 0.0, lam)) < 1e-12


def test_no_leak_without_state_fidelity():
    # A state that does not record the joint behavior (lam = 0) induces
    # nothing: conditioning on noise is a no-op.
    for rho in GRID:
        assert abs(true_conditional_mi(rho, rho, 0.0)) < 1e-12


def test_full_leak_is_one_bit():
    # Deterministic behavior + perfect state record: conditioning manufactures
    # a full bit between two agents with no channel between them.
    assert abs(true_conditional_mi(1.0, 1.0, 1.0) - 1.0) < 1e-12


def test_product_law():
    # The manufactured dependence is governed by the single product
    # rho1*rho2*lam: any factorization of the same product gives the same value.
    assert abs(
        true_conditional_mi(0.9, 0.8, 0.5) - true_conditional_mi(0.9 * 0.8 * 0.5, 1.0, 1.0)
    ) < 1e-12
    assert abs(1.0 - 2.0 * channel_flip(0.9, 0.8, 0.5) - 0.9 * 0.8 * 0.5) < 1e-12


def test_estimators_converge_to_true():
    rng = np.random.default_rng(1)
    rho, lam = 0.75, 0.8
    s, a1, a2 = sample(rho, rho, lam, 200_000, rng)
    assert abs(conditional_mi(s, a1, a2, 2, 2, corrected=True)
               - true_conditional_mi(rho, rho, lam)) < TOL
    assert abs(miller_madow_mi(a1, a2, 2)) < TOL


def test_manufactured_dependence_at_finite_n():
    # The headline behavioural check, mirroring the confound phantom's: with
    # persistent behavior and a faithful joint state, the conditional-MI
    # estimate fires while the plain-MI estimate sits at its floor -- and here
    # the plain estimate is the RIGHT one. Across independent re-samples.
    rng = np.random.default_rng(7)
    rho, lam, n = 0.9, 1.0, 2000
    mi = np.empty(200)
    cmi = np.empty(200)
    for r in range(200):
        s, a1, a2 = sample(rho, rho, lam, n, rng)
        mi[r] = miller_madow_mi(a1, a2, 2)
        cmi[r] = conditional_mi(s, a1, a2, 2, 2, corrected=True)
    assert cmi.mean() > 0.2            # conditioning manufactures a clear signal
    assert abs(mi.mean()) < 0.02       # plain MI reads the truth: no coupling
    assert cmi.mean() > 10 * abs(mi.mean())


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
