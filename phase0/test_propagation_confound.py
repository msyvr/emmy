"""Known-answer tests for the propagation discriminant (confound) phantom.

Guards the closed forms and the discriminant behaviour: a shared trigger inflates
the naive contagion slope while the conditioned slope reads zero. Run directly
(``uv run python phase0/test_propagation_confound.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from propagation_confound_source import (
    sample,
    true_conditioned_slope,
    true_naive_slope,
)
from propagation_estimators import conditioned_contagion_slope, naive_contagion_slope

N_STRATA = 2
TOL = 0.02


def test_true_conditioned_is_kappa_link():
    for klink in (0.0, 0.15, 0.3):
        assert abs(true_conditioned_slope(klink) - klink) < 1e-12


def test_naive_equals_genuine_without_trigger():
    # kappa_cc = 0 -> no common cause -> naive slope == genuine contagion.
    for klink in (0.0, 0.1, 0.2, 0.3):
        assert abs(true_naive_slope(0.0, klink) - klink) < 1e-12


def test_naive_slope_fooled_by_pure_trigger():
    # kappa_link = 0 but kappa_cc > 0: the shared trigger looks like spreading
    # (naive slope positive) while the genuine contagion is zero.
    assert true_naive_slope(0.3, 0.0) > 0.05
    assert abs(true_conditioned_slope(0.0)) < 1e-12


def test_conditioned_estimator_zero_under_pure_trigger():
    rng = np.random.default_rng(0)
    s, zs, zt = sample(0.3, 0.0, 200_000, rng)
    assert abs(conditioned_contagion_slope(s, zs, zt, N_STRATA)) < TOL


def test_estimators_converge_to_true():
    rng = np.random.default_rng(1)
    kcc, klink = 0.2, 0.2
    s, zs, zt = sample(kcc, klink, 300_000, rng)
    assert abs(naive_contagion_slope(zs, zt) - true_naive_slope(kcc, klink)) < TOL
    assert abs(conditioned_contagion_slope(s, zs, zt, N_STRATA) - klink) < TOL


def test_discriminant_behaviour_at_finite_n():
    # The headline behavioural check: under a strong pure trigger the naive slope
    # fires while the conditioned slope sits near zero, across re-samples.
    rng = np.random.default_rng(7)
    kcc, klink, n = 0.3, 0.0, 3000
    naive = np.empty(200)
    cond = np.empty(200)
    for r in range(200):
        s, zs, zt = sample(kcc, klink, n, rng)
        naive[r] = naive_contagion_slope(zs, zt)
        cond[r] = conditioned_contagion_slope(s, zs, zt, N_STRATA)
    assert naive.mean() > 0.05          # shared trigger looks like spreading
    assert abs(cond.mean()) < 0.03      # conditioned rejects it
    assert naive.mean() > 3 * abs(cond.mean())


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
