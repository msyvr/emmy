"""Known-answer tests for calibration-curve inversion (the cross-environment move).

These guard the closed-form curve, the inversion round-trip, and the behaviour
the demo exists to show: raw readings differ across environments at the same
coupling (the trap), inverted estimates agree (the fix). Run directly
(``uv run python phase0/test_inversion.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from coupled_source import sample, true_conditional_mi
from inversion import analytic_curve, calibration_curve, invert_reading
from mi_estimators import conditional_mi

LEAN = dict(n_actions=2, p_s=(0.5, 0.5))
RICH = dict(n_actions=8, p_s=(0.5, 0.3, 0.2))
GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)


def _true_reading(kappa: float, n_actions: int, p_s) -> float:
    kappa_s = np.full(len(p_s), kappa)
    return true_conditional_mi(kappa_s, p_s, n_actions)


def test_analytic_curve_matches_coupled_source():
    # Two independent derivations of the same curve: the closed-form expression
    # vs the exact-joint computation in coupled_source (context mixture drops
    # out because kappa is the same in every context).
    for K, p_s in ((2, (0.5, 0.5)), (4, (0.5, 0.5)), (8, (0.5, 0.3, 0.2))):
        for kappa in GRID:
            assert abs(analytic_curve(kappa, K) - _true_reading(kappa, K, p_s)) < 1e-12


def test_curve_endpoints():
    for K in (2, 4, 8):
        assert abs(analytic_curve(0.0, K)) < 1e-12
        assert abs(analytic_curve(1.0, K) - np.log2(K)) < 1e-12


def test_curve_monotone():
    for K in (2, 8):
        grid, readings = calibration_curve(K)
        assert np.all(np.diff(readings) >= 0.0)
        assert readings[-1] > readings[0]


def test_inversion_round_trip():
    # invert(f(kappa)) == kappa on the interior, to interpolation resolution.
    for K in (2, 8):
        grid, readings = calibration_curve(K)
        for kappa in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            khat = invert_reading(analytic_curve(kappa, K), grid, readings)
            assert abs(khat - kappa) < 1e-4


def test_inversion_clips():
    grid, readings = calibration_curve(4)
    assert invert_reading(-0.05, grid, readings) == 0.0     # below f(0): clip to 0
    assert invert_reading(2.5, grid, readings) == 1.0       # above f(1) = 2 bits: clip to 1


def test_the_trap_is_real():
    # Same coupling, structurally different environments: the true readings
    # differ by a large factor. Raw bits are environment-relative by definition.
    lean = _true_reading(0.6, **LEAN)
    rich = _true_reading(0.6, **RICH)
    assert rich / lean > 3.5


def test_end_to_end_recovered_coupling_agrees():
    # The demo's headline, as a test: estimate in both environments at the same
    # kappa, invert each environment's own curve, and the recovered couplings
    # agree with the truth and with each other — while the raw readings differ
    # by more than half a bit.
    rng = np.random.default_rng(3)
    kappa, n, repeats = 0.6, 10_000, 50
    khat = {}
    raw = {}
    for label, env in (("lean", LEAN), ("rich", RICH)):
        grid, readings = calibration_curve(env["n_actions"])
        kappa_s = np.full(len(env["p_s"]), kappa)
        estimates = np.empty(repeats)
        for r in range(repeats):
            s, a1, a2 = sample(kappa_s, env["p_s"], env["n_actions"], n, rng)
            estimates[r] = conditional_mi(
                s, a1, a2, env["n_actions"], len(env["p_s"]), corrected=True
            )
        raw[label] = estimates.mean()
        khat[label] = float(np.mean(invert_reading(estimates, grid, readings)))

    assert abs(raw["rich"] - raw["lean"]) > 0.5          # the trap: bits disagree
    assert abs(khat["lean"] - kappa) < 0.03              # the fix: kappa-hat is true...
    assert abs(khat["rich"] - kappa) < 0.03
    assert abs(khat["lean"] - khat["rich"]) < 0.03       # ...and agrees across environments


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} known-answer tests passed.")


if __name__ == "__main__":
    _run_all()
