"""Calibration-curve inversion: the cross-environment move (Phase 0).

The invariance target is NOT that raw information readings agree across
environments — they provably do not. `coupled_source`'s copy mechanism with the
same coupling kappa yields a different true I(a1; a2 | s) in every action
alphabet: the reading is alphabet- and structure-relative by definition, before
any estimation error enters. Comparing raw bits across environments compares
the environments.

What CAN travel is the recovered generative parameter. Each environment E has a
calibration curve f_E: kappa -> true reading, known in closed form on a phantom
(measured, on a real system). The move this module implements:

    estimate the reading  ->  invert the environment's own curve  ->  compare
    the recovered couplings kappa-hat, never the raw readings.

Same kappa, different environments, same kappa-hat within the propagated
floors: that is what "the measurement travels" means, made operational. On real
LLM-agent systems the closed-form curve is replaced by a measured dose-response
curve on engineered couplings; the inversion logic is identical.

Closed form for the copy mechanism (uniform marginals, alphabet size K), same
kappa in every context so the context mixture drops out:

    f_K(kappa) = [(1 + (K-1)kappa)/K] * log2(1 + (K-1)kappa)
                 + [(K-1)(1-kappa)/K] * log2(1-kappa)

with f_K(0) = 0 and f_K(1) = log2(K). Strictly increasing on (0, 1], flat at
kappa = 0 (the usual quadratic onset of MI near independence) — so inversion is
ill-conditioned exactly where the curve is flat, and the kappa-hat floor widens
near zero. That is a property of the problem, reported rather than hidden.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Environment(NamedTuple):
    """A structurally distinct setting sharing the copy-mechanism coupling dial."""

    name: str
    n_actions: int
    p_s: tuple[float, ...]

    @property
    def n_contexts(self) -> int:
        return len(self.p_s)


def analytic_curve(kappa, n_actions: int):
    """Closed-form f_K(kappa) in bits; scalar in -> float out, array in -> array out.

    Independent of the mi_of_joint route in `coupled_source` — the tests check
    the two derivations against each other.
    """
    k = np.asarray(kappa, dtype=float)
    K = float(n_actions)
    a_diag = 1.0 + (K - 1.0) * k              # K^2 * P(diagonal cell) = 1 + (K-1)kappa
    b_off = 1.0 - k                           # K^2 * P(off-diagonal cell) = 1 - kappa
    term_diag = (a_diag / K) * np.log2(a_diag)
    with np.errstate(divide="ignore", invalid="ignore"):
        term_off = np.where(
            b_off > 0.0,
            ((K - 1.0) * b_off / K) * np.log2(np.where(b_off > 0.0, b_off, 1.0)),
            0.0,
        )
    out = term_diag + term_off
    return float(out) if out.ndim == 0 else out


def calibration_curve(n_actions: int, resolution: int = 20_001):
    """The environment's calibration curve as a fine grid: (kappa_grid, reading_grid)."""
    kappa_grid = np.linspace(0.0, 1.0, resolution)
    return kappa_grid, analytic_curve(kappa_grid, n_actions)


def invert_reading(reading, kappa_grid: np.ndarray, reading_grid: np.ndarray):
    """Recover kappa-hat from a reading by inverting the calibration curve.

    Monotone interpolation on the fine grid; readings below f(0) = 0 (possible
    for a bias-corrected estimator near independence) clip to kappa-hat = 0,
    readings above f(1) = log2(K) clip to 1. Scalar in -> float out.
    """
    khat = np.interp(reading, reading_grid, kappa_grid)
    return float(khat) if np.ndim(reading) == 0 else khat
