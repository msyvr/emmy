"""Analytic phantom for the fragility / antifragility metric (Phase 0).

The fragility family (CAFE and kin) reduces to one estimation problem: the
*curvature* of a stress-response function R(sigma) -- how a collective's
performance bends as a stress parameter sigma increases. The sign of the
curvature at the operating point is the fragility index:

    d2R/dsigma2 < 0  ->  concave   ->  fragile      (degradation accelerates)
    d2R/dsigma2 = 0  ->  linear    ->  robust
    d2R/dsigma2 > 0  ->  convex    ->  antifragile  (improves under stress)

This phantom has a *known* curvature kappa, observed as noisy replicate rollouts:

    R(sigma) = R0 + slope * (sigma - sigma*) + (kappa / 2) * (sigma - sigma*)^2
    y_i      = R(sigma_i) + noise

so d2R/dsigma2 = kappa exactly, at every point. The estimator's job is to
recover kappa -- a second derivative -- from finitely many noisy points, which
is where the difficulty lives: curvature is a weak, high-order feature, easily
swamped by sampling noise near kappa = 0 (the fragile/antifragile boundary).
The calibration measures how much data it takes to resolve the sign and
magnitude before the metric is trusted on real LLM-agent stress sweeps.
"""

from __future__ import annotations

import numpy as np


def true_curvature(kappa: float) -> float:
    """The fragility index of the phantom: d2R/dsigma2 = kappa (closed form)."""
    return float(kappa)


def response(sigma, kappa: float, *, r0: float = 0.5, slope: float = 0.0,
             sigma_star: float = 0.5) -> np.ndarray:
    """Noise-free response R(sigma) with curvature ``kappa`` about ``sigma_star``."""
    sigma = np.asarray(sigma, dtype=float)
    centered = sigma - sigma_star
    return r0 + slope * centered + 0.5 * kappa * centered**2


def sample(kappa: float, sigma_grid, replicates: int, noise: float,
           rng: np.random.Generator, *, r0: float = 0.5, slope: float = 0.0,
           sigma_star: float = 0.5):
    """Draw noisy replicate rollouts at each stress level.

    Returns flat arrays (sigma, y) of length ``len(sigma_grid) * replicates`` --
    the data an estimator sees: a stress sweep with ``replicates`` runs per level.
    """
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    sigma = np.repeat(sigma_grid, replicates)
    mean = response(sigma, kappa, r0=r0, slope=slope, sigma_star=sigma_star)
    y = mean + rng.normal(0.0, noise, size=sigma.size)
    return sigma, y
