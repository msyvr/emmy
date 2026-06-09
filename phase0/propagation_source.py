"""Analytic phantom for the misalignment-propagation (contagion) metric (Phase 0).

Motivated by the finding that a multi-agent collective can be *less aligned than
its members*: plant a misalignment in one agent and ask how much it shows up in
the rest. The propagation metric is the coefficient ``beta`` by which a seed
agent's planted-misalignment dose ``m`` raises a target agent's misalignment
probability:

    p_target(m) = b0 + beta * m

    beta = 0  ->  planted misalignment stays contained (no contagion)
    beta = 1  ->  full propagation

so ``beta`` is the known ground truth. The estimator recovers it from binary
misaligned/aligned actions observed across a dose sweep -- a weak slope in
binomial noise, where small ``beta`` (faint contagion) is the hard case and the
question that matters is whether contagion is resolvable from none at all.
"""

from __future__ import annotations

import numpy as np

B0 = 0.05  # baseline (un-seeded) misalignment probability


def true_propagation(beta: float) -> float:
    """The propagation metric: d p_target / d m = beta (closed form)."""
    return float(beta)


def target_misalignment(m, beta: float, b0: float = B0) -> np.ndarray:
    """Target misalignment probability at seed dose ``m``: b0 + beta * m, clipped."""
    return np.clip(b0 + beta * np.asarray(m, dtype=float), 0.0, 1.0)


def sample(beta: float, dose_grid, n_targets: int, replicates: int,
           rng: np.random.Generator, *, b0: float = B0):
    """Observed collective misalignment rate at each seed dose.

    For each dose ``m`` in ``dose_grid``, pools ``n_targets * replicates`` binary
    target actions ~ Bernoulli(b0 + beta*m). Returns (dose_grid, rates).
    """
    dose_grid = np.asarray(dose_grid, dtype=float)
    n_per_dose = n_targets * replicates
    rates = np.empty(dose_grid.size)
    for j, m in enumerate(dose_grid):
        p = float(target_misalignment(m, beta, b0))
        rates[j] = rng.binomial(n_per_dose, p) / n_per_dose
    return dose_grid, rates
