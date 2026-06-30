"""Estimator for the misalignment-propagation coefficient.

The propagation coefficient is the slope of target misalignment vs the seed's
planted-misalignment dose. A least-squares line through the observed
(dose, misalignment-rate) points recovers it. The estimator is well-behaved
(an OLS slope on binomial rates); the calibration's job is to set the dose-sweep
budget at which a faint contagion is resolvable from zero.
"""

from __future__ import annotations

import numpy as np


def estimate_propagation(doses, rates) -> float:
    """Propagation coefficient estimate: the OLS slope of rates on doses.

    Returns NaN with fewer than two distinct dose levels.
    """
    doses = np.asarray(doses, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if np.unique(doses).size < 2:
        return float("nan")
    return float(np.polyfit(doses, rates, 1)[0])  # slope of the line


def naive_contagion_slope(z_seed, z_target) -> float:
    """Observational contagion: E[target | seed misaligned] - E[target | seed aligned].

    The OLS slope of target on the binary seed indicator. Confounded if a shared
    cause drives both; NaN if either seed group is empty.
    """
    z_seed = np.asarray(z_seed, dtype=int)
    z_target = np.asarray(z_target, dtype=float)
    misaligned = z_seed == 1
    aligned = z_seed == 0
    if not misaligned.any() or not aligned.any():
        return float("nan")
    return float(z_target[misaligned].mean() - z_target[aligned].mean())


def conditioned_contagion_slope(s, z_seed, z_target, n_strata: int) -> float:
    """Contagion stratified by a shared cause ``s`` and averaged over P(s).

    Within each value of s the seed->target difference is the genuine contagion;
    the empirical-P(s) average removes a common-cause confound the naive slope is
    fooled by. Strata that cannot form the difference (a seed group is empty) are
    dropped; NaN if none contribute.
    """
    s = np.asarray(s, dtype=int)
    z_seed = np.asarray(z_seed, dtype=int)
    z_target = np.asarray(z_target, dtype=float)
    n = s.size
    total = 0.0
    contributed = False
    for stratum in range(n_strata):
        in_s = s == stratum
        n_s = int(in_s.sum())
        if n_s == 0:
            continue
        zs = z_seed[in_s]
        zt = z_target[in_s]
        if not (zs == 1).any() or not (zs == 0).any():
            continue
        slope_s = zt[zs == 1].mean() - zt[zs == 0].mean()
        total += (n_s / n) * slope_s
        contributed = True
    return float(total) if contributed else float("nan")
