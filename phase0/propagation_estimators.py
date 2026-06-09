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
