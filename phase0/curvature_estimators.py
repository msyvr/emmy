"""Curvature (fragility-index) estimator for the Phase 0 calibration.

The fragility index is the second derivative of the stress-response at the
operating point. A least-squares quadratic fit to the (sigma, y) rollout data
recovers it: for the fit p0*sigma^2 + p1*sigma + p2, the curvature is
d2/dsigma2 = 2 * p0, independent of where the operating point sits.

Second-derivative estimation is the noise-sensitive part of any fragility
measurement, so this is deliberately the simplest unbiased-in-expectation
estimator; the calibration quantifies how its variance (the noise floor) shrinks
with the rollout budget, and at what budget the sign becomes resolvable.
"""

from __future__ import annotations

import numpy as np


def estimate_curvature(sigma, y) -> float:
    """Fragility index estimate: 2 * (quadratic coefficient) of an OLS fit.

    Returns NaN if there are too few distinct stress levels to fit a quadratic.
    """
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.unique(sigma).size < 3:
        return float("nan")
    # Curvature is shift-invariant; centering keeps the quadratic fit well-conditioned.
    quad_coef = np.polyfit(sigma - sigma.mean(), y, 2)[0]  # highest-degree term first
    return float(2.0 * quad_coef)
