"""Mutual-information estimators for the Phase 0 calibration.

Plug-in MI from empirical counts is bias-dominated at small sample sizes -- it
*overestimates* dependence because finite samples look more structured than the
distribution they came from. The Miller-Madow correction subtracts the leading
1/N entropy bias term and is the named, bias-corrected estimator the program
uses for discrete information quantities.

For the conditional metric I(a1; a2 | s) the estimator stratifies by context and
averages over the empirical P(s). Both a plug-in and a Miller-Madow conditional
estimator are provided so the calibration can show the correction earning its
place against the known ground truth.
"""

from __future__ import annotations

import numpy as np


def plugin_mi(a1, a2, n_actions: int) -> float:
    """Plug-in I(a1; a2) in bits from the empirical joint of two action arrays."""
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    n = a1.size
    if n == 0:
        return 0.0

    counts = np.bincount(a1 * n_actions + a2, minlength=n_actions * n_actions)
    joint = counts.reshape(n_actions, n_actions).astype(float) / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    independent = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask])))


def miller_madow_mi(a1, a2, n_actions: int) -> float:
    """Miller-Madow bias-corrected I(a1; a2) in bits.

    MI = H(a1) + H(a2) - H(a1, a2); each plug-in entropy carries a +(m - 1)/(2N)
    bias term in nats, where m is the number of *observed* categories. The net
    correction to MI is (m1 + m2 - m12 - 1) / (2N), converted to bits.
    """
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    n = a1.size
    if n == 0:
        return 0.0

    m1 = np.unique(a1).size
    m2 = np.unique(a2).size
    m12 = np.unique(a1 * n_actions + a2).size
    correction = (m1 + m2 - m12 - 1) / (2.0 * n * np.log(2.0))
    return plugin_mi(a1, a2, n_actions) + correction


def conditional_mi(s, a1, a2, n_actions: int, n_contexts: int, *, corrected: bool) -> float:
    """I(a1; a2 | s) in bits: per-context MI weighted by the empirical P(s).

    ``corrected`` selects the Miller-Madow estimator (True) or plug-in (False).
    """
    s = np.asarray(s, dtype=int)
    n = s.size
    if n == 0:
        return 0.0

    estimator = miller_madow_mi if corrected else plugin_mi
    total = 0.0
    for context in range(n_contexts):
        in_context = s == context
        n_s = int(in_context.sum())
        if n_s == 0:
            continue
        total += (n_s / n) * estimator(a1[in_context], a2[in_context], n_actions)
    return total
