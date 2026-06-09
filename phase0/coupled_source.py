"""Analytic phantom for the conditional-MI coordination metric (Phase 0).

A synthetic two-agent source whose conditional mutual information
I(a1; a2 | s) is *known in closed form*. On real LLM-agent rollouts the
estimator has to recover a collective property it cannot check against a true
value; here the true value is constructed, so estimator bias and the sampling
noise floor can be measured directly -- the de-risk gate the rest of the
program runs behind.

Generative model, per context s in {0, ..., S-1}:

    a1 | s       ~ Uniform{0, ..., K-1}
    a2 | a1, s   =  a1            with probability  kappa_s   (coordination)
                    Uniform{...}  with probability  1 - kappa_s

so the conditional joint is

    P(a1 = i, a2 = j | s) = (1 / K) * ( kappa_s * [i == j] + (1 - kappa_s) / K ).

kappa_s in [0, 1] tunes coordination within context s:

    kappa_s = 0  ->  a1, a2 independent uniform  ->  I(a1; a2 | s) = 0
    kappa_s = 1  ->  a2 == a1                     ->  I(a1; a2 | s) = log2(K)

The metric is *conditional* MI, so an estimator must stratify by s and average
over P(s). Conditioning splits the sample across strata, which is exactly where
finite-sample bias bites -- the reason the phantom calibrates the conditional
quantity rather than a single unconditional MI.
"""

from __future__ import annotations

import numpy as np


def conditional_joint(kappa_s, n_actions: int) -> np.ndarray:
    """P(a1, a2 | s) for every context, shape ``(S, K, K)``.

    ``kappa_s`` is the per-context coupling, shape ``(S,)``, each entry in [0, 1].
    """
    kappa_s = np.asarray(kappa_s, dtype=float).reshape(-1)
    eye = np.eye(n_actions)
    # P(i, j | s) = (1/K) * (kappa_s * [i == j] + (1 - kappa_s) / K)
    diagonal = kappa_s[:, None, None] * eye[None, :, :]
    floor = (1.0 - kappa_s)[:, None, None] / n_actions
    return (diagonal + floor) / n_actions


def mi_of_joint(joint) -> float:
    """Exact mutual information in bits of a 2-D joint distribution.

    This is the ground-truth computation: a finite sum over a known joint, with
    the convention 0 * log(0) = 0.
    """
    joint = np.asarray(joint, dtype=float)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    independent = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask])))


def true_conditional_mi(kappa_s, p_s, n_actions: int) -> float:
    """Closed-form I(a1; a2 | s) in bits = sum_s P(s) * I(a1; a2 | s)."""
    p_s = np.asarray(p_s, dtype=float)
    joints = conditional_joint(kappa_s, n_actions)
    per_context = np.array([mi_of_joint(joints[s]) for s in range(joints.shape[0])])
    return float(np.sum(p_s * per_context))


def sample(kappa_s, p_s, n_actions: int, n: int, rng: np.random.Generator):
    """Draw ``n`` independent triples (s, a1, a2) from the phantom.

    Returns three integer arrays of length ``n``.
    """
    kappa_s = np.asarray(kappa_s, dtype=float)
    p_s = np.asarray(p_s, dtype=float)
    n_contexts = kappa_s.size

    s = rng.choice(n_contexts, size=n, p=p_s)
    a1 = rng.integers(0, n_actions, size=n)
    coupled = rng.random(n) < kappa_s[s]
    a2_independent = rng.integers(0, n_actions, size=n)
    a2 = np.where(coupled, a1, a2_independent)
    return s, a1, a2
