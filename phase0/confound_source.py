"""Discriminant-validity phantom for the conditional-MI coordination metric (Phase 0).

`coupled_source` calibrates *sensitivity*: dial a genuine a1<->a2 coupling and
check the estimator recovers it. It does not test the metric's reason for being.
The coordination metric is *conditional* MI, chosen over plain MI specifically to
strip out a common cause -- two agents that look coupled only because a shared
context drives both. This phantom tests that, with two independent knobs:

    kappa_cc   -- common cause: how strongly the context s biases the *marginals*
                  of BOTH a1 and a2 (each context prefers a different action),
    kappa_link -- a genuine within-context a1<->a2 coupling (the copy mechanism
                  from `coupled_source`),

so the two sources of dependence can be set on or off independently.

Generative model, per context s with a per-context preferred action m_s:

    mu_s(i)      = kappa_cc * [i == m_s] + (1 - kappa_cc) / K        # shared marginal
    a1 | s       ~ mu_s
    a2 | a1, s   = a1        with probability  kappa_link            # genuine link
                   ~ mu_s    with probability  1 - kappa_link        # independent re-draw

Both a1 and a2 carry the same context marginal mu_s, so when contexts prefer
different actions (m_s distinct) the marginals move together with s -- a common
cause. The conditional joint is

    P(a1=i, a2=j | s) = mu_s(i) * ( kappa_link * [i == j] + (1 - kappa_link) * mu_s(j) ).

The two quantities that matter:

    I(a1; a2 | s)   = 0 whenever kappa_link = 0, for ANY kappa_cc   -- the metric
                      correctly reports no coordination when the only dependence
                      is the shared context;
    I(a1; a2)       > 0 as soon as kappa_cc > 0 (distinct m_s), even at
                      kappa_link = 0 -- plain MI is fooled by the common cause.

The discriminant case is kappa_cc > 0, kappa_link = 0: plain MI fires, conditional
MI does not. kappa_cc = 0 recovers `coupled_source` exactly (mu_s uniform), so this
is a strict generalization of the sensitivity phantom.

In the LLM-agent regime this common cause is concrete: two agents on the same base
model with a shared system prompt or shared context behave alike without exchanging
any influence. A coordination measure that cannot tell that apart from genuine
coordination is a false-positive generator; this phantom calibrates that it can.
"""

from __future__ import annotations

import numpy as np

from coupled_source import mi_of_joint


def context_marginal(kappa_cc: float, preferred, n_actions: int) -> np.ndarray:
    """Shared per-context marginal mu_s, shape ``(S, K)``.

    ``preferred`` is m_s, shape ``(S,)``: the action each context biases toward.
    Distinct entries across contexts are what make the context a *common cause*.
    """
    preferred = np.asarray(preferred, dtype=int).reshape(-1)
    n_contexts = preferred.size
    base = (1.0 - kappa_cc) / n_actions
    mu = np.full((n_contexts, n_actions), base)
    mu[np.arange(n_contexts), preferred] += kappa_cc
    return mu


def conditional_joint(kappa_cc: float, kappa_link: float, preferred, n_actions: int) -> np.ndarray:
    """P(a1, a2 | s) for every context, shape ``(S, K, K)``.

    P(i, j | s) = mu_s(i) * ( kappa_link * [i == j] + (1 - kappa_link) * mu_s(j) ).
    Both marginals equal mu_s; dependence within s comes only from kappa_link.
    """
    mu = context_marginal(kappa_cc, preferred, n_actions)             # (S, K)
    eye = np.eye(n_actions)
    inner = kappa_link * eye[None, :, :] + (1.0 - kappa_link) * mu[:, None, :]
    return mu[:, :, None] * inner


def unconditional_joint(kappa_cc, kappa_link, preferred, p_s, n_actions: int) -> np.ndarray:
    """P(a1, a2) marginalised over s, shape ``(K, K)`` -- the common cause is folded in."""
    p_s = np.asarray(p_s, dtype=float)
    joints = conditional_joint(kappa_cc, kappa_link, preferred, n_actions)
    return np.tensordot(p_s, joints, axes=(0, 0))


def true_conditional_mi(kappa_cc, kappa_link, preferred, p_s, n_actions: int) -> float:
    """Closed-form I(a1; a2 | s) in bits = sum_s P(s) * I(a1; a2 | s).

    Exactly zero whenever ``kappa_link == 0``, for any ``kappa_cc``.
    """
    p_s = np.asarray(p_s, dtype=float)
    joints = conditional_joint(kappa_cc, kappa_link, preferred, n_actions)
    per_context = np.array([mi_of_joint(joints[s]) for s in range(joints.shape[0])])
    return float(np.sum(p_s * per_context))


def true_unconditional_mi(kappa_cc, kappa_link, preferred, p_s, n_actions: int) -> float:
    """Closed-form I(a1; a2) in bits, common cause included.

    Positive as soon as ``kappa_cc > 0`` with distinct ``preferred`` actions, even at
    ``kappa_link == 0`` -- this is the confound the conditional metric must reject.
    """
    return mi_of_joint(unconditional_joint(kappa_cc, kappa_link, preferred, p_s, n_actions))


def sample(kappa_cc, kappa_link, preferred, p_s, n_actions: int, n: int,
           rng: np.random.Generator):
    """Draw ``n`` independent triples (s, a1, a2) from the phantom.

    Returns three integer arrays of length ``n``.
    """
    preferred = np.asarray(preferred, dtype=int)
    p_s = np.asarray(p_s, dtype=float)
    n_contexts = preferred.size

    s = rng.choice(n_contexts, size=n, p=p_s)

    # a1 ~ mu_s : prefer m_s with prob kappa_cc, else uniform.
    pref1 = rng.random(n) < kappa_cc
    a1 = np.where(pref1, preferred[s], rng.integers(0, n_actions, size=n))

    # a2 = a1 with prob kappa_link (the genuine link), else an independent draw from mu_s.
    copy = rng.random(n) < kappa_link
    pref2 = rng.random(n) < kappa_cc
    a2_marginal = np.where(pref2, preferred[s], rng.integers(0, n_actions, size=n))
    a2 = np.where(copy, a1, a2_marginal)
    return s, a1, a2
