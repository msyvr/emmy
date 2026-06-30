"""Discriminant-validity phantom for the misalignment-propagation metric (Phase 0).

`propagation_source` calibrates the *interventional* contagion coefficient: you
plant a dose in the seed and measure how the target responds. That controlled
dose is confound-free by construction. But an outside observer often cannot
intervene -- they see one agent misaligned and others misaligned, and must ask
whether the misalignment *spread* (seed -> target) or whether a *shared trigger*
tripped both independently. This phantom calibrates that distinction, with two
knobs:

    kappa_cc   -- a shared trigger s that raises BOTH the seed's and the targets'
                  misalignment (a common cause, nothing spreading),
    kappa_link -- genuine contagion: the seed's misalignment raises the target's.

Generative model (binary misalignment, binary trigger s):

    s             ~ Bernoulli(ps)
    z_seed   | s  ~ Bernoulli(p0 + kappa_cc * s)
    z_target | z_seed, s ~ Bernoulli(q0 + kappa_link * z_seed + kappa_cc * s)

Two ways to read the contagion:

    naive slope        = E[z_target | z_seed=1] - E[z_target | z_seed=0]
                         -- fooled: the shared trigger correlates seed and target
                            even when nothing spreads (kappa_link = 0);
    conditioned slope  = the same difference computed within each trigger value s
                         and averaged over P(s) -- recovers kappa_link, the genuine
                         contagion, and reads zero when kappa_link = 0 for any kappa_cc.

The discriminant case is kappa_cc > 0, kappa_link = 0: the naive slope is positive
(a shared trigger looks like spreading) while the conditioned slope is zero. In the
LLM-agent regime the trigger is concrete -- a shared prompt, a common base model, a
shared environmental event -- and it is observable, hence conditionable.
"""

from __future__ import annotations

import numpy as np

PS = 0.5     # P(shared trigger active)
P0 = 0.2     # seed baseline misalignment (un-triggered)
Q0 = 0.05    # target baseline misalignment (un-triggered, aligned seed)


def _joint(kappa_cc: float, kappa_link: float, ps: float = PS, p0: float = P0,
           q0: float = Q0) -> np.ndarray:
    """P(s, z_seed, z_target) for the 2x2x2 binary system, shape ``(2, 2, 2)``."""
    p_s = np.array([1.0 - ps, ps])                      # [s]
    p_seed1 = np.array([p0, p0 + kappa_cc])             # P(z_seed = 1 | s)
    j = np.zeros((2, 2, 2))
    for s in (0, 1):
        for zs in (0, 1):
            p_zs = p_seed1[s] if zs == 1 else 1.0 - p_seed1[s]
            p_t1 = q0 + kappa_link * zs + kappa_cc * s
            for zt in (0, 1):
                p_zt = p_t1 if zt == 1 else 1.0 - p_t1
                j[s, zs, zt] = p_s[s] * p_zs * p_zt
    return j


def true_conditioned_slope(kappa_link: float) -> float:
    """Genuine contagion: the seed->target difference within a fixed trigger = kappa_link."""
    return float(kappa_link)


def true_naive_slope(kappa_cc: float, kappa_link: float, ps: float = PS, p0: float = P0,
                     q0: float = Q0) -> float:
    """E[target | seed=1] - E[target | seed=0], marginal over the trigger.

    Equals kappa_link only when kappa_cc = 0; otherwise inflated by the shared
    trigger -- the confound the conditioned slope removes.
    """
    j = _joint(kappa_cc, kappa_link, ps, p0, q0)
    p_zs_zt = j.sum(axis=0)                  # (z_seed, z_target)
    p_zs = p_zs_zt.sum(axis=1)               # (z_seed,)
    e_t_given_zs1 = p_zs_zt[1, 1] / p_zs[1]
    e_t_given_zs0 = p_zs_zt[0, 1] / p_zs[0]
    return float(e_t_given_zs1 - e_t_given_zs0)


def sample(kappa_cc: float, kappa_link: float, n: int, rng: np.random.Generator, *,
           ps: float = PS, p0: float = P0, q0: float = Q0):
    """Draw ``n`` trials (s, z_seed, z_target), each a binary integer array."""
    s = (rng.random(n) < ps).astype(int)
    z_seed = (rng.random(n) < (p0 + kappa_cc * s)).astype(int)
    p_t = q0 + kappa_link * z_seed + kappa_cc * s
    z_target = (rng.random(n) < p_t).astype(int)
    return s, z_seed, z_target
