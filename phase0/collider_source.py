"""Negative-control phantom for the conditioning rule itself (Phase 0).

`confound_source` shows conditioning *rescuing* the coordination metric: the
context s there is a common CAUSE of both agents' actions, and conditioning on
it strips a false positive out. This phantom is the mirror case, and the reason
"condition on everything shared" is the wrong rule: here s is a common EFFECT —
a state the two agents jointly produced — and conditioning on it *manufactures*
dependence between agents that share nothing at all. In causal-graph language s
is a collider; stratifying on a collider opens a path between its causes.

Generative model (all binary, all draws independent):

    m1, m2 ~ Bernoulli(1/2)                      # private, independent memories
    a1 = m1 XOR Bernoulli(eps1)                  # behavior reflects private memory
    a2 = m2 XOR Bernoulli(eps2)                  #   (eps = memory->action noise)
    s  = m1 XOR m2 XOR Bernoulli(delta)          # jointly-produced state, recorded
                                                 #   with fidelity 1 - 2*delta

Parameterization used throughout: per-agent behavioral persistence
rho_i = 1 - 2*eps_i in [0, 1] (rho = 1: action == memory; rho = 0: memoryless)
and state fidelity lam = 1 - 2*delta in [0, 1] (lam = 1: s records the joint
product exactly; lam = 0: s is noise).

The two quantities that matter — both known exactly:

    I(a1; a2)      =  0, identically: (m1, eps1-noise) is independent of
                      (m2, eps2-noise), so the actions are independent. Zero
                      coupling exists. Plain MI gives the CORRECT answer here.
    I(a1; a2 | s)  =  1 - H_b((1 - rho1*rho2*lam) / 2)   bits,

where H_b is the binary entropy. Derivation: conditioned on s, the channel
a1 -> a2 is a binary symmetric channel with flip probability
phi = eps1 * eps2 * delta (* = binary convolution, p*q = p(1-q) + q(1-p)),
and 1 - 2*phi = (1-2*eps1)(1-2*eps2)(1-2*delta) = rho1*rho2*lam.

So the manufactured dependence is governed by one product, rho1*rho2*lam, and
vanishes when ANY factor is zero: memoryless behavior leaks nothing, and a
state that does not actually record the joint behavior induces nothing. At
rho1 = rho2 = lam = 1 conditioning manufactures a full bit between two agents
with no channel between them.

The moral, as a rule for the conditioning set: condition on CAUSES of the
actions being compared (exogenous inputs, task specification, configuration),
never on their EFFECTS. In the LLM-agent regime the effect case is generic,
not exotic: the environment state at time t — a shared scratchpad, a market
price, a task artifact — is a product of both agents' earlier actions, and an
audit that stratifies on it fabricates exactly the coordination it is looking
for. Paired with `confound_source`, this phantom completes the rule: condition
on the shared cause; do not condition on the shared effect.
"""

from __future__ import annotations

import numpy as np

from coupled_source import mi_of_joint


def flip_prob(strength: float) -> float:
    """Map a persistence/fidelity strength in [0, 1] to a flip probability in [0, 1/2]."""
    return (1.0 - strength) / 2.0


def convolve(p: float, q: float) -> float:
    """Binary convolution p * q = p(1-q) + q(1-p): flip probability of chained flips."""
    return p * (1.0 - q) + q * (1.0 - p)


def binary_entropy(p: float) -> float:
    """H_b(p) in bits, with the 0 log 0 = 0 convention."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def channel_flip(rho1: float, rho2: float, lam: float) -> float:
    """Flip probability phi of the a1 -> a2 channel given s: 1 - 2*phi = rho1*rho2*lam."""
    return (1.0 - rho1 * rho2 * lam) / 2.0


def conditional_joint(rho1: float, rho2: float, lam: float) -> np.ndarray:
    """P(a1, a2 | s) for s in {0, 1}, shape ``(2, 2, 2)``.

    Given s = sigma: a1 is uniform and a2 = a1 XOR sigma XOR Bernoulli(phi), so
    P(i, j | sigma) = (1/2) * ((1 - phi) if j == i XOR sigma else phi).
    """
    phi = channel_flip(rho1, rho2, lam)
    joints = np.empty((2, 2, 2))
    for sigma in range(2):
        for i in range(2):
            for j in range(2):
                joints[sigma, i, j] = 0.5 * ((1.0 - phi) if j == (i ^ sigma) else phi)
    return joints


def unconditional_joint(rho1: float, rho2: float, lam: float) -> np.ndarray:
    """P(a1, a2) marginalized over s (P(s) = 1/2 each) — uniform, exactly.

    The per-sigma joints average to 1/4 in every cell: the actions really are
    independent, whatever the knobs. Kept as a function so the tests can assert
    it rather than assume it.
    """
    return conditional_joint(rho1, rho2, lam).mean(axis=0)


def true_conditional_mi(rho1: float, rho2: float, lam: float) -> float:
    """Closed-form I(a1; a2 | s) in bits: 1 - H_b((1 - rho1*rho2*lam) / 2).

    This is the MANUFACTURED dependence — it is nonzero although no coupling
    exists, and grows with behavioral persistence and state fidelity.
    """
    return 1.0 - binary_entropy(channel_flip(rho1, rho2, lam))


def true_unconditional_mi(rho1: float, rho2: float, lam: float) -> float:
    """Closed-form I(a1; a2) in bits: exactly zero — the actions are independent.

    Zero is the CORRECT reading for this phantom; the conditional value above is
    the artifact.
    """
    return 0.0


def sample(rho1: float, rho2: float, lam: float, n: int, rng: np.random.Generator):
    """Draw ``n`` independent triples (s, a1, a2) from the phantom.

    Returns three integer arrays of length ``n``.
    """
    eps1, eps2, delta = flip_prob(rho1), flip_prob(rho2), flip_prob(lam)
    m1 = rng.integers(0, 2, size=n)
    m2 = rng.integers(0, 2, size=n)
    a1 = m1 ^ (rng.random(n) < eps1)
    a2 = m2 ^ (rng.random(n) < eps2)
    s = m1 ^ m2 ^ (rng.random(n) < delta)
    return s.astype(int), a1.astype(int), a2.astype(int)
