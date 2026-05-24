"""Observables for the IPD evaluation-invariance demo.

Four quantities computed from a rollout of a two-agent collective. Two are
*behavioral* (functions of the action stream) and are expected to be invariant
under reward rescaling; two are *reward-based* and are not.

- ``pairwise_mutual_information(a1, a2)`` -- I(a1; a2) in bits. Behavioral.
  Measures coordination strength. Expected invariant under reward rescaling.
- ``action_autocorrelation(actions, lag)`` -- lag-`lag` autocorrelation of one
  agent's action sequence. A critical-slowing-down indicator borrowed from the
  tipping-point literature (ecology, finance). Behavioral. Expected invariant.
- ``joint_return(rewards)`` -- mean per-step joint reward. Reward-based.
  Scales *linearly* with the reward-rescaling factor (the obvious
  evaluation-setup dependence).
- ``per_step_reward_variance(rewards)`` -- variance of per-step joint reward.
  The *negative control*: scales *quadratically* with the rescaling factor,
  so it must fail the invariance test for the discrimination to be non-vacuous.

The behavioral observables depend only on the learned policy. Rescaling all
rewards by a positive constant c leaves the Bellman fixed point at c times its
original value, so the greedy policy is unchanged and epsilon-greedy
exploration is reward-scale-free. Hence the behavioral observables are invariant
in expectation while the reward-based ones carry the c (and c-squared) factors.
"""

from __future__ import annotations

import numpy as np


def pairwise_mutual_information(a1, a2, n_actions: int = 2) -> float:
    """Mutual information I(a1; a2) in bits from the empirical joint
    distribution of two equal-length discrete action sequences.

    Uses the convention 0 * log(0) = 0. Returns 0.0 for empty input.
    """
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    if a1.size == 0 or a1.shape != a2.shape:
        return 0.0

    idx = a1 * n_actions + a2
    counts = np.bincount(idx, minlength=n_actions * n_actions).astype(float)
    joint = counts.reshape(n_actions, n_actions)
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)  # (n, 1)
    py = joint.sum(axis=0, keepdims=True)  # (1, n)
    independent = px @ py  # outer product p(x)p(y)

    mask = joint > 0  # terms with joint == 0 contribute 0 by convention
    return float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask])))


def action_autocorrelation(actions, lag: int = 1) -> float:
    """Lag-`lag` Pearson autocorrelation of a single agent's action sequence.

    A constant sequence has undefined Pearson correlation (zero variance); we
    define it as 1.0, treating a never-changing policy as maximally persistent.
    Returns NaN if the sequence is too short for the requested lag.
    """
    x = np.asarray(actions, dtype=float)
    if x.size <= lag:
        return float("nan")

    head = x[:-lag]
    tail = x[lag:]
    if head.std() == 0.0 or tail.std() == 0.0:
        return 1.0
    return float(np.corrcoef(head, tail)[0, 1])


def joint_return(per_step_joint_reward) -> float:
    """Mean per-step joint reward over the rollout (normalized return).

    `per_step_joint_reward` is the per-timestep sum of both agents' rewards,
    reported at the *native* training reward scale.
    """
    r = np.asarray(per_step_joint_reward, dtype=float)
    return float(r.mean()) if r.size else 0.0


def per_step_reward_variance(per_step_joint_reward) -> float:
    """Variance of per-step joint reward (the negative-control observable)."""
    r = np.asarray(per_step_joint_reward, dtype=float)
    return float(r.var()) if r.size else 0.0
