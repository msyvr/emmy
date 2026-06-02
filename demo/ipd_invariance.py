"""Evaluation-invariance of behavioral observables under reward rescaling.

A smoke-test on the simplest, *provable* corner of the program's invariance
question: under reward rescaling, action-derived quantities are invariant (the
greedy policy is unchanged, so the behavior is too) while reward-derived quantities
scale. This exercises the measurement pipeline on a case with a known answer; it does
not, on its own, evidence the harder bet -- invariance under algorithm or environment
changes, where the answer is genuinely unknown.

Setup: two tabular Q-learning agents play the iterated prisoner's dilemma. We
run a controlled experiment -- per seed, the *only* thing varied is the reward
scale c in {1, 2}; the same random stream drives both runs. Because reward
rescaling is a strategic relabeling (it multiplies the Bellman fixed point by c
and leaves the greedy policy unchanged), the learned behavior is invariant, so:

    - pairwise mutual information I(a1; a2)  -> invariant   (behavioral)
    - lag-1 action autocorrelation           -> invariant   (behavioral, a CSD
                                                              early-warning indicator)
    - joint return (mean per-step reward)    -> scales x c   (reward-based)
    - per-step reward variance               -> scales x c^2 (negative control)

Run:  uv run python demo/ipd_invariance.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from observables import (
    action_autocorrelation,
    joint_return,
    pairwise_mutual_information,
    per_step_reward_variance,
)

N_ACTIONS = 2  # 0 = cooperate, 1 = defect
HISTORY = 2  # state = last 2 joint actions
N_STATES = N_ACTIONS ** (2 * HISTORY)  # 16

# PAYOFF[my_action, opponent_action] for the row agent (T=5 > R=3 > P=1 > S=0).
PAYOFF = np.array(
    [
        [3.0, 0.0],  # I cooperate: opp C -> 3, opp D -> 0
        [5.0, 1.0],  # I defect:    opp C -> 5, opp D -> 1
    ]
)


def step_state(state: int, a1: int, a2: int) -> int:
    """Shift the newest joint action into the last-2-joint-actions state."""
    joint = a1 * N_ACTIONS + a2  # 0..3
    return (state * N_ACTIONS**2 + joint) % N_STATES


class QLearner:
    def __init__(self, rng: np.random.Generator, alpha: float = 0.1, gamma: float = 0.95,
                 q_init: float = 0.0):
        self.q = np.full((N_STATES, N_ACTIONS), q_init)
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng

    def act(self, state: int, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        row = self.q[state]
        # random tie-break so an all-zero table doesn't bias toward action 0
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    def update(self, s: int, a: int, r: float, s_next: int) -> None:
        target = r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])


def train(seed: int, reward_scale: float, n_steps: int = 50_000,
          eps_start: float = 0.5, eps_end: float = 0.05,
          q_init: float = 0.0) -> tuple[QLearner, QLearner]:
    # q_init is in *unscaled* payoff units; multiply by reward_scale so that the
    # controlled-experiment coupling Q2 == 2*Q1 holds from initialization.
    a1 = QLearner(np.random.default_rng(seed * 2 + 1), q_init=q_init * reward_scale)
    a2 = QLearner(np.random.default_rng(seed * 2 + 2), q_init=q_init * reward_scale)
    state = 0
    for t in range(n_steps):
        eps = eps_start + (eps_end - eps_start) * (t / n_steps)
        x = a1.act(state, eps)
        y = a2.act(state, eps)
        r1 = PAYOFF[x, y] * reward_scale
        r2 = PAYOFF[y, x] * reward_scale
        nxt = step_state(state, x, y)
        a1.update(state, x, r1, nxt)
        a2.update(state, y, r2, nxt)
        state = nxt
    return a1, a2


def rollout(a1: QLearner, a2: QLearner, reward_scale: float, seed: int,
            n_steps: int = 2000, eps: float = 0.05):
    """Evaluate the learned (near-greedy) policy. Actions are reward-scale-free;
    per-step joint reward is recorded at the native training scale c."""
    a1.rng = np.random.default_rng(seed + 100)
    a2.rng = np.random.default_rng(seed + 200)
    state = 0
    acts1, acts2, joint_r = [], [], []
    for _ in range(n_steps):
        x = a1.act(state, eps)
        y = a2.act(state, eps)
        acts1.append(x)
        acts2.append(y)
        joint_r.append((PAYOFF[x, y] + PAYOFF[y, x]) * reward_scale)
        state = step_state(state, x, y)
    return np.array(acts1), np.array(acts2), np.array(joint_r)


def run(scales=(1.0, 2.0), seeds=range(5)) -> list[dict]:
    records: list[dict] = []
    for c in scales:
        for s in seeds:
            a1, a2 = train(s, c)
            x, y, jr = rollout(a1, a2, c, seed=s)
            records.append(
                {
                    "scale": c,
                    "seed": s,
                    "mutual_information": pairwise_mutual_information(x, y),
                    "action_autocorrelation": 0.5
                    * (action_autocorrelation(x) + action_autocorrelation(y)),
                    "joint_return": joint_return(jr),
                    "reward_variance": per_step_reward_variance(jr),
                    "defect_rate": 0.5 * (x.mean() + y.mean()),
                }
            )
    return records


# (key, label, expected x2/x1 ratio, kind)
OBSERVABLES = [
    ("defect_rate", "defection rate", "invariant", "behavioral"),
    ("mutual_information", "I(a1;a2)  [bits]", "invariant", "behavioral"),
    ("action_autocorrelation", "action autocorr (lag 1)", "invariant", "behavioral"),
    ("joint_return", "joint return (per step)", "x c", "reward"),
    ("reward_variance", "per-step reward variance", "x c^2 (neg. control)", "reward"),
]


def summarize(records: list[dict]) -> None:
    header = f"{'observable':<28}{'x1 mean':>12}{'x2 mean':>12}{'x2 / x1':>10}   expected"
    print("\n" + header)
    print("-" * len(header))
    for key, label, expected, kind in OBSERVABLES:
        m1 = np.mean([r[key] for r in records if r["scale"] == 1.0])
        m2 = np.mean([r[key] for r in records if r["scale"] == 2.0])
        ratio = m2 / m1 if abs(m1) > 1e-9 else float("nan")
        print(f"{label:<28}{m1:>12.4f}{m2:>12.4f}{ratio:>10.3f}   {expected}")
    print(
        "\nBehavioral observables (functions of the action stream) are invariant;\n"
        "reward-based observables carry the rescaling factor. Independent Q-learners\n"
        "converge to mutual defection in the IPD, so coordination (MI) is genuinely\n"
        "low here -- and that low value is itself reported invariantly across scale."
    )


def save_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(OBSERVABLES), figsize=(4 + 2.6 * len(OBSERVABLES), 4.4))
    colors = {1.0: "#1f77b4", 2.0: "#ff7f0e"}
    tints = {"behavioral": "#eef7ee", "reward": "#fdeeee"}
    jitter = np.random.default_rng(0)
    for ax, (key, label, expected, kind) in zip(axes, OBSERVABLES):
        ax.set_facecolor(tints[kind])
        for c in (1.0, 2.0):
            vals = [r[key] for r in records if r["scale"] == c]
            xs = np.full(len(vals), c) + jitter.normal(0, 0.03, len(vals))
            ax.scatter(xs, vals, color=colors[c], s=55, alpha=0.85, label=f"reward x{c:.0f}")
            ax.scatter([c], [np.mean(vals)], color=colors[c], marker="_", s=900, linewidths=2.5)
        flag = "INVARIANT" if kind == "behavioral" else f"scales {expected}"
        ax.set_title(f"{label}\n[{flag}]", fontsize=10)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(["x1", "x2"])
        ax.set_xlim(0.6, 2.4)
        ax.margins(y=0.18)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Evaluation-invariance under reward rescaling  (IPD, tabular Q-learning, 5 seeds)\n"
        "green = behavioral observables (overlap across reward scale);  "
        "red = reward-based observables (shift with it)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=130)
    print(f"\nFigure -> {path}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    records = run()
    summarize(records)
    save_csv(records, results_dir / "results.csv")
    plot(records, results_dir / "invariance.png")


if __name__ == "__main__":
    main()
