"""Phase 0 calibration: recover a known fragility index from noisy stress sweeps.

Runs the curvature estimator on the fragility phantom (`fragility_source`) across
a grid of true curvatures and rollout budgets, and reports:

  - trueness   -- does the estimate track the *known* curvature kappa, including
                  its sign (fragile vs antifragile), within a stated bound;
  - bias       -- the (small) finite-sample bias of the quadratic-fit estimator;
  - noise floor -- the 95% interval half-width of the estimate over independent
                  re-samples. Here it is the load-bearing number: it sets the
                  smallest |kappa| whose *sign* can be called above the noise, i.e.
                  the budget at which "fragile or antifragile?" stops being a guess.

The deliverable is the resolution limit: fragility is a second-order feature, so a
stress sweep that is too cheap cannot distinguish fragile from antifragile -- and
this says, per budget, where that line is, before any LLM-agent stress run.

Run:  uv run python phase0/calibrate_fragility.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from curvature_estimators import estimate_curvature
from fragility_source import sample, true_curvature

SIGMA_GRID = tuple(np.linspace(0.0, 1.0, 9))
NOISE = 0.05                       # observation noise on the performance score
KAPPAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
REPLICATES = (1, 3, 10, 30, 100)   # rollouts per stress level -> budget = 9 * r
REPEATS = 300
SEED = 0
REPORT_KAPPA = -1.0                # representative fragile case for bias-vs-budget


def run_sweep(kappas=KAPPAS, replicates=REPLICATES, *, sigma_grid=SIGMA_GRID,
              noise=NOISE, repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_levels = len(sigma_grid)
    records: list[dict] = []
    for kappa in kappas:
        true = true_curvature(kappa)
        for r in replicates:
            est = np.empty(repeats)
            for i in range(repeats):
                sigma, y = sample(kappa, sigma_grid, r, noise, rng)
                est[i] = estimate_curvature(sigma, y)
            lo, hi = np.percentile(est, [2.5, 97.5])
            floor = float((hi - lo) / 2.0)
            records.append(
                {
                    "kappa": kappa,
                    "budget": n_levels * r,
                    "replicates": r,
                    "true_curvature": true,
                    "est_mean": float(est.mean()),
                    "bias": float(est.mean() - true),
                    "floor": floor,
                    "sign_resolved": bool(abs(est.mean()) > floor and true != 0.0),
                }
            )
    return records


def summarize(records: list[dict]) -> None:
    max_budget = max(r["budget"] for r in records)
    print(f"\nTRUENESS across the fragility range  (budget = {max_budget} rollouts)")
    header = f"{'true kappa':>11}{'estimate':>11}{'bias':>9}{'floor +/-':>11}{'sign resolved':>15}"
    print(header)
    print("-" * len(header))
    for r in [r for r in records if r["budget"] == max_budget]:
        label = "robust (kappa=0)" if r["true_curvature"] == 0 else ("yes" if r["sign_resolved"] else "NO")
        print(f"{r['kappa']:>11.2f}{r['est_mean']:>11.3f}{r['bias']:>9.3f}{r['floor']:>11.3f}{label:>15}")

    print(f"\nNOISE FLOOR vs rollout budget  (kappa = {REPORT_KAPPA}, fragile)")
    header = f"{'budget':>8}{'replicates/level':>18}{'estimate':>11}{'bias':>9}{'floor +/-':>11}"
    print(header)
    print("-" * len(header))
    for r in [r for r in records if r["kappa"] == REPORT_KAPPA]:
        print(f"{r['budget']:>8}{r['replicates']:>18}{r['est_mean']:>11.3f}{r['bias']:>9.3f}{r['floor']:>11.3f}")

    print(
        "\nThe estimator is near-unbiased; the floor is what binds. Fragility is a\n"
        "second-order feature, so the sign is callable only once the floor drops below\n"
        "|kappa| -- small-magnitude fragility needs a larger stress-sweep budget to resolve."
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

    max_budget = max(r["budget"] for r in records)
    at_max = [r for r in records if r["budget"] == max_budget]
    at_kappa = [r for r in records if r["kappa"] == REPORT_KAPPA]

    fig, (ax_true, ax_floor) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    true_vals = [r["true_curvature"] for r in at_max]
    est_vals = [r["est_mean"] for r in at_max]
    floors = [r["floor"] for r in at_max]
    lim = max(abs(min(true_vals)), abs(max(true_vals))) * 1.15
    ax_true.axhline(0, color="#cccccc", lw=1)
    ax_true.axvline(0, color="#cccccc", lw=1)
    ax_true.plot([-lim, lim], [-lim, lim], color="#999999", lw=1, ls="--", label="exact recovery")
    ax_true.errorbar(true_vals, est_vals, yerr=floors, fmt="o", color="#1f77b4",
                     capsize=4, ms=7, label="curvature estimate")
    ax_true.set_xlabel("true fragility index  kappa  (<0 fragile, >0 antifragile)")
    ax_true.set_ylabel("estimated  kappa")
    ax_true.set_title(f"Trueness + sign resolution\n(budget = {max_budget} rollouts, error bars = 95% floor)")
    ax_true.set_xlim(-lim, lim)
    ax_true.legend(loc="best", fontsize=9)

    # Floor vs budget against the ideal 1/sqrt rate; the signal line shows where a
    # fragility of this size sits, so the sign is callable where the floor dips below it.
    # (Bias is near zero -- shown in the trueness panel and the CSV -- so it is not
    # plotted here, where on a log axis it would only trace Monte-Carlo scatter.)
    budgets = np.array([r["budget"] for r in at_kappa])
    floors = np.array([r["floor"] for r in at_kappa])
    ax_floor.plot(budgets, floors, "^--", color="#777777", label="noise floor (95% half-width)")
    ax_floor.plot(budgets, floors[0] * np.sqrt(budgets[0] / budgets), ":", color="#bbbbbb",
                  lw=1.5, label="ideal  1 / sqrt(budget)")
    ax_floor.axhline(abs(REPORT_KAPPA), color="#d62728", lw=1.2, ls=":",
                     label=f"|kappa| = {abs(REPORT_KAPPA)} (signal)")
    ax_floor.set_xscale("log")
    ax_floor.set_yscale("log")
    ax_floor.set_xlabel("rollout budget  (stress levels x replicates)")
    ax_floor.set_ylabel("curvature units")
    ax_floor.set_title(f"Noise floor vs rollout budget  (kappa = {REPORT_KAPPA})"
                       "\nfloor tracks 1/sqrt(budget); sign callable where it dips below |kappa|")
    ax_floor.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 calibration -- fragility (response-curvature) metric on an analytic phantom\n"
        "estimator recovers a known fragility index; the floor sets the budget at which fragile-vs-antifragile is resolvable",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=130)
    print(f"\nFigure -> {path}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    records = run_sweep()
    summarize(records)
    save_csv(records, results_dir / "fragility_calibration.csv")
    plot(records, results_dir / "fragility_calibration.png")


if __name__ == "__main__":
    main()
