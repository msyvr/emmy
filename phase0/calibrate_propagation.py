"""Phase 0 calibration: recover a known misalignment-propagation coefficient.

Runs the propagation estimator on the contagion phantom (`propagation_source`)
across a grid of true coefficients and dose-sweep budgets, and reports:

  - trueness    -- does the estimate track the known propagation coefficient beta;
  - bias        -- the finite-sample bias of the OLS-slope estimator;
  - noise floor -- the 95% interval half-width over independent re-samples, which
                   sets the smallest beta resolvable from zero -- i.e. the budget
                   at which "did the planted misalignment spread?" stops being a guess.

The deliverable is that detection threshold, before any LLM-agent run with a
planted misaligned agent.

Run:  uv run python phase0/calibrate_propagation.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from propagation_estimators import estimate_propagation
from propagation_source import sample, true_propagation

DOSE_GRID = tuple(np.linspace(0.0, 1.0, 5))
N_TARGETS = 4
BETAS = (0.0, 0.2, 0.4, 0.6, 0.8)
REPLICATES = (5, 15, 50, 150, 500)   # per (dose, target); budget = 5 doses * 4 targets * r
REPEATS = 300
SEED = 0
REPORT_BETA = 0.2                    # faint-contagion case for floor-vs-budget


def run_sweep(betas=BETAS, replicates=REPLICATES, *, dose_grid=DOSE_GRID,
              n_targets=N_TARGETS, repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    budget_unit = len(dose_grid) * n_targets
    records: list[dict] = []
    for beta in betas:
        true = true_propagation(beta)
        for r in replicates:
            est = np.empty(repeats)
            for i in range(repeats):
                doses, rates = sample(beta, dose_grid, n_targets, r, rng)
                est[i] = estimate_propagation(doses, rates)
            lo, hi = np.percentile(est, [2.5, 97.5])
            floor = float((hi - lo) / 2.0)
            records.append(
                {
                    "beta": beta,
                    "budget": budget_unit * r,
                    "replicates": r,
                    "true_beta": true,
                    "est_mean": float(est.mean()),
                    "bias": float(est.mean() - true),
                    "floor": floor,
                    "resolved": bool(est.mean() > floor and true > 0.0),
                }
            )
    return records


def summarize(records: list[dict]) -> None:
    max_budget = max(r["budget"] for r in records)
    print(f"\nTRUENESS across the propagation range  (budget = {max_budget:,} actions)")
    header = f"{'true beta':>10}{'estimate':>11}{'bias':>9}{'floor +/-':>11}{'contagion resolved':>20}"
    print(header)
    print("-" * len(header))
    for r in [r for r in records if r["budget"] == max_budget]:
        label = "none (beta=0)" if r["true_beta"] == 0 else ("yes" if r["resolved"] else "NO")
        print(f"{r['beta']:>10.2f}{r['est_mean']:>11.3f}{r['bias']:>9.3f}{r['floor']:>11.3f}{label:>20}")

    print(f"\nNOISE FLOOR vs dose-sweep budget  (beta = {REPORT_BETA}, faint contagion)")
    header = f"{'budget':>9}{'replicates':>12}{'estimate':>11}{'bias':>9}{'floor +/-':>11}"
    print(header)
    print("-" * len(header))
    for r in [r for r in records if r["beta"] == REPORT_BETA]:
        print(f"{r['budget']:>9,}{r['replicates']:>12}{r['est_mean']:>11.3f}{r['bias']:>9.3f}{r['floor']:>11.3f}")

    print(
        "\nThe estimator is near-unbiased; the floor sets the detection threshold.\n"
        "Faint contagion (small beta) is resolvable from zero only once the floor drops\n"
        "below beta -- the dose-sweep budget needed to claim the misalignment spread."
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
    at_beta = [r for r in records if r["beta"] == REPORT_BETA]

    fig, (ax_true, ax_floor) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    true_vals = [r["true_beta"] for r in at_max]
    est_vals = [r["est_mean"] for r in at_max]
    floors = [r["floor"] for r in at_max]
    lim = max(true_vals) * 1.15
    ax_true.plot([0, lim], [0, lim], color="#999999", lw=1, ls="--", label="exact recovery")
    ax_true.errorbar(true_vals, est_vals, yerr=floors, fmt="o", color="#1f77b4",
                     capsize=4, ms=7, label="propagation estimate")
    ax_true.set_xlabel("true propagation coefficient  beta  (0 = contained, 1 = full)")
    ax_true.set_ylabel("estimated  beta")
    ax_true.set_title(f"Trueness across propagation strength\n(budget = {max_budget:,}, error bars = 95% floor)")
    ax_true.set_xlim(-0.05, lim)
    ax_true.set_ylim(-0.05, lim)
    ax_true.legend(loc="best", fontsize=9)

    # Floor vs budget against the ideal 1/sqrt rate; the signal line shows where a
    # contagion of this size sits, so it is detectable where the floor dips below it.
    # (Bias is near zero -- shown in the trueness panel and the CSV -- so it is not
    # plotted here, where on a log axis it would only trace Monte-Carlo scatter.)
    budgets = np.array([r["budget"] for r in at_beta])
    floors = np.array([r["floor"] for r in at_beta])
    ax_floor.plot(budgets, floors, "^--", color="#777777", label="noise floor (95% half-width)")
    ax_floor.plot(budgets, floors[0] * np.sqrt(budgets[0] / budgets), ":", color="#bbbbbb",
                  lw=1.5, label="ideal  1 / sqrt(budget)")
    ax_floor.axhline(REPORT_BETA, color="#d62728", lw=1.2, ls=":",
                     label=f"beta = {REPORT_BETA} (signal)")
    ax_floor.set_xscale("log")
    ax_floor.set_yscale("log")
    ax_floor.set_xlabel("dose-sweep budget  (doses x targets x replicates)")
    ax_floor.set_ylabel("propagation units")
    ax_floor.set_title(f"Noise floor vs dose-sweep budget  (beta = {REPORT_BETA})"
                       "\nfloor tracks 1/sqrt(budget); detectable where it dips below beta")
    ax_floor.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 calibration -- misalignment-propagation (contagion) metric on an analytic phantom\n"
        "estimator recovers a known propagation coefficient; the floor sets the budget at which faint contagion is detectable",
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
    save_csv(records, results_dir / "propagation_calibration.csv")
    plot(records, results_dir / "propagation_calibration.png")


if __name__ == "__main__":
    main()
