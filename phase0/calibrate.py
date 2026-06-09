"""Phase 0 calibration: recover a known conditional-MI value from finite samples.

Runs the conditional-MI estimator on the analytic phantom (`coupled_source`)
across a grid of coupling strengths and sample sizes, and reports the three
quantities that decide whether the metric is usable downstream:

  - trueness  -- does the estimate track the *known* I(a1; a2 | s) across the
                 coordination range, monotonically and within a stated bound?
  - bias      -- plug-in vs Miller-Madow, as a function of sample size N;
  - noise floor -- the spread of the estimate at a fixed configuration (95%
                 interval half-width over independent re-samples). A cross-setup
                 difference smaller than the floor is not a result.

This is the de-risk gate: with the floor printed under every number, a later
"this metric doesn't travel" cannot be confused with "we estimated it badly".

Run:  uv run python phase0/calibrate.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from coupled_source import sample, true_conditional_mi
from mi_estimators import conditional_mi

N_ACTIONS = 4
N_CONTEXTS = 2
P_S = (0.5, 0.5)
KAPPAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SAMPLE_SIZES = (100, 300, 1000, 3000, 10_000)
REPEATS = 200
SEED = 0
REPORT_KAPPA = 0.6  # representative coupling for the bias-vs-N view


def run_sweep(kappas=KAPPAS, sample_sizes=SAMPLE_SIZES, *, n_actions=N_ACTIONS,
              n_contexts=N_CONTEXTS, p_s=P_S, repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for kappa in kappas:
        kappa_s = np.full(n_contexts, kappa)
        true = true_conditional_mi(kappa_s, p_s, n_actions)
        for n in sample_sizes:
            plugin = np.empty(repeats)
            mm = np.empty(repeats)
            for r in range(repeats):
                s, a1, a2 = sample(kappa_s, p_s, n_actions, n, rng)
                plugin[r] = conditional_mi(s, a1, a2, n_actions, n_contexts, corrected=False)
                mm[r] = conditional_mi(s, a1, a2, n_actions, n_contexts, corrected=True)
            lo, hi = np.percentile(mm, [2.5, 97.5])
            records.append(
                {
                    "kappa": kappa,
                    "n": n,
                    "true_cmi": true,
                    "plugin_mean": float(plugin.mean()),
                    "plugin_bias": float(plugin.mean() - true),
                    "mm_mean": float(mm.mean()),
                    "mm_bias": float(mm.mean() - true),
                    "floor": float((hi - lo) / 2.0),
                }
            )
    return records


def summarize(records: list[dict]) -> None:
    max_n = max(r["n"] for r in records)
    print(f"\nTRUENESS across the coordination range  (N = {max_n:,}, Miller-Madow)")
    header = f"{'kappa':>7}{'true I':>12}{'estimate':>12}{'bias':>10}{'floor +/-':>12}"
    print(header)
    print("-" * len(header))
    for r in [r for r in records if r["n"] == max_n]:
        print(
            f"{r['kappa']:>7.2f}{r['true_cmi']:>12.4f}{r['mm_mean']:>12.4f}"
            f"{r['mm_bias']:>10.4f}{r['floor']:>12.4f}"
        )

    print(f"\nBIAS and NOISE FLOOR vs sample size  (kappa = {REPORT_KAPPA}, true I ="
          f" {next(r['true_cmi'] for r in records if r['kappa'] == REPORT_KAPPA):.4f} bits)")
    header = f"{'N':>8}{'plug-in bias':>15}{'Miller-Madow bias':>20}{'floor +/-':>12}"
    print(header)
    print("-" * len(header))
    for r in [r for r in records if r["kappa"] == REPORT_KAPPA]:
        print(
            f"{r['n']:>8,}{r['plugin_bias']:>15.4f}{r['mm_bias']:>20.4f}{r['floor']:>12.4f}"
        )

    print(
        "\nMiller-Madow removes most of the plug-in's positive (over-coordination)\n"
        "bias; both the residual bias and the noise floor shrink with N. The floor is\n"
        "the smallest cross-setup difference this estimator could resolve at each N."
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

    max_n = max(r["n"] for r in records)
    at_max_n = [r for r in records if r["n"] == max_n]
    at_kappa = [r for r in records if r["kappa"] == REPORT_KAPPA]

    fig, (ax_true, ax_bias) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: estimate vs known value across the coordination range.
    true_vals = [r["true_cmi"] for r in at_max_n]
    est_vals = [r["mm_mean"] for r in at_max_n]
    floors = [r["floor"] for r in at_max_n]
    lim = max(true_vals) * 1.08
    ax_true.plot([0, lim], [0, lim], color="#999999", lw=1, ls="--", label="exact recovery")
    ax_true.errorbar(true_vals, est_vals, yerr=floors, fmt="o", color="#1f77b4",
                     capsize=4, ms=7, label="Miller-Madow estimate")
    ax_true.set_xlabel("true  I(a1; a2 | s)  [bits]")
    ax_true.set_ylabel("estimated  I(a1; a2 | s)  [bits]")
    ax_true.set_title(f"Trueness across coordination strength\n(N = {max_n:,}, error bars = 95% noise floor)")
    ax_true.set_xlim(-0.05, lim)
    ax_true.set_ylim(-0.05, lim)
    ax_true.legend(loc="best", fontsize=9)

    # Right: bias (plug-in vs corrected) and the floor vs N, against the ideal rate.
    # Miller-Madow bias is plotted as markers, not a line -- it is near zero, so a
    # line would only trace Monte-Carlo scatter on the log axis.
    ns = np.array([r["n"] for r in at_kappa])
    floors = np.array([r["floor"] for r in at_kappa])
    ax_bias.plot(ns, [abs(r["plugin_bias"]) for r in at_kappa], "o-", color="#d62728",
                 label="|bias|  plug-in")
    ax_bias.plot(ns, [abs(r["mm_bias"]) for r in at_kappa], "s", color="#1f77b4", ms=6,
                 label="|bias|  Miller-Madow")
    ax_bias.plot(ns, floors, "^--", color="#777777", label="noise floor (95% half-width)")
    ax_bias.plot(ns, floors[0] * np.sqrt(ns[0] / ns), ":", color="#bbbbbb", lw=1.5,
                 label="ideal  1 / sqrt(N)")
    ax_bias.set_xscale("log")
    ax_bias.set_yscale("log")
    ax_bias.set_xlabel("sample size  N")
    ax_bias.set_ylabel("bits")
    ax_bias.set_title("Bias and noise floor vs N  "
                      f"(kappa = {REPORT_KAPPA})\nMiller-Madow removes most of the plug-in bias; floor tracks 1/sqrt(N)")
    ax_bias.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 calibration -- conditional-MI coordination metric on an analytic phantom\n"
        "estimator recovers a known I(a1; a2 | s); bias and the resample floor are quantified before any LLM-agent run",
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
    save_csv(records, results_dir / "calibration.csv")
    plot(records, results_dir / "calibration.png")


if __name__ == "__main__":
    main()
