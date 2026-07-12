"""Phase 0 negative control: conditioning on a jointly-produced state manufactures dependence.

`calibrate_confound.py` shows conditioning *rescuing* the coordination metric —
the context there is a common cause, and conditioning strips a false positive
out. This script is the mirror, on the `collider_source` phantom: the state s is
a common EFFECT (jointly produced by the two agents), no coupling exists at all,
and the calibration shows

  - plain MI sitting at its floor around the true value zero — the CORRECT
    reading for this phantom;
  - conditional MI reading the closed-form manufactured value
    1 - H_b((1 - rho1*rho2*lam)/2) — a dependence that conditioning itself
    created. The estimator is faithful; the *conditioning choice* is wrong.

Together the pair calibrates the conditioning rule, not just the estimator:
condition on the shared CAUSE (confound phantom); do not condition on the
shared EFFECT (this phantom). The rule for real audits: condition on variables
causally upstream of the actions compared — exogenous inputs, task
specification, configuration — never on state the agents jointly produced.

Run:  uv run python phase0/calibrate_collider.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from collider_source import sample, true_conditional_mi, true_unconditional_mi
from mi_estimators import conditional_mi, miller_madow_mi

RHO = (0.0, 0.25, 0.5, 0.75, 1.0)       # behavioral persistence (rho1 = rho2 = rho)
LAM = (0.0, 0.25, 0.5, 0.75, 1.0)       # state fidelity
SAMPLE_SIZES = (300, 1000, 3000, 10_000)
REPEATS = 200
SEED = 0
REPORT_RHO = 0.75                        # persistence level for the lambda view
REPORT_LAM = 1.0                         # fidelity level for the rho view
REPORT_N = 10_000


def run_sweep(rho_grid=RHO, lam_grid=LAM, sample_sizes=SAMPLE_SIZES, *,
              repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for rho in rho_grid:
        for lam in lam_grid:
            true_cmi = true_conditional_mi(rho, rho, lam)
            true_mi = true_unconditional_mi(rho, rho, lam)
            for n in sample_sizes:
                cmi = np.empty(repeats)
                mi = np.empty(repeats)
                for r in range(repeats):
                    s, a1, a2 = sample(rho, rho, lam, n, rng)
                    cmi[r] = conditional_mi(s, a1, a2, 2, 2, corrected=True)
                    mi[r] = miller_madow_mi(a1, a2, 2)
                lo, hi = np.percentile(cmi, [2.5, 97.5])
                mlo, mhi = np.percentile(mi, [2.5, 97.5])
                records.append(
                    {
                        "rho": rho,
                        "lam": lam,
                        "n": n,
                        "true_mi": true_mi,
                        "true_cmi": true_cmi,
                        "mi_mean": float(mi.mean()),
                        "mi_floor": float((mhi - mlo) / 2.0),
                        "cmi_mean": float(cmi.mean()),
                        "cmi_bias": float(cmi.mean() - true_cmi),
                        "cmi_floor": float((hi - lo) / 2.0),
                    }
                )
    return records


def summarize(records: list[dict]) -> None:
    at_n = [r for r in records if r["n"] == REPORT_N]

    print(f"\nMANUFACTURED DEPENDENCE -- no coupling exists; true I(a1; a2) = 0 throughout"
          f"  (lam = {REPORT_LAM}, N = {REPORT_N:,}, Miller-Madow)")
    print("plain MI must stay at the floor (the correct reading); conditioning on the"
          " jointly-produced state manufactures the closed-form value")
    header = f"{'rho':>6}{'true CMI':>10}{'CMI est':>10}{'CMI floor':>11}{'MI est':>10}{'MI floor':>10}"
    print(header)
    print("-" * len(header))
    for r in [r for r in at_n if r["lam"] == REPORT_LAM]:
        print(
            f"{r['rho']:>6.2f}{r['true_cmi']:>10.4f}{r['cmi_mean']:>10.4f}"
            f"{r['cmi_floor']:>11.4f}{r['mi_mean']:>10.4f}{r['mi_floor']:>10.4f}"
        )

    print(f"\nSTATE FIDELITY -- the more faithfully the joint state is recorded, the larger"
          f" the artifact  (rho = {REPORT_RHO}, N = {REPORT_N:,})")
    header = f"{'lam':>6}{'true CMI':>10}{'CMI est':>10}{'CMI bias':>10}{'CMI floor':>11}"
    print(header)
    print("-" * len(header))
    for r in [r for r in at_n if r["rho"] == REPORT_RHO]:
        print(
            f"{r['lam']:>6.2f}{r['true_cmi']:>10.4f}{r['cmi_mean']:>10.4f}"
            f"{r['cmi_bias']:>10.4f}{r['cmi_floor']:>11.4f}"
        )

    print(
        "\nThe estimator is faithful in both columns -- it recovers the closed-form value\n"
        "of the quantity it was pointed at. The failure is the conditioning choice: s is\n"
        "an effect of the agents' behavior, not a cause of it, and stratifying on it\n"
        "manufactures dependence between agents that share nothing. Rule, paired with\n"
        "the confound calibration: condition on the shared cause; never on the shared effect."
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

    at_n = [r for r in records if r["n"] == REPORT_N]
    rho_view = [r for r in at_n if r["lam"] == REPORT_LAM]
    lam_view = [r for r in at_n if r["rho"] == REPORT_RHO]

    fig, (ax_rho, ax_lam) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: persistence sweep at perfect state fidelity.
    rho = [r["rho"] for r in rho_view]
    ax_rho.plot(rho, [r["true_cmi"] for r in rho_view], "--", color="#999999", lw=1,
                label="true  I(a1; a2 | s)  (manufactured)")
    ax_rho.errorbar(rho, [r["cmi_mean"] for r in rho_view],
                    yerr=[r["cmi_floor"] for r in rho_view], fmt="s", color="#1f77b4",
                    capsize=4, ms=7, label="estimated conditional MI (95% floor)")
    ax_rho.axhline(0.0, color="#d62728", lw=1, ls="--",
                   label="true  I(a1; a2) = 0  (no coupling exists)")
    ax_rho.errorbar(rho, [r["mi_mean"] for r in rho_view],
                    yerr=[r["mi_floor"] for r in rho_view], fmt="o", color="#d62728",
                    capsize=4, ms=7, label="estimated plain MI (95% floor)")
    ax_rho.set_xlabel(f"behavioral persistence  rho   (state fidelity lam = {REPORT_LAM})")
    ax_rho.set_ylabel("information  [bits]")
    ax_rho.set_title("Conditioning on a jointly-produced state manufactures dependence\n"
                     "plain MI reads the truth (zero); conditional MI reads the artifact")
    ax_rho.legend(loc="best", fontsize=8)

    # Right: state-fidelity sweep at fixed persistence.
    lam = [r["lam"] for r in lam_view]
    ax_lam.plot(lam, [r["true_cmi"] for r in lam_view], "--", color="#999999", lw=1,
                label="true  I(a1; a2 | s)  (manufactured)")
    ax_lam.errorbar(lam, [r["cmi_mean"] for r in lam_view],
                    yerr=[r["cmi_floor"] for r in lam_view], fmt="s", color="#1f77b4",
                    capsize=4, ms=7, label="estimated conditional MI (95% floor)")
    ax_lam.axhline(0.0, color="#d62728", lw=1, ls="--",
                   label="true  I(a1; a2) = 0")
    ax_lam.set_xlabel(f"state fidelity  lam   (persistence rho = {REPORT_RHO})")
    ax_lam.set_ylabel("conditional MI  I(a1; a2 | s)  [bits]")
    ax_lam.set_title("The better the joint state is recorded, the larger the artifact\n"
                     "a faithful log of the shared effect is the worst thing to condition on")
    ax_lam.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 negative control -- the collider phantom: zero coupling, by construction\n"
        f"conditioning on the agents' jointly-produced state fabricates coordination  (N = {REPORT_N:,})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=130)
    print(f"\nFigure -> {path}")


def heatmap(records: list[dict], path: Path) -> None:
    """Plain MI vs conditional MI over the (rho, lam) grid.

    Plain MI is flat at zero everywhere -- the truth. Conditional MI rises with
    the product rho^2 * lam: the artifact needs BOTH persistent behavior and a
    faithful record of the jointly-produced state.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhos = sorted({r["rho"] for r in records})
    lams = sorted({r["lam"] for r in records})
    at_n = {(r["rho"], r["lam"]): r for r in records if r["n"] == REPORT_N}
    grid_mi = np.array([[at_n[(rho, lm)]["mi_mean"] for lm in lams] for rho in rhos])
    grid_cmi = np.array([[at_n[(rho, lm)]["cmi_mean"] for lm in lams] for rho in rhos])
    vmax = float(grid_cmi.max())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, grid, title in (
        (axes[0], grid_mi, "Plain MI   I(a1; a2)      (truth: 0 everywhere)"),
        (axes[1], grid_cmi, "Conditional MI   I(a1; a2 | s)      (the artifact)"),
    ):
        im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(lams)))
        ax.set_xticklabels([f"{x:.2f}" for x in lams])
        ax.set_yticks(range(len(rhos)))
        ax.set_yticklabels([f"{y:.2f}" for y in rhos])
        ax.set_xlabel("state fidelity  lam")
        ax.set_ylabel("behavioral persistence  rho")
        ax.set_title(title)
        for i in range(len(rhos)):
            for j in range(len(lams)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.55 * vmax else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="bits")

    fig.suptitle(
        "Phase 0 negative control -- plain MI vs conditional MI over the (rho, lam) grid  (N = 10,000)\n"
        "no coupling exists anywhere on this grid; every nonzero cell on the right is manufactured by conditioning",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=130)
    print(f"Heatmap -> {path}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    records = run_sweep()
    summarize(records)
    save_csv(records, results_dir / "collider_calibration.csv")
    plot(records, results_dir / "collider_calibration.png")
    heatmap(records, results_dir / "collider_heatmap.png")


if __name__ == "__main__":
    main()
