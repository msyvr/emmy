"""Phase 0 discriminant-validity calibration: does conditional MI reject a common cause?

`calibrate.py` shows the conditional-MI estimator recovers a known coupling
(sensitivity). This script tests the complementary property -- specificity -- on
the two-knob `confound_source`:

  - common cause only (kappa_link = 0, kappa_cc > 0): plain MI is positive, but the
    *conditional* MI must sit at its noise floor. This is the confound the metric
    exists to reject -- two agents that look coupled only because a shared context
    drives both (the same-base-model / shared-prompt case, in the LLM regime).
  - genuine coupling (kappa_link > 0): conditional MI must still fire, even with a
    common cause present.

Together with `calibrate.py` this makes the coordination metric a full
sensitivity + specificity instrument before any LLM-agent run: it fires on genuine
coordination and does *not* fire on a shared-cause look-alike.

Run:  uv run python phase0/calibrate_confound.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from confound_source import sample, true_conditional_mi, true_unconditional_mi
from mi_estimators import conditional_mi, miller_madow_mi

N_ACTIONS = 4
N_CONTEXTS = 2
PREFERRED = (0, 1)          # distinct preferred actions -> the context is a common cause
P_S = (0.5, 0.5)
KAPPA_CC = (0.0, 0.3, 0.6, 0.9)
KAPPA_LINK = (0.0, 0.3, 0.6, 0.9)
SAMPLE_SIZES = (300, 1000, 3000, 10_000)
REPEATS = 200
SEED = 0
REPORT_CC = 0.6            # common-cause level for the sensitivity view
REPORT_N = 10_000


def run_sweep(kappa_cc=KAPPA_CC, kappa_link=KAPPA_LINK, sample_sizes=SAMPLE_SIZES, *,
              n_actions=N_ACTIONS, n_contexts=N_CONTEXTS, preferred=PREFERRED, p_s=P_S,
              repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for kcc in kappa_cc:
        for klink in kappa_link:
            true_cmi = true_conditional_mi(kcc, klink, preferred, p_s, n_actions)
            true_mi = true_unconditional_mi(kcc, klink, preferred, p_s, n_actions)
            for n in sample_sizes:
                cmi = np.empty(repeats)
                mi = np.empty(repeats)
                for r in range(repeats):
                    s, a1, a2 = sample(kcc, klink, preferred, p_s, n_actions, n, rng)
                    cmi[r] = conditional_mi(s, a1, a2, n_actions, n_contexts, corrected=True)
                    mi[r] = miller_madow_mi(a1, a2, n_actions)
                lo, hi = np.percentile(cmi, [2.5, 97.5])
                records.append(
                    {
                        "kappa_cc": kcc,
                        "kappa_link": klink,
                        "n": n,
                        "true_mi": true_mi,
                        "true_cmi": true_cmi,
                        "mi_mean": float(mi.mean()),
                        "cmi_mean": float(cmi.mean()),
                        "cmi_bias": float(cmi.mean() - true_cmi),
                        "cmi_floor": float((hi - lo) / 2.0),
                    }
                )
    return records


def summarize(records: list[dict]) -> None:
    at_n = [r for r in records if r["n"] == REPORT_N]

    print(f"\nSPECIFICITY -- pure common cause (kappa_link = 0, N = {REPORT_N:,}, Miller-Madow)")
    print("conditional MI must stay at the floor while plain MI is fooled by the shared context")
    header = f"{'kappa_cc':>9}{'true MI':>10}{'MI est':>10}{'true CMI':>10}{'CMI est':>10}{'CMI floor':>11}"
    print(header)
    print("-" * len(header))
    for r in [r for r in at_n if r["kappa_link"] == 0.0]:
        print(
            f"{r['kappa_cc']:>9.2f}{r['true_mi']:>10.4f}{r['mi_mean']:>10.4f}"
            f"{r['true_cmi']:>10.4f}{r['cmi_mean']:>10.4f}{r['cmi_floor']:>11.4f}"
        )

    print(f"\nSENSITIVITY -- with a common cause present (kappa_cc = {REPORT_CC}, N = {REPORT_N:,})")
    print("conditional MI must still track the genuine coupling")
    header = f"{'kappa_link':>11}{'true CMI':>10}{'CMI est':>10}{'CMI bias':>10}{'CMI floor':>11}"
    print(header)
    print("-" * len(header))
    for r in [r for r in at_n if r["kappa_cc"] == REPORT_CC]:
        print(
            f"{r['kappa_link']:>11.2f}{r['true_cmi']:>10.4f}{r['cmi_mean']:>10.4f}"
            f"{r['cmi_bias']:>10.4f}{r['cmi_floor']:>11.4f}"
        )

    print(
        "\nLeft column is the confound: plain MI rises with the shared context while\n"
        "conditional MI stays within its noise floor of zero -- the metric does not\n"
        "mistake a common cause for coordination. Right column confirms it still fires\n"
        "on a genuine link even when that common cause is present."
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
    confound = [r for r in at_n if r["kappa_link"] == 0.0]
    sensitivity = [r for r in at_n if r["kappa_cc"] == REPORT_CC]

    fig, (ax_conf, ax_sens) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: pure common cause. Plain MI is fooled; conditional MI rejects it.
    cc = [r["kappa_cc"] for r in confound]
    ax_conf.plot(cc, [r["true_mi"] for r in confound], "--", color="#d62728", lw=1,
                 label="true  I(a1; a2)  (plain MI)")
    ax_conf.plot(cc, [r["mi_mean"] for r in confound], "o", color="#d62728", ms=7,
                 label="estimated plain MI")
    ax_conf.axhline(0.0, color="#999999", lw=1, ls="--", label="true  I(a1; a2 | s) = 0")
    ax_conf.errorbar(cc, [r["cmi_mean"] for r in confound],
                     yerr=[r["cmi_floor"] for r in confound], fmt="s", color="#1f77b4",
                     capsize=4, ms=7, label="estimated conditional MI (95% floor)")
    ax_conf.set_xlabel("common-cause strength  kappa_cc   (kappa_link = 0)")
    ax_conf.set_ylabel("information  [bits]")
    ax_conf.set_title("Specificity: conditional MI rejects a pure common cause\n"
                      "plain MI rises with the shared context; conditional MI stays at the floor")
    ax_conf.legend(loc="best", fontsize=8)

    # Right: genuine coupling with a common cause present. Conditional MI tracks truth.
    link = [r["kappa_link"] for r in sensitivity]
    lim = max(r["true_cmi"] for r in sensitivity) * 1.1 + 0.05
    ax_sens.plot(link, [r["true_cmi"] for r in sensitivity], "--", color="#999999", lw=1,
                 label="true  I(a1; a2 | s)")
    ax_sens.errorbar(link, [r["cmi_mean"] for r in sensitivity],
                     yerr=[r["cmi_floor"] for r in sensitivity], fmt="o", color="#1f77b4",
                     capsize=4, ms=7, label="estimated conditional MI (95% floor)")
    ax_sens.set_xlabel(f"genuine coupling  kappa_link   (kappa_cc = {REPORT_CC})")
    ax_sens.set_ylabel("conditional MI  I(a1; a2 | s)  [bits]")
    ax_sens.set_ylim(-0.05, lim)
    ax_sens.set_title("Sensitivity holds under a common cause\n"
                      "conditional MI still tracks the genuine link")
    ax_sens.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 discriminant validity -- conditional-MI coordination metric on the two-knob phantom\n"
        f"the metric fires on genuine coupling and not on a shared-cause look-alike  (N = {REPORT_N:,})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=130)
    print(f"\nFigure -> {path}")


def heatmap(records: list[dict], path: Path) -> None:
    """Plain MI vs conditional MI over the full (kappa_cc, kappa_link) grid.

    The discriminant signal is the no-link edge (kappa_link = 0): plain MI climbs
    with the common cause while conditional MI stays at zero. (Conditional MI does
    fall off at very high kappa_cc for a fixed link, as the per-context marginal
    concentrates and leaves less within-context variation -- real, not a confound
    effect.)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ccs = sorted({r["kappa_cc"] for r in records})
    links = sorted({r["kappa_link"] for r in records})
    at_n = {(r["kappa_cc"], r["kappa_link"]): r for r in records if r["n"] == REPORT_N}
    grid_mi = np.array([[at_n[(cc, lk)]["mi_mean"] for lk in links] for cc in ccs])
    grid_cmi = np.array([[at_n[(cc, lk)]["cmi_mean"] for lk in links] for cc in ccs])
    vmax = float(grid_mi.max())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, grid, title in (
        (axes[0], grid_mi, "Plain MI   I(a1; a2)"),
        (axes[1], grid_cmi, "Conditional MI   I(a1; a2 | s)"),
    ):
        im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(links)))
        ax.set_xticklabels([f"{x:.1f}" for x in links])
        ax.set_yticks(range(len(ccs)))
        ax.set_yticklabels([f"{y:.1f}" for y in ccs])
        ax.set_xlabel("genuine coupling  kappa_link")
        ax.set_ylabel("common cause  kappa_cc")
        ax.set_title(title)
        for i in range(len(ccs)):
            for j in range(len(links)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.55 * vmax else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="bits")

    fig.suptitle(
        "Phase 0 discriminant validity -- plain MI vs conditional MI over the two-knob grid  (N = 10,000)\n"
        "along the no-link edge (kappa_link = 0) plain MI climbs with the common cause; conditional MI stays at zero",
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
    save_csv(records, results_dir / "confound_calibration.csv")
    plot(records, results_dir / "confound_calibration.png")
    heatmap(records, results_dir / "confound_heatmap.png")


if __name__ == "__main__":
    main()
