"""Phase 0 discriminant-validity calibration for misalignment-propagation.

`calibrate_propagation.py` calibrates the interventional contagion coefficient
(you control the dose). This script tests the *observational* case, where an
outside observer cannot intervene and must separate genuine spreading from a
shared trigger, on the two-knob `propagation_confound_source`:

  - shared trigger only (kappa_link = 0, kappa_cc > 0): the naive contagion slope
    is positive (the trigger correlates seed and target), but the *conditioned*
    slope must read zero -- nothing actually spread.
  - genuine contagion (kappa_link > 0): the conditioned slope must still recover it,
    even with a trigger present.

This is the propagation analogue of the coordination discriminant: did the
misalignment spread (seed -> target), or were both agents tripped by a shared
cause (a common prompt, a base model, an environmental event)?

Run:  uv run python phase0/calibrate_propagation_confound.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from propagation_confound_source import sample, true_conditioned_slope, true_naive_slope
from propagation_estimators import conditioned_contagion_slope, naive_contagion_slope

N_STRATA = 2
KAPPA_CC = (0.0, 0.1, 0.2, 0.3)
KAPPA_LINK = (0.0, 0.1, 0.2, 0.3)
SAMPLE_SIZES = (300, 1000, 3000, 10_000)
REPEATS = 200
SEED = 0
REPORT_CC = 0.3            # trigger strength for the sensitivity view
REPORT_N = 10_000


def run_sweep(kappa_cc=KAPPA_CC, kappa_link=KAPPA_LINK, sample_sizes=SAMPLE_SIZES, *,
              repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for kcc in kappa_cc:
        for klink in kappa_link:
            true_link = true_conditioned_slope(klink)
            true_naive = true_naive_slope(kcc, klink)
            for n in sample_sizes:
                naive = np.empty(repeats)
                cond = np.empty(repeats)
                for r in range(repeats):
                    s, zs, zt = sample(kcc, klink, n, rng)
                    naive[r] = naive_contagion_slope(zs, zt)
                    cond[r] = conditioned_contagion_slope(s, zs, zt, N_STRATA)
                lo, hi = np.percentile(cond, [2.5, 97.5])
                records.append(
                    {
                        "kappa_cc": kcc,
                        "kappa_link": klink,
                        "n": n,
                        "true_naive": true_naive,
                        "true_link": true_link,
                        "naive_mean": float(naive.mean()),
                        "cond_mean": float(cond.mean()),
                        "cond_bias": float(cond.mean() - true_link),
                        "cond_floor": float((hi - lo) / 2.0),
                    }
                )
    return records


def summarize(records: list[dict]) -> None:
    at_n = [r for r in records if r["n"] == REPORT_N]

    print(f"\nSPECIFICITY -- shared trigger only (kappa_link = 0, N = {REPORT_N:,})")
    print("the conditioned slope must read zero while the naive slope is fooled by the trigger")
    header = f"{'kappa_cc':>9}{'true naive':>12}{'naive est':>11}{'true link':>11}{'cond est':>10}{'cond floor':>12}"
    print(header)
    print("-" * len(header))
    for r in [r for r in at_n if r["kappa_link"] == 0.0]:
        print(
            f"{r['kappa_cc']:>9.2f}{r['true_naive']:>12.3f}{r['naive_mean']:>11.3f}"
            f"{r['true_link']:>11.3f}{r['cond_mean']:>10.3f}{r['cond_floor']:>12.3f}"
        )

    print(f"\nSENSITIVITY -- with a trigger present (kappa_cc = {REPORT_CC}, N = {REPORT_N:,})")
    print("the conditioned slope must still track the genuine contagion")
    header = f"{'kappa_link':>11}{'true link':>11}{'cond est':>10}{'cond bias':>11}{'cond floor':>12}"
    print(header)
    print("-" * len(header))
    for r in [r for r in at_n if r["kappa_cc"] == REPORT_CC]:
        print(
            f"{r['kappa_link']:>11.2f}{r['true_link']:>11.3f}{r['cond_mean']:>10.3f}"
            f"{r['cond_bias']:>11.3f}{r['cond_floor']:>12.3f}"
        )

    print(
        "\nLeft: a shared trigger drives the naive slope up while the conditioned slope\n"
        "stays at zero -- the misalignment did not spread, it was triggered. Right: the\n"
        "conditioned slope still recovers genuine contagion when a trigger is present."
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
    spec = [r for r in at_n if r["kappa_link"] == 0.0]
    sens = [r for r in at_n if r["kappa_cc"] == REPORT_CC]

    fig, (ax_spec, ax_sens) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: shared trigger only. Naive slope fooled; conditioned slope rejects it.
    cc = [r["kappa_cc"] for r in spec]
    ax_spec.plot(cc, [r["true_naive"] for r in spec], "--", color="#d62728", lw=1,
                 label="true naive slope")
    ax_spec.plot(cc, [r["naive_mean"] for r in spec], "o", color="#d62728", ms=7,
                 label="estimated naive slope")
    ax_spec.axhline(0.0, color="#999999", lw=1, ls="--", label="true genuine contagion = 0")
    ax_spec.errorbar(cc, [r["cond_mean"] for r in spec],
                     yerr=[r["cond_floor"] for r in spec], fmt="s", color="#1f77b4",
                     capsize=4, ms=7, label="estimated conditioned slope (95% floor)")
    ax_spec.set_xlabel("shared-trigger strength  kappa_cc   (kappa_link = 0)")
    ax_spec.set_ylabel("contagion slope")
    ax_spec.set_title("Specificity: the conditioned slope rejects a shared trigger\n"
                      "the naive slope rises with the trigger; the conditioned slope stays at zero")
    ax_spec.legend(loc="best", fontsize=8)

    # Right: genuine contagion with a trigger present. Conditioned slope tracks truth.
    link = [r["kappa_link"] for r in sens]
    lim = max(r["true_link"] for r in sens) * 1.2 + 0.03
    ax_sens.plot(link, [r["true_link"] for r in sens], "--", color="#999999", lw=1,
                 label="true genuine contagion")
    ax_sens.errorbar(link, [r["cond_mean"] for r in sens],
                     yerr=[r["cond_floor"] for r in sens], fmt="o", color="#1f77b4",
                     capsize=4, ms=7, label="estimated conditioned slope (95% floor)")
    ax_sens.set_xlabel(f"genuine contagion  kappa_link   (kappa_cc = {REPORT_CC})")
    ax_sens.set_ylabel("conditioned contagion slope")
    ax_sens.set_ylim(-0.03, lim)
    ax_sens.set_title("Sensitivity holds under a trigger\n"
                      "the conditioned slope still tracks genuine contagion")
    ax_sens.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 discriminant validity -- misalignment-propagation on the two-knob phantom\n"
        f"did it spread, or was it triggered? the conditioned slope tells them apart  (N = {REPORT_N:,})",
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
    save_csv(records, results_dir / "propagation_confound_calibration.csv")
    plot(records, results_dir / "propagation_confound_calibration.png")


if __name__ == "__main__":
    main()
