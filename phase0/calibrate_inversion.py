"""Phase 0 cross-environment inversion: the number that travels is kappa-hat, not the bits.

The invariance sweep's core move, exercised where the answer is known exactly.
The same copy-mechanism coupling kappa is planted in two structurally different
environments:

    lean  --  K = 2 actions, 2 contexts, uniform context mix
    rich  --  K = 8 actions, 3 contexts, skewed context mix (0.5, 0.3, 0.2)

The raw conditional-MI readings then differ BY CONSTRUCTION (at kappa = 0.6 the
closed forms give ~0.28 vs ~1.08 bits — same coupling, ~3.9x the bits): raw
information values are alphabet- and structure-relative, so comparing them
across environments compares the environments. That is the trap.

The fix: each environment's calibration curve f_E(kappa) is known (measured, on
a real system); invert it. Estimate the reading, map it back through the
environment's own curve, and compare the recovered couplings. Same kappa in,
same kappa-hat out — within floors propagated through the inversion — is what
"the measurement travels" means, operationally.

Honesty note the figure makes visible: inversion is ill-conditioned where the
curve is flat. MI has a quadratic onset at independence, so the kappa-hat floor
widens near kappa = 0 — a property of the problem, reported rather than hidden.

Run:  uv run python phase0/calibrate_inversion.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from coupled_source import sample, true_conditional_mi
from inversion import Environment, calibration_curve, invert_reading
from mi_estimators import conditional_mi

ENVS = (
    Environment("lean (K=2, 2 uniform contexts)", 2, (0.5, 0.5)),
    Environment("rich (K=8, 3 skewed contexts)", 8, (0.5, 0.3, 0.2)),
)
KAPPAS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
SAMPLE_SIZES = (300, 1000, 3000, 10_000)
REPEATS = 200
SEED = 0
REPORT_N = 10_000


def run_sweep(envs=ENVS, kappas=KAPPAS, sample_sizes=SAMPLE_SIZES, *,
              repeats=REPEATS, seed=SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for env in envs:
        grid, readings = calibration_curve(env.n_actions)
        for kappa in kappas:
            kappa_s = np.full(env.n_contexts, kappa)
            true_cmi = true_conditional_mi(kappa_s, env.p_s, env.n_actions)
            for n in sample_sizes:
                est = np.empty(repeats)
                for r in range(repeats):
                    s, a1, a2 = sample(kappa_s, env.p_s, env.n_actions, n, rng)
                    est[r] = conditional_mi(s, a1, a2, env.n_actions, env.n_contexts,
                                            corrected=True)
                khat = invert_reading(est, grid, readings)
                lo, hi = np.percentile(est, [2.5, 97.5])
                klo, khi = np.percentile(khat, [2.5, 97.5])
                records.append(
                    {
                        "env": env.name,
                        "n_actions": env.n_actions,
                        "kappa": kappa,
                        "n": n,
                        "true_cmi": true_cmi,
                        "cmi_mean": float(est.mean()),
                        "cmi_floor": float((hi - lo) / 2.0),
                        "khat_mean": float(np.mean(khat)),
                        "khat_bias": float(np.mean(khat) - kappa),
                        "khat_floor": float((khi - klo) / 2.0),
                    }
                )
    return records


def summarize(records: list[dict]) -> None:
    lean, rich = ENVS[0].name, ENVS[1].name
    at = {(r["env"], r["kappa"]): r for r in records if r["n"] == REPORT_N}

    print(f"\nTHE TRAP -- same coupling, different bits  (N = {REPORT_N:,}, Miller-Madow)")
    print("raw conditional-MI readings are alphabet/structure-relative by construction")
    header = (f"{'kappa':>7}{'true lean':>11}{'est lean':>10}{'true rich':>11}"
              f"{'est rich':>10}{'rich/lean':>11}")
    print(header)
    print("-" * len(header))
    for kappa in KAPPAS:
        a, b = at[(lean, kappa)], at[(rich, kappa)]
        ratio = b["true_cmi"] / a["true_cmi"] if a["true_cmi"] > 0 else float("nan")
        print(
            f"{kappa:>7.2f}{a['true_cmi']:>11.4f}{a['cmi_mean']:>10.4f}"
            f"{b['true_cmi']:>11.4f}{b['cmi_mean']:>10.4f}{ratio:>11.2f}"
        )

    print(f"\nTHE FIX -- invert each environment's calibration curve  (N = {REPORT_N:,})")
    print("recovered couplings agree with the truth and with each other, within floors")
    header = (f"{'kappa':>7}{'khat lean':>11}{'floor':>8}{'khat rich':>11}{'floor':>8}"
              f"{'|delta|':>9}{'combined':>10}")
    print(header)
    print("-" * len(header))
    for kappa in KAPPAS:
        a, b = at[(lean, kappa)], at[(rich, kappa)]
        delta = abs(a["khat_mean"] - b["khat_mean"])
        combined = a["khat_floor"] + b["khat_floor"]
        mark = "  agree" if delta <= combined else "  APART"
        print(
            f"{kappa:>7.2f}{a['khat_mean']:>11.4f}{a['khat_floor']:>8.4f}"
            f"{b['khat_mean']:>11.4f}{b['khat_floor']:>8.4f}{delta:>9.4f}{combined:>10.4f}{mark}"
        )

    print(
        "\nComparing raw readings across environments compares the environments -- the\n"
        "left table's ratio column is pure structure, not coupling. Inverting each\n"
        "environment's own calibration curve recovers the generative coupling, and THAT\n"
        "number travels. On real systems the closed-form curve is replaced by a measured\n"
        "dose-response curve on engineered couplings; the inversion logic is identical.\n"
        "The kappa-hat floor widens near kappa = 0, where the curve is flat -- inversion\n"
        "is ill-conditioned exactly where MI has its quadratic onset; a property of the\n"
        "problem, printed rather than hidden."
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

    colors = {ENVS[0].name: "#1f77b4", ENVS[1].name: "#ff7f0e"}
    short = {ENVS[0].name: "lean (K=2)", ENVS[1].name: "rich (K=8)"}
    at_n = [r for r in records if r["n"] == REPORT_N]

    fig, (ax_raw, ax_inv) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: the trap. Raw readings vs kappa, per environment, with closed forms.
    kappa_fine = np.linspace(0.0, 0.9, 200)
    for env in ENVS:
        rows = [r for r in at_n if r["env"] == env.name]
        ks = [r["kappa"] for r in rows]
        from inversion import analytic_curve
        ax_raw.plot(kappa_fine, analytic_curve(kappa_fine, env.n_actions), "--",
                    color=colors[env.name], lw=1,
                    label=f"true curve, {short[env.name]}")
        ax_raw.errorbar(ks, [r["cmi_mean"] for r in rows],
                        yerr=[r["cmi_floor"] for r in rows], fmt="o",
                        color=colors[env.name], capsize=4, ms=6,
                        label=f"estimated, {short[env.name]} (95% floor)")
    ax_raw.set_xlabel("planted coupling  kappa   (same mechanism in both environments)")
    ax_raw.set_ylabel("conditional MI  I(a1; a2 | s)  [bits]")
    ax_raw.set_title("The trap: same coupling, different bits\n"
                     "raw readings are environment-relative by construction")
    ax_raw.legend(loc="best", fontsize=8)

    # Right: the fix. Recovered kappa-hat vs true kappa, per environment.
    ax_inv.plot([0, 0.9], [0, 0.9], "--", color="#999999", lw=1, label="identity (perfect recovery)")
    for env in ENVS:
        rows = [r for r in at_n if r["env"] == env.name]
        ks = [r["kappa"] for r in rows]
        ax_inv.errorbar(ks, [r["khat_mean"] for r in rows],
                        yerr=[r["khat_floor"] for r in rows], fmt="s",
                        color=colors[env.name], capsize=4, ms=6,
                        label=f"recovered kappa-hat, {short[env.name]} (95% floor)")
    ax_inv.set_xlabel("planted coupling  kappa")
    ax_inv.set_ylabel("recovered coupling  kappa-hat")
    ax_inv.set_title("The fix: invert each environment's own calibration curve\n"
                     "recovered couplings agree across environments, within floors")
    ax_inv.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Phase 0 cross-environment inversion -- the number that travels is the recovered coupling,"
        f" not the bits  (N = {REPORT_N:,})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    print(f"\nFigure -> {path}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    records = run_sweep()
    summarize(records)
    save_csv(records, results_dir / "inversion_calibration.csv")
    plot(records, results_dir / "inversion_calibration.png")


if __name__ == "__main__":
    main()
