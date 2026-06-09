# Phase 0 — estimator calibration on a known-answer phantom

Before a collective-behavior metric is run on LLM-agent rollouts, its estimator
is calibrated against a synthetic source whose value is known in closed form. On
real data the estimator recovers a property it cannot check against ground truth;
here the ground truth is constructed, so estimator **bias** and the sampling
**noise floor** are measured directly. This is the de-risk gate the rest of the
program runs behind, at ~zero compute.

The first metric calibrated here is a **coordination** measure: the conditional
pairwise mutual information `I(a1; a2 | s)` between two agents' actions, in bits —
high when agents coordinate within a context `s`, zero when they act independently.

## The phantom

A coupled categorical source (`coupled_source.py`) with a tunable coupling
`kappa` per context:

```
a1 | s      ~ Uniform{0..K-1}
a2 | a1, s  =  a1 with probability kappa  (coordination),  else Uniform
```

so `I(a1; a2 | s)` is known exactly — 0 at `kappa = 0`, `log2(K)` at `kappa = 1`,
and monotone in between. Because the metric is *conditional*, the estimator
stratifies by `s` and averages over `P(s)`; conditioning splits the sample across
strata, which is where finite-sample bias bites.

## What it reports

`calibrate.py` sweeps coupling strength and sample size and reports:

- **trueness** — does the estimate track the known value across the coordination
  range, within a stated bound;
- **bias** — plug-in vs the bias-corrected **Miller-Madow** estimator, vs sample size;
- **noise floor** — the 95% interval half-width of the estimate over independent
  re-samples at a fixed configuration. A cross-setup difference below the floor is
  not a result.

## Result (default run)

- At `N = 10,000` the Miller-Madow estimate recovers the known `I(a1; a2 | s)`
  across the coordination range to within **~0.001 bits**.
- Miller-Madow roughly **halves** the plug-in's small-N over-coordination bias
  (0.071 vs 0.152 bits at `N = 100`, `kappa = 0.6`); residual bias is ~0 by
  `N = 1,000`.
- The **noise floor** is **0.025 bits at `N = 10,000`**, rising to 0.29 at
  `N = 100` — the smallest difference this estimator could resolve at each N.

Figure: [`results/calibration.png`](results/calibration.png) · table:
`results/calibration.csv`.

## Run

```bash
uv run python phase0/calibrate.py     # the sweep -> results/
uv run python phase0/test_phase0.py   # known-answer tests (or via pytest)
```

## Scope

This calibrates one metric's estimator on a known answer. It establishes the
floor under which a later "this metric does / does not travel across LLM-agent
setups" result is interpretable rather than an artifact of estimation — it does
not, on its own, evidence cross-setup invariance, which is the next phase. The
remaining battery members (a fragility/antifragility measure, a
misalignment-propagation measure) are calibrated the same way, each against a
source built so its value is known.

## Files

- `coupled_source.py` — the phantom: sampler + closed-form `I(a1; a2 | s)`.
- `mi_estimators.py` — plug-in and Miller-Madow conditional-MI estimators.
- `calibrate.py` — the sweep, summary tables, CSV + figure.
- `test_phase0.py` — known-answer tests for the phantom and estimators.
