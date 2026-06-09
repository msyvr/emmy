# Phase 0 — estimator calibration on known-answer phantoms

Before a collective-behavior metric is run on LLM-agent rollouts, its estimator
is calibrated against a synthetic source whose value is known in closed form. On
real data the estimator recovers a property it cannot check against ground truth;
here the ground truth is constructed, so estimator **bias** and the sampling
**noise floor** are measured directly. This is the de-risk gate the rest of the
program runs behind, at ~zero compute.

Each metric in the battery is calibrated this way. Two are done so far.

## What each calibration reports

- **trueness** — does the estimate track the known value across the metric's
  range, within a stated bound (sign included, for fragility);
- **bias** — the finite-sample bias of the estimator vs sample size / budget;
- **noise floor** — the 95% interval half-width of the estimate over independent
  re-samples at a fixed configuration. A difference below the floor is not a
  result; with the floor printed, a later "this metric doesn't travel" cannot be
  confused with "we estimated it badly".

## Coordination — conditional mutual information `I(a1; a2 | s)`

High when two agents coordinate within a context `s`, zero when independent. The
phantom (`coupled_source.py`) is a coupled categorical source with tunable
coupling `kappa`, so `I(a1; a2 | s)` is known exactly (0 at `kappa=0`, `log2(K)`
at `kappa=1`). The metric is conditional, so the estimator stratifies by `s` and
averages — and conditioning splits the sample, which is where bias bites.

**Result (default run):** at `N = 10,000` the Miller-Madow estimator recovers the
known value to **~0.001 bits** across the coordination range; it roughly halves
the plug-in's small-N over-coordination bias (0.071 vs 0.152 bits at `N=100`); the
**noise floor is 0.025 bits at `N=10,000`** (0.29 at `N=100`).
Run: `uv run python phase0/calibrate.py` · figure: `results/calibration.png`.

## Fragility / antifragility — response curvature

The fragility family (CAFE and kin) reduces to the curvature of a stress-response
`R(sigma)`: `d2R/dsigma2 < 0` is fragile (concave), `> 0` antifragile (convex). The
phantom (`fragility_source.py`) has a known curvature `kappa`, observed as noisy
replicate rollouts; the estimator (`curvature_estimators.py`) fits the curve and
recovers it. Curvature is a weak second-order feature, so the **noise floor is the
load-bearing number** — it sets the budget at which the *sign* (fragile vs
antifragile) is callable.

**Result (default run):** the estimator is near-unbiased (bias ~0.003 at 900
rollouts) and recovers the sign across the range. The **floor falls from 0.69
(budget 9) to 0.066 (budget 900)**, so a `|kappa| = 1` fragility becomes resolvable
by ~budget 27, while a `|kappa| = 0.5` needs ~budget 270+. The deliverable is that
resolution limit, before any LLM-agent stress run.
Run: `uv run python phase0/calibrate_fragility.py` · figure:
`results/fragility_calibration.png`.

## Scope

These calibrate metric *estimators* on known answers. They establish the floor
under which a later "this metric does / does not travel across LLM-agent setups"
result is interpretable rather than an artifact of estimation — they do not, on
their own, evidence cross-setup invariance, which is the next phase. The remaining
battery member (a misalignment-propagation measure) is calibrated the same way,
against a planted-contagion source whose value is known.

## Run

```bash
uv run python phase0/calibrate.py            # coordination metric -> results/
uv run python phase0/calibrate_fragility.py  # fragility metric    -> results/
uv run python phase0/test_phase0.py          # coordination known-answer tests
uv run python phase0/test_fragility.py       # fragility known-answer tests
```

## Files

- `coupled_source.py` / `mi_estimators.py` / `calibrate.py` / `test_phase0.py` —
  coordination metric: phantom, plug-in + Miller-Madow estimators, sweep, tests.
- `fragility_source.py` / `curvature_estimators.py` / `calibrate_fragility.py` /
  `test_fragility.py` — fragility metric: known-curvature source, curvature
  estimator, sweep, tests.
- `results/` — calibration figures and CSVs.
