# Phase 0 — estimator calibration on known-answer phantoms

Before a collective-behavior metric is run on LLM-agent rollouts, its estimator
is calibrated against a synthetic source whose value is known in closed form. On
real data the estimator recovers a property it cannot check against ground truth;
here the ground truth is constructed, so estimator **bias** and the sampling
**noise floor** are measured directly. This is the de-risk gate the rest of the
program runs behind, at ~zero compute.

All three battery metrics are calibrated here.

## What each calibration reports

- **trueness** — does the estimate track the known value across the metric's
  range, within a stated bound (sign included, where the sign is the question);
- **bias** — the finite-sample bias of the estimator vs sample size / budget;
- **noise floor** — the 95% interval half-width of the estimate over independent
  re-samples at a fixed configuration. A difference below the floor is not a
  result; with the floor printed, a later "this metric doesn't travel" cannot be
  confused with "we estimated it badly".

## Coordination — conditional mutual information `I(a1; a2 | s)`

High when two agents coordinate within a context `s`, zero when independent. The
phantom (`coupled_source.py`) is a coupled categorical source with tunable
coupling `kappa`, so `I(a1; a2 | s)` is known exactly. The metric is conditional,
so the estimator stratifies by `s` and averages — and conditioning splits the
sample, which is where bias bites.

**Result:** at `N = 10,000` the Miller-Madow estimator recovers the known value to
**~0.001 bits** across the range; it roughly halves the plug-in's small-N
over-coordination bias (0.071 vs 0.152 bits at `N=100`); the **noise floor is
0.025 bits at `N=10,000`** (0.29 at `N=100`).
Run: `uv run python phase0/calibrate.py` · figure: `results/calibration.png`.

## Fragility / antifragility — response curvature

The fragility family (CAFE and kin) reduces to the curvature of a stress-response
`R(sigma)`: `< 0` is fragile (concave), `> 0` antifragile (convex). The phantom
(`fragility_source.py`) has a known curvature, observed as noisy replicate
rollouts; the estimator (`curvature_estimators.py`) fits the curve. Curvature is a
weak second-order feature, so the **floor sets the budget at which the sign is
callable**.

**Result:** near-unbiased (bias ~0.003 at 900 rollouts), sign recovered across the
range. The **floor falls from 0.69 (budget 9) to 0.066 (budget 900)**, so a
`|kappa| = 1` fragility resolves by ~budget 27, while `|kappa| = 0.5` needs
~budget 270+.
Run: `uv run python phase0/calibrate_fragility.py` · figure:
`results/fragility_calibration.png`.

## Misalignment-propagation — contagion coefficient

Plant a misalignment dose in a seed agent; the metric is the coefficient `beta` by
which it raises the rest of the collective's misalignment (`0` = contained, `1` =
full propagation — the *AI Organizations* mechanism, made controllable). The
phantom (`propagation_source.py`) has a known `beta`; the estimator
(`propagation_estimators.py`) is the dose-response slope. The **floor sets the
detection threshold** — when faint contagion is resolvable from none.

**Result:** near-unbiased (bias ~0.001), coefficient recovered across the range.
The **floor falls from 0.185 (budget 100) to 0.018 (budget 10,000)**, so a faint
contagion (`beta = 0.2`) is resolvable from zero by ~budget 300–1,000 — the
dose-sweep budget needed to claim a planted misalignment spread.
Run: `uv run python phase0/calibrate_propagation.py` · figure:
`results/propagation_calibration.png`.

## Scope

These calibrate metric *estimators* on known answers — they establish the floor
under which a later "this metric does / does not travel across LLM-agent setups"
result is interpretable rather than an artifact of estimation. They do not, on
their own, evidence cross-setup invariance. The **next phase** runs these
calibrated estimators on small LLM-agent teams — the invariance sweep — with each
metric's floor, established here, printed under the result.

## Run

```bash
uv run python phase0/calibrate.py             # coordination
uv run python phase0/calibrate_fragility.py   # fragility
uv run python phase0/calibrate_propagation.py # misalignment-propagation
# tests:
uv run python phase0/test_phase0.py
uv run python phase0/test_fragility.py
uv run python phase0/test_propagation.py
```

## Files

Per metric: a `*_source.py` phantom (closed-form ground truth + sampler), an
estimators module, a `calibrate_*.py` sweep, and a `test_*.py` known-answer suite.

- coordination: `coupled_source.py` · `mi_estimators.py` · `calibrate.py` · `test_phase0.py`
- fragility: `fragility_source.py` · `curvature_estimators.py` · `calibrate_fragility.py` · `test_fragility.py`
- propagation: `propagation_source.py` · `propagation_estimators.py` · `calibrate_propagation.py` · `test_propagation.py`
- `results/` — calibration figures and CSVs.
