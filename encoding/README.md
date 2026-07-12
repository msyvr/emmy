# Encoding layer — text and actions → the symbols the estimators consume

Implements §1 of the pre-sweep design spec: the first act of measurement, where
each agent's *emitted* stream (and only that — never its received context)
becomes a categorical series the phase0-calibrated estimators can consume.

**Status: built and tested; Gate E not yet passed.** What exists here covers
Gate E checks 1–2 (per-stream isolation and determinism, as executable tests)
and the tooling for check 3 (the `entropy_bits` utilization measure). Checks
4–5 and 7 (construct sensitivity, nuisance stability, per-setup probe
calibration) require pilot episodes and are still open — *built is not
validated*, and none of these encoders is licensed for a swept cell until
Gate E passes.

## The battery

| Encoder | Role | K | Fitting |
|---|---|---|---|
| `action_type` | primary, structured actions | 8 | none (declared taxonomy) |
| `speech_act` | primary, NL messages | 6 | none (pinned single-stream judge, injected) |
| `codebook` | sensitivity appendix | k+1 | fit **once** on the pooled calibration set, then frozen (Decision D-CB) |
| `surface` | discriminant control | 8 | none (declared bins) |

Contracts, enforced in code: one symbol per event; per-stream by construction
(the E1 test replaces the other agent's stream wholesale and asserts
bit-identical output); encoder identity = `config_id(name, version, params)` —
any change is a different instrument; the codebook refuses to refit and
refuses to encode before fitting.

Wiring pending (deliberately not stubbed with defaults): a real pinned judge
model for `speech_act` (temperature 0; its `judge_id` enters the config) and a
fixed open-weight embedder for `codebook` — which, per the spec, must not be
one of the backbones under test.

Run tests: `uv run python encoding/test_encoding.py`
