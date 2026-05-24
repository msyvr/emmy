# Emmy

Evaluation-invariant measurement for multi-agent AI systems.

**Status:** Pre-experiment. The
[program overview](docs/program-overview.md) is the current canonical
reference; additional documents will be added to `docs/` as they are
finalized for public release. A small runnable demonstration of the core
idea is in [`demo/`](demo/).

## The gap

Multi-agent AI systems — reinforcement-learning agent collectives,
large-language-model agent ensembles, and emerging active-inference
multi-agent setups — are moving toward deployment faster than their
measurement practices are maturing. Empirical claims about coordination,
robustness, and emergent failure modes are reported in apparatus-specific
quantities (joint return, ad-hoc behavioral indicators) that do not compose
across papers and do not transfer across deployment conditions. The
reporting conventions that downstream safety and evaluation work will
inherit are crystallizing now.

## What this is

A research program establishing measurement foundations for multi-agent
AI: evaluation-invariant observables, perturbation–response analysis of
collective fragility, and lengthening recovery times as early warning
of catastrophic transitions. Methodological orientation drawn from
physics, systems biology, and statistical mechanics; first-principles
measurement theory designed to complement, not replace, benchmark-based
evaluation.

The first deliverable is a paper plus open-source library establishing
a small canonical set of evaluation-invariant observables for cooperative
multi-agent collectives, with pre-registered falsifiable claims
(discrimination, invariance, transfer, system-validation) and a published
null-result protocol.

## Demo

[`demo/`](demo/) contains a small, runnable instance of the central claim:
two tabular Q-learning agents in the iterated prisoner's dilemma, showing
that behavioral observables (coordination, action autocorrelation) are
invariant under reward rescaling while reward-based quantities are not.
It is a smoke-test of the measurement machinery, not a research result.
See [`demo/README.md`](demo/README.md).

## Name

After [Emmy Noether](https://en.wikipedia.org/wiki/Emmy_Noether) (1882–1935),
whose foundational work connecting symmetries to invariants underlies the
framing of evaluation-invariant measurement.

## License

Apache 2.0 — see `LICENSE`.

## Contact

Issues and discussion via this repository. Direct correspondence:
[email].
