# Emmy

Evaluation-invariant measurement for multi-agent AI systems.

**Status:** Pre-experiment. The
[program overview](docs/program-overview.md) is the canonical reference.
Concrete work so far: the Phase 0 estimator calibrations in
[`phase0/`](phase0/), and a smoke-test of the invariance idea in
[`demo/`](demo/).

## The gap

One question runs under everything here: **do numbers about collectives mean
anything beyond the setup that produced them?**

AI agents increasingly work in groups — language-model agents that call
tools and coordinate with one another, reinforcement-learning agents
acting in a shared environment. These collectives are already moving
into real-world deployment.

We have little settled practice for measuring what such a group does
collectively. The closest tools look elsewhere: single-model methods —
evaluation, interpretability, AI control — inspect one model at a time,
and reveal little about the behavior of the group. The field is starting
to publish collective metrics, and some test robustness across models or environments. But a measure of a
genuine group property should read the same when the setup changes in ways that
shouldn't matter, and shift only when the property itself does — and that
invariance is not yet treated as something a measure must establish, nor are the
measures calibrated against known ground truth. So whether a published number
reflects the group or the setup it ran in stays unclear, and such numbers rarely
carry from one paper to the next, or from a lab setup to a deployment. And the reporting conventions that downstream safety and
evaluation work will inherit are taking root now — before the measurement
practice supporting them is sound.

## The approach

Emmy is a research program building the measurement foundations for
groups of AI agents. Its approach is atypical: instead of starting from an
anthropomorphized characterization of what the agents are doing —
cooperating, competing, deceiving — and building a proxy metric around it,
emmy starts from quantities measurable from behavior and asks what they
reveal about safety and alignment.

The field is now publishing collective-behavior metrics for LLM-agent
systems — fragility/antifragility, misalignment propensity, multi-agent
evaluation suites, interaction-graph measures. Emmy takes that battery of
published metrics and characterizes, for each, two things: how much it
**travels** across setup changes that should not move a genuine collective
property (invariance), and whether it tracks a collective property that
joint task-performance cannot separate (construct validity). Underneath, these
are one question: does the metric track something that belongs to the group
itself, rather than the particular setup it was run in? Where a metric
holds on both, two payoffs follow: claims about coordination, robustness,
and failure become comparable across papers, and an external evaluator
gains a way to inspect a deployed group directly — the inspection layer
single-model methods aren't equipped to provide.

The measurement depends only on agents' actions and observations, so it
needs no privileged access to the underlying models — the surface a
third-party evaluator actually has. In multi-principal deployments — agents
from different vendors or operators interacting in one arena — that surface
is not a preference but the terrain: no single party holds privileged access
to every agent, so the behavioral record is the only common evidence an
auditor can work from. And every metric is calibrated first
against synthetic systems with known ground truth — a practice adapted from
information dynamics — so a printed
estimator-noise floor makes each result interpretable rather than an
artifact of estimation. The controlled setting is doing different work here
than in most of the field: controlled environments usually *characterize*
systems — where control is the limitation; emmy uses them to *calibrate*
instruments — where control is the point, because calibration needs known
ground truth. Invariance is then the licensed exit from the lab — what must
hold before a lab number can say anything about a deployment.

This is pre-experiment work; the claims are not yet validated. It measures
behavior rather than internal cognition, so what a positive identifies is a
behavioral disposition — a property of the group, not its latent intent (a
principled limit of behavioral evaluation, and one that weakens further against
agents optimizing to fool the measurement). It builds on metrics the field has
already published, and is complementary to benchmark evaluation and
interpretability.

## Phase 0 — estimator calibration

Before any of these metrics is run on real LLM-agent rollouts, its estimator
is calibrated against a synthetic source whose value is known in closed
form — measuring estimator bias and the sampling noise floor directly, at
~zero compute. [`phase0/`](phase0/) calibrates all three battery metrics:

- **coordination** (conditional mutual information) — recovers the known
  value to ~0.001 bits at N=10,000, with the noise floor printed (0.025 bits);
- **fragility / antifragility** (response curvature) — recovers the sign and
  magnitude; the floor sets the budget at which fragile-vs-antifragile is callable;
- **misalignment-propagation** (contagion) — recovers the coefficient; the
  floor sets the budget at which faint contagion is detectable from zero.

The coordination metric is also calibrated for **specificity** — that it does
*not* fire on a shared-cause look-alike. On a two-knob synthetic phantom, a pure
common cause (two agents driven by a shared context, with no link between them)
drives plain mutual information to 0.64 bits while the conditional metric stays
at zero within its noise floor; a genuine coupling still registers. This is the
construct-validity check above made concrete on known ground truth — and the
case that matters for real agents, where a shared base model or system prompt
makes two agents behave alike with no influence passing between them.

![Plain MI is fooled by a shared cause while conditional MI stays at zero](phase0/results/confound_heatmap.png)

The conditioning rule itself is calibrated from both sides: a companion
negative-control phantom (the *collider* case) demonstrates the opposite
failure — two agents with zero coupling jointly produce an environment state,
and conditioning on that state *manufactures* up to a full bit of apparent
coordination, while plain MI correctly reads zero. The rule this pair pins
down: condition on the shared cause, never on the shared effect — and for
deployed agents the shared effect is exactly the jointly-produced environment
state an audit is most tempted to stratify by.

Each result is the estimator's resolution limit — the floor under which a
later "this metric does or does not travel across setups" finding is
interpretable rather than an estimation artifact. The next phase runs these
calibrated estimators on small LLM-agent teams (the invariance sweep). See
[`phase0/README.md`](phase0/README.md).

## Demo — invariance under reward rescaling

[`demo/`](demo/) is a small, runnable smoke-test of the measurement
machinery on the simplest, provable case — reward rescaling, where
behavioral invariance follows from policy-invariance: two tabular Q-learning
agents in the iterated prisoner's dilemma, showing that behavioral
observables (coordination, action autocorrelation) are invariant under
reward rescaling while reward-based quantities are not. It illustrates the
invariance question on the one corner where the answer is provable — a
smoke-test of the machinery, not a research result. See
[`demo/README.md`](demo/README.md).

## Emmy Noether

After [Emmy Noether](https://en.wikipedia.org/wiki/Emmy_Noether) (1882–1935),
whose foundational work connecting symmetries to invariants underlies the
framing of evaluation-invariant measurement.

## License

Apache 2.0 — see `LICENSE`.
