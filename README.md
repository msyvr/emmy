# Emmy

Evaluation-invariant measurement for multi-agent AI systems.

**Status:** Pre-experiment. The
[program overview](docs/program-overview.md) is the current canonical
reference; additional documents will be added to `docs/` as they are
finalized for public release. A small runnable demonstration of the core
idea is in [`demo/`](demo/).

## The gap

AI agents increasingly work in groups — language-model agents that call
tools and coordinate with one another, reinforcement-learning agents
acting in a shared environment. These collectives are already moving
into real-world deployment.

We have little settled practice for measuring what such a group does
collectively. The closest tools look elsewhere: single-model methods —
evaluation, interpretability, AI control — inspect one model at a time,
and reveal little about the behavior of the group. Multi-agent results,
where reported, are expressed in terms tied to one setup — a score that
depends on a particular environment's payoffs, a behavioral signal
defined for a single experiment. Those numbers rarely carry from one
paper to the next, or from a lab setup to a deployment. And the reporting
conventions that downstream safety and evaluation work will inherit are
taking root now — before the measurement practice supporting them is
sound.

## The bridge

Emmy is a research program building the measurement foundations for
groups of AI agents. Its approach is atypical: instead of starting from
an anthropomorphized characterization of what the agents are doing —
cooperating, competing, deceiving — and building a proxy metric around
those behaviors, Emmy starts from measurable quantities and asks what
those reveal about safety and alignment.

In its simplest form, the hypothesis is that the right unit of
measurement is the _observable_ — a quantity computed from what the
agents do, their actions and observations, defined so that it does not
change when you rescale rewards or relabel a setup. A standard
set of such observables yields two useful outcomes: 1. Claims about
coordination, robustness, and failure become comparable across papers. 2.
An external evaluator gains a way to inspect a deployed group of agents
directly — the inspection layer that single-model methods aren't equipped
to provide.

Because the measurement depends only on behavior, it requires no
privileged access to the underlying models, and the same instruments
apply whether the agents are reinforcement-learning policies,
language-model ensembles, or active-inference systems.

The first deliverable is a paper and an open-source library: a small,
canonical set of these observables for cooperative multi-agent
collectives, with pre-registered, falsifiable claims and a published
null-result protocol.

A couple of notes: First, this is pre-experiment work — these claims are
not yet validated. Second, the framework measures behavior, not internal
cognition — making it complementary to benchmark evaluation and
interpretability.

## Demo: invariance under reward rescaling

[`demo/`](demo/) contains a small, runnable smoke-test of the measurement machinery on the simplest, provable case — reward rescaling, where behavioral invariance follows from policy-invariance:
two tabular Q-learning agents in the iterated prisoner's dilemma, showing
that behavioral observables (coordination, action autocorrelation) are
invariant under reward rescaling while reward-based quantities are not.
It is a smoke-test of the measurement machinery, not a research result.
See [`demo/README.md`](demo/README.md).

## Emmy Noether

After [Emmy Noether](https://en.wikipedia.org/wiki/Emmy_Noether) (1882–1935),
whose foundational work connecting symmetries to invariants underlies the
framing of evaluation-invariant measurement.

## License

Apache 2.0 — see `LICENSE`.
