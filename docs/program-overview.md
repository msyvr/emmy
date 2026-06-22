# Measurement Foundations for Multi-Agent AI: Program Overview

**Status:** Current synthesis, 2026-06-08. Pre-experiment. This document is the
canonical public reference for the program; other documents will be added to `docs/`.

**Summary.** Emmy builds measurement foundations for multi-agent AI. The field is now
publishing collective-behavior metrics for LLM-agent systems — fragility/antifragility,
misalignment propensity, multi-agent evaluation suites, interaction-graph measures — but
each is computed on a single setup, and none tests whether the metric survives a change
of setup. Emmy takes that battery of published metrics and characterizes, for each, two
things jointly: how much it travels across setup changes that should not move a genuine
collective property (**invariance**), and whether it tracks a collective property that
joint task-performance cannot separate (**construct validity**). Measurement is from
action–observation streams, without privileged model access — the surface a third-party
auditor actually has. Every metric is calibrated first against analytic synthetic
collectives with known ground truth, so a printed estimator-noise floor makes each result
interpretable rather than an artifact of estimation. The work produces a paper and an
open-source library — an open, reproducible map placing each metric in the invariant ×
construct-valid plane. This is pre-experiment work, building on metrics the field has
already published.

---

## 1. Motivation

Multi-agent AI systems — large-language-model agent ensembles acting on a shared task or
environment — are moving from research into deployment faster than their measurement
practices are maturing. The tools built for the prior era characterize one model at a
time: single-model evaluation and interpretability. The field is now publishing
collective-level metrics — fragility/antifragility of multi-agent LLM systems,
misalignment propensity for agentic systems, multi-agent evaluation suites,
interaction-graph measures — but each is reported on a single setup, and none tests
whether the metric survives a change of setup. Even applied practice shows the same shape:
Anthropic's engineering guide to evaluating AI agents is organized entirely around
_single_ agents — coding, conversational, research, computer-use — and names multi-agent
collaboration as a frontier its techniques will still need to adapt to
([Anthropic Engineering, 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

Four consequences:

- **Cross-paper claims do not compose.** A robustness or coordination finding reported in
  one evaluation setup does not transfer, because the collective metrics now being reported
  are not known to travel beyond the setup that produced them.
- **Evaluation-setup dependence is implicit.** Every claim is conditional on backbone
  model, decoding seed, prompt phrasing, team size, communication topology, and task
  instance — but these dependencies are not characterized. Reproducibility is
  point-estimate-within-setup, not class-of-setup.
- **Safety-relevant questions are unreachable.** Whether a controlled experiment maps to
  deployment, whether a collective is fragile under stress, whether injected misalignment
  in one agent propagates — none answerable cleanly without measurement that travels across
  evaluation setups.
- **The clock is capability, not convention.** Reporting conventions are forming now, but
  the binding deadline is the trajectory: the field is racing toward autonomous AI R&D,
  carried out by agent collectives — "synthetic teams" of AI systems managing one another.
  Anthropic co-founder Jack Clark, in his _Import AI_ newsletter, puts ~60% on AI systems
  autonomously running their own R&D by the end of 2028
  ([Import AI 455](https://importai.substack.com/p/import-ai-455-automating-ai-research)).
  The inspection layer for collective behavior has to exist before collectives do
  high-stakes autonomous work — and once humans leave the loop, external measurement is the
  oversight handle that remains.

The gap is in the measurement layer, and closing it starts with finding out which of the
field's collective metrics travel across setups — and which are setup artifacts.

## 2. Thesis

The program's first project is empirical and diagnostic. It does not assume a canonical
set of evaluation-invariant metrics exists; it takes the metrics the field is already
publishing and asks which of them earn cross-setup trust — which is to say, which track a
property of the collective itself rather than of the setup it was measured in. For each metric
in the battery, it characterizes two properties jointly:

- **Invariance** — how much the metric moves under a change of setup. Some changes should
  not move a genuine collective property (decoding stochasticity, prompt paraphrase,
  task-instance resampling); a metric that moves under these above its noise floor is not a
  cross-setup instrument. Other changes (backbone model, team size, communication topology)
  may legitimately matter — there the metric's _profile_ across the axis is the result, and
  a warning to anyone comparing the metric across deployments that differ on it.
- **Construct validity** — whether the metric tracks a collective property that is
  deliberately planted (so the answer is known) and that joint task-performance cannot
  separate. A metric can be perfectly stable and still measure nothing; construct validity
  is what tells invariant-and-meaningful apart from invariant junk.

The result is the map: each metric placed in the **invariant × construct-valid** plane,
with a calibration phantom and a printed estimator-noise floor making every placement
interpretable. Whether any of the field's metrics land in the invariant-and-valid cell is
the question the study answers, not a premise it assumes; a finding that few or none do —
that the field's collective metrics are setup-bound, and by how much — is itself a result
worth publishing and a direct warning to anyone comparing collective metrics across
deployments.

This is the **setup-invariance** axis — whether a metric travels across deployments. A
companion concern runs on the **time** axis: behavior drifts as a system keeps learning, so
evaluation can no longer assume a frozen artefact (Pacchiardi, Hernández-Orallo et al.,
_Continual Learning Requires Evaluating Trajectories_, 2026). The two are composable — a
collective of continually-learning agents breaks both.

**The stance is measurement-first.** Most multi-agent and safety work starts from a
human-meaningful concept — cooperation, deception, empowerment, power-seeking — and looks
for a proxy that captures it in some environment. This program inverts that: it starts from
metrics that are operationally defined and measurable from behavior, and treats both their
invariance structure and their relationship to safety as empirical questions to be
characterized, not settled by naming. The move is the discipline evolutionary biology
brought to traits once assumed adaptive (Gould & Lewontin's "spandrels," 1979): build the
controlled alternative and measure, rather than read function into what you observe. The
approach's payoff depends on the bridge from measured metrics to safety-relevant behavior
being real, which the program treats as something to demonstrate (see §8), not assert.

**The scope of the claim.** What a positive result identifies is a behavioral _disposition_ —
the group's stable input→behavior structure — not its latent intent. This is a principled limit
of behavioral evaluation (Santos-Grueiro, _Normative Indistinguishability under Behavioral
Evaluation_, 2026): finite behavioral evidence pins down an equivalence class of collectives
consistent with the observed streams, not a unique internal state — which is exactly why the
invariance class has to be specified, and why the metrics are reported as profiles, not verdicts.
The bound tightens further when agents optimize against the measurement; that adversarial regime
is where internals-based methods take over, and is out of scope here.

The approach rests on a few commitments.

1. **Build on the field's published metrics.** The battery is the metrics others have
   published; emmy's contribution is the comparability layer over them. The result is a
   metric an auditor can already find in the literature, now carrying a cross-setup profile.
2. **Invariance demanded on nuisances, measured on setup axes.** Each metric is reported
   with the transformations under which it stays put and the axes along which it varies. The
   split is what keeps the invariance claim non-vacuous: invariance is _demanded_ only on
   unambiguous nuisances and _measured_ on the ambiguous setup axes.
3. **Construct validity against planted ground truth.** Validity is scored against a
   property emmy plants and therefore knows — not against the nuisance axis itself, which is
   what keeps the validity test from collapsing into the invariance test.
4. **Calibration before spend.** Every metric's estimator is run first on analytic
   synthetic collectives with known values, establishing a trueness bound and an
   estimator-noise floor before any LLM-agent inference. A non-invariance result is
   meaningless without a floor under it.
5. **Setup reflexivity.** Every claim reports the setup class under which it was produced
   _and_ the negative space it admits but does not measure. Reproducibility means
   setup-class reproducibility.
6. **No privileged access.** Metrics are computed from action–observation streams — no
   weights, activations, or training data — matching the interface a third-party auditor or
   regulator actually works under.

## 3. Methodological discipline

The discipline matched to a discovery study is not pre-registration — that is a tool for
_confirmatory_ hypothesis-testing, and forcing it onto exploratory characterization buys
rigidity and perverse design incentives (a test no one wants to fail gets designed to be
easy to pass) more than rigor. The discipline here is **systematic coverage, complete
reporting, and reproducibility**:

- **Declared scope, swept completely.** The battery, the transformation set, and the
  LLM-agent setups are fixed and stated up front, then swept as a full grid. The guard
  against cherry-picking is that the _whole_ grid is reported — including metrics that
  neither travel nor measure their construct — not a pre-committed go/no-go on a favored
  hypothesis.
- **Profiles, not verdicts.** Invariance and construct validity are reported as continuous
  quantities with re-sample bands, not binary "is/isn't" labels. Validity is not binary;
  "how invariant, under which transformations, within what bounds" is the right unit.
- **Reproducibility as the primary contribution.** Code, configs, agent setups, and the
  measured-metric library are released so the community can re-analyze, re-run, and build on
  the map — the open-science investment that compounds far more than a pre-registration does.
- **Negative results are inherent, not bolted on.** Because the result is the map, a metric
  that fails to travel or to track its construct is a reported data point, not a "failed"
  experiment — no pre-registration is needed to make null findings publishable.

This is not a purely academic concern. Production eval practice hits the same wall from the
other side: shared state across trials — leftover files, cached data, resource contention —
produces _correlated_ failures that look independent, and the numbers become unreliable for
measuring the system ([Anthropic Engineering, 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
Harness hygiene handles that case; the metrics here go one step further — when the
correlation comes from the agents themselves sharing structure, it is treated as a quantity
to measure, not only a confound to isolate.

Pre-registration is held in reserve for a later, _specific_ confirmatory claim — if and
when the program makes one (e.g. "metric M is invariant under transformation T within bound
ε"). That is the regime it was built for; this first, map-making phase is not.

## 4. First project: paper 1 + library v0.1

**Working title:** _Do collective-behavior metrics for LLM-agent teams travel? An
invariance and construct-validity characterization of the field's emerging measures._

The paper places each metric in the battery in the invariant × construct-valid plane, every
cell reported as a measured profile rather than asserted:

| | **construct-valid** (tracks planted ground truth) | **not construct-valid** |
|---|---|---|
| **invariant** (flat across core nuisances, above floor) | **usable cross-setup instrument** — travels _and_ means something | **invariant junk** — stable but measures a setup-independent nuisance |
| **not invariant** | **real but incomparable** — tracks the construct on its home setup; a warning to anyone comparing across deployments | **discard** |

**The battery — which published metrics are under test.** A metric enters the battery iff
it (a) claims to measure a _collective-level_ property, not a single-agent property
aggregated; (b) is computable from action–observation streams with no privileged model
access; and (c) has a definition precise enough to reimplement. Candidates from a recent
literature scan, final selection pending a reimplementability pass:

- **Fragility / antifragility of multi-agent LLM systems** (the anchor — a published
  collective metric with no cross-setup test).
- **Misalignment propensity for agentic systems.**
- **A multi-agent-system evaluation suite.**
- **An interaction-graph / coordination measure** — a coordination or influence scalar
  built from the agent–agent interaction graph (e.g. lagged action correlation or
  transfer-entropy-weighted edges; algebraic connectivity). Emmy defines a clean reference
  version where no single published one dominates.
- **A misalignment-propagation / contagion measure** — how a planted misalignment in one
  agent shows up in the collective's behavior (motivated by Anthropic's _AI Organizations_
  finding that collectives can be less aligned than their members).

The battery stays at 3–4 metrics actually reimplemented and run: a deep, calibrated
characterization of four says more than a shallow sweep over ten. The reimplementability
pass doubles as a citation-verification pass.

**Optional battery member — a dynamic response observable.** A time-to-recovery of a
collective order parameter after a defined perturbation pulse — a dynamic susceptibility the
field's static metrics lack. It is carried as another candidate on the same footing, tested
for invariance and construct validity like the rest, and is the first thing dropped if the
perturbation protocol on LLM agents proves expensive or noisy. The paper stands without it.

**The transformation set — what "change of setup" means.** Three classes, and the split is
what makes the invariance claim non-vacuous:

- **(A) Core nuisances — invariance demanded.** Decoding stochasticity (temperature /
  sampling seed); prompt / scaffold paraphrase (same task and roles, reworded); task-instance
  resampling (same construct, different surface instances). A metric that moves here above
  the floor is not a cross-setup instrument.
- **(B) Setup axes — invariance measured, not demanded.** Backbone model swap (within and
  across capability tiers); team size N; communication topology / role assignment. A metric
  may legitimately depend on these; emmy reports the _profile_ per metric — the
  apples-to-oranges map — rather than a pass/fail.
- **(C) Construct axis — variation expected, ground-truth-controlled.** Properties emmy
  plants so it knows the answer: incentive structure (cooperative ↔ mixed ↔ adversarial),
  injected coordination, injected single-agent misalignment. A valid metric must move _with_
  the planted construct.

The target shape, per metric: flat across (A), profiled across (B), tracking across (C).

**Substrate — small LLM-agent teams.** This is where the field's metrics live and where
emmy's target users (auditors, operators) work, so invariance is characterized here
directly. Team size N = 3–5; open-weight backbones first (one lead family plus a second for
the cross-backbone axis) plus one frontier/API model for a capability-tier transfer check;
short, multi-step collaborative tasks with a clear collective-behavior surface, on existing
harnesses where possible.

**The calibration ladder.** Analytic phantom (known ground truth, near-zero compute) →
[optional] semi-controllable MARL bridge (coupling and N directly settable; thousands of
cheap re-rollouts validate the floor machinery before it is trusted on pricier LLM-agent
runs) → LLM-agent teams. Trueness established cheaply, then carried up.

**Library v0.1.** Open-source Python library computing the battery metrics from
action–observation streams. Audit-friendly by construction: no privileged access to
weights, training data, or internal representations required. Initial scope is
metrics-and-API; reference setups and analysis examples follow the pilot.

**How the work is structured.** The calibration phantom comes first — estimators, noise
floor, and construct-validity calibration on synthetic data whose answers are known. The
LLM-agent study follows: a single cell with the floor printed, then the invariance sweep
across nuisances and setup axes, then construct validity on planted-ground-truth teams. The
library and data are released alongside. Each step yields a result that stands without the
ones after it.

## 5. Later directions

Beyond the first project, in roughly increasing order of ambition:

- **The dynamic observable as a contribution.** If the optional response/recovery observable
  earns its place in paper 1, a follow-on develops it as emmy's own metric — a dynamic
  susceptibility for collectives — characterized for invariance and validity in its own
  right.
- **Phenomenology.** A second paper applying the invariant-and-valid metrics — likely a
  characterization of intermittent adversarial injection, with response curves as the
  primary findings.
- **Scaling and structure.** How the surviving metrics scale with N, communication
  bandwidth, and backbone capability; where in setup space collective behaviors change
  qualitatively.
- **Comparative and applied.** Larger LLM-agent organizations; the _AI Organizations_
  alignment-gap finding tested at scale; diversity and collective robustness; real-world
  mapping conditions.

The first project stands on its own; the directions above extend it, and nothing here
depends on them.

## 6. Open questions before the first runs

A few choices are settled before the LLM-agent study begins: the final 3–4 metrics — fixed
by a reimplementability pass that also states a clean reference version for any
underspecified published metric, and that verifies every citation before it is printed; the
backbone families and the task harness; and which planted manipulations produce a collective
behavior that joint task-performance cannot already separate. A solo program at this ambition
carries real risk; a collaborator with LLM-agent-evaluation experience would help both the
technical work and its adoption.

## 7. Governance and standards relevance

Audit-friendly architecture is a founding design constraint. The library computes metrics
from action–observation streams with no privileged-access requirement; this matches the
interface conditions under which third-party auditors and regulators must work.

Where the metrics touch policy:

- _Per-metric invariance profiles → comparability standards._ Which collective metrics a
  regulator can compare across two deployments, and which it cannot.
- _Setup-class declarations → transparency standards._
- _No-privileged-access metrics → architecture-agnostic regulation._
- _Negative-space reporting → standards humility._

Relevant standards efforts: AISI-UK methodology team, NIST CAISI, CEN-CENELEC JTC 21,
Cooperative AI Foundation, OECD AI standards working groups.

## 8. AI safety relevance

The program's safety relevance is downstream option value — a precondition for other safety
work that holds whatever the risk picture turns out to be.

**The inspection gap (the headline).** Powerful models from many developers are increasingly
composed into _collectives_ by agent frameworks: multiple model-agents acting on a shared
task or environment. AI-control and single-model evaluation assess the trustworthiness of
_individual_ models; they do not characterize the collective's emergent behavior — the group-level behavior its individual-model checks don't predict — and no
external inspection layer for it exists. That surface is already a demonstrated concern: Anthropic's alignment team finds that
multi-agent _AI organizations_
can be more effective but _less aligned_ than the individual agents composing them, and
argues such systems need testing for misalignment across organizational structures — the
measurement no single-model check provides
([Anthropic, 2026](https://alignment.anthropic.com/2026/ai-organizations/)). The Cooperative AI Foundation's _Multi-Agent Risks from Advanced AI_ ([Hammond et al., 2025](https://arxiv.org/abs/2502.14143)) catalogues the failure modes — miscoordination, conflict, collusion — that such an inspection layer would need to measure.

**Why a comparability instrument is the precondition.** Today a collective-(mis)alignment
metric reported by a vendor on its own setup cannot be trusted by an external auditor on a
_different_ deployment — there is no evidence the number travels. Emmy's map says, per metric,
whether it travels, and on the no-privileged-access surface a third party actually has. That
is the external-inspection layer AI-control (white-box, owner-side) does not provide, and the
precondition for ever comparing one collective to another. AI-control evaluates the trust relationship between agents — including where its protocols
involve multiple agents, e.g. untrusted monitors. The collective dynamics those protocols run
inside are what it does not characterize, and what emmy measures.

**Falsifiable, composable claims.** Robustness, alignment, and emergent-failure-mode claims
about multi-agent systems are uninterpretable without measurement that travels across testing
setups; with it, they become comparable across labs and falsifiable.

**Early warning — a hypothesis held in reserve.** Whether multi-agent collapses carry the
slow-precursor signatures (rising autocorrelation, variance, recovery time) that precede
regime shifts in climate, ecology, and finance is an open empirical question — some collective
failures may be discrete or adversarial, with no slow precursor. It is a complement to the
headline for catastrophic-risk-focused audiences, pursued only if a planted construct sweep
produces a qualitative transition, and not part of the first paper.

## 9. What a result looks like

The phantom calibration comes first: every metric's estimator checked against known ground
truth, with a trueness bound and a noise floor under it, at near-zero compute. On that
footing the LLM-agent study produces the map — which of the field's metrics travel across
the core nuisances, which only profile across setup axes, and which track a planted construct
that joint task-performance cannot separate, including the ones that do neither. A useful
outcome is at least one published metric an auditor can carry across setups. A quantified
null is equally citable: the field's collective metrics are setup-bound, and here is how far.
Either way the whole grid is reported — profiles rather than verdicts — with code and data
released so the map is reproducible.

## 10. Document map

This is currently the only public document in `docs/`. Additional documentation will be
added as it is finalized for public release.
