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
interpretable rather than an artifact of estimation. The first deliverable is a paper plus
open-source library: an open, reproducible map placing each metric in the
invariant × construct-valid plane. This is pre-experiment work; it builds on the field's
metrics rather than competing with a new one.

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

The gap is foundational, and closing it starts with finding out which of the field's
collective metrics travel across setups — and which are setup artifacts.

## 2. Thesis

The program's first project is empirical and diagnostic. It does not assume a canonical
set of evaluation-invariant metrics exists; it takes the metrics the field is already
publishing and asks which of them earn cross-setup trust. For each metric in the battery,
it characterizes two properties jointly:

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

The deliverable is the resulting map: each metric placed in the **invariant ×
construct-valid** plane, with a calibration phantom and a printed estimator-noise floor
making every placement interpretable. Whether any of the field's metrics land in the
invariant-and-valid cell is the question the study answers, not a premise it assumes; a
finding that few or none do — that the field's collective metrics are setup-bound, and by
how much — is itself a result worth publishing and a direct warning to anyone comparing
collective metrics across deployments.

**The stance is measurement-first.** Most multi-agent and safety work starts from a
human-meaningful concept — cooperation, deception, empowerment, power-seeking — and looks
for a proxy that captures it in some environment. This program inverts that: it starts from
metrics that are operationally defined and measurable from behavior, and treats both their
invariance structure and their relationship to safety as empirical questions to be
characterized, not settled by naming. The move is the discipline evolutionary biology
brought to traits once assumed adaptive (Gould & Lewontin's "spandrels," 1979): build the
controlled alternative and measure, rather than read function into what you observe. The
approach's payoff depends on the bridge from measured metrics to safety-relevant behavior
being real, which the program treats as something to demonstrate (see §9), not assert.

The approach rests on a few commitments.

1. **Build on the field, don't compete.** The battery is the metrics others have
   published; emmy's contribution is the comparability layer over them, not a rival
   observable. This is both more defensible for a newcomer and more useful to an auditor — a
   metric already in the literature, now with a cross-setup profile attached.
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
- **Negative results are inherent, not bolted on.** Because the deliverable is the map, a
  metric that fails to travel or to track its construct is a reported data point, not a
  "failed" experiment — no pre-registration is needed to make null findings publishable.

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

The paper places each metric in the battery in the invariant × construct-valid plane,
every cell reported as a measured profile rather than asserted:

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

Discipline: keep the battery to **3–4 metrics actually reimplemented and run**, not a long
list name-checked. A deep, calibrated characterization of four beats a shallow sweep over
ten. The reimplementability pass doubles as a citation-verification pass.

**Optional battery member — a dynamic response observable (descopable).** A
time-to-recovery of a collective order parameter after a defined perturbation pulse — a
dynamic susceptibility the field's static metrics lack. It is carried _only_ as another
candidate on the same footing, tested for invariance and construct validity like the rest,
and is the first thing descoped if the perturbation protocol on LLM agents proves expensive
or noisy. The paper stands without it.

**The transformation set — what "change of setup" means.** Three classes, and the split is
what makes the invariance claim non-vacuous:

- **(A) Core nuisances — invariance demanded.** Decoding stochasticity (temperature /
  sampling seed); prompt / scaffold paraphrase (same task and roles, reworded); task-instance
  resampling (same construct, different surface instances). A metric that moves here above
  the floor is not a cross-setup instrument, full stop.
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
harnesses where possible. Building the substrate is not the contribution.

**The calibration ladder.** Analytic phantom (known ground truth, ~$0) → [optional]
semi-controllable MARL bridge (coupling and N directly settable; thousands of cheap
re-rollouts validate the floor machinery before it is trusted on pricier LLM-agent runs) →
LLM-agent teams (the target). Trueness established cheaply, then carried up.

**Library v0.1.** Open-source Python library computing the battery metrics from
action–observation streams. Audit-friendly by construction: no privileged access to
weights, training data, or internal representations required. Initial scope is
metrics-and-API; reference setups and analysis examples populate post-pilot.

**Plan.** Stage-0 phantom (estimators + floor + construct-validity calibration on synthetic
data) → one LLM-agent cell with the floor printed → the invariance sweep (nuisances +
setup-axis profiling) → construct validity on planted-ground-truth teams → library v0.1 +
data release + writeup — sequenced so partial results are publishable at each step rather
than gated on a single final deliverable.

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

The first project's paper + library is the load-bearing deliverable — it stands on its own
if the broader program goes no further.

## 6. Open operational decisions

The framing is mature; the operational commitments below are locked in before pilot.

1. **Final battery selection.** The reimplementability pass fixes the 3–4 metrics and, for
   any underspecified published metric, states the clean reference version emmy will use
   (underspecification is itself a comparability finding). This pass also verifies every
   citation before it is printed.
2. **Backbone and harness.** The open-weight lead family, the second family for the
   cross-backbone axis, the one frontier-tier slice, and whether to reuse a battery metric's
   own task harness or a lightweight agent framework.
3. **Construct-axis manipulations.** Which planted properties (incentive structure,
   injected coordination, injected misalignment) produce a collective behavior that joint
   task-performance cannot already separate — chosen so aggregate performance saturates while
   collective _structure_ still varies (Gate C).
4. **Optional rungs.** Whether to carry the dynamic observable, the MARL bridge, and the
   frontier-tier slice — each descopable independently without collapsing the paper.
5. **Co-author search.** A solo program at this ambition is a real risk. An
   LLM-agent-evaluation-credible co-author early would de-risk both the technical work and
   adoption.

**Go/no-go gates.**

- **Gate A (phantom).** Estimators recover known ground truth within a stated trueness bound
  at a feasible sample size. Fail → fix the estimator / sample budget, or drop that metric
  before any LLM-agent run.
- **Gate B (noise floor).** The re-sample floor on real LLM-agent runs is small enough that
  cross-setup signal _could_ clear it. If the floor already swamps plausible effects at
  feasible sample sizes, stop and report the measurement-limit finding rather than run the
  full sweep into noise.
- **Gate C (is-there-a-there-there).** The planted construct axis actually produces a
  collective behavior joint task-performance cannot separate. If aggregate performance
  separates everything, the construct-validity test is vacuous and the manipulation must
  change.

## 7. Landscape position

**The battery metrics themselves are the closest prior art — and emmy's relationship to
them is "item under test," not "competitor."** Each defines a collective-level metric on a
single setup; emmy characterizes their cross-setup comparability. The recently published
fragility/antifragility, misalignment-propensity, and multi-agent-evaluation measures are
single-setup by construction with no invariance test — exactly the open lane.

**Adjacent work and where the gap remains:**

- _Continual-evaluation position work_ (Pacchiardi, Hernández-Orallo et al., _Continual
  Learning Requires Evaluating Trajectories_, 2026) argues current evaluation wrongly assumes
  a frozen artefact and that behavioral-invariance assumptions are eroding. Complementary, not
  competing: they attack the **temporal** invariance axis (behavior drifts as a system keeps
  learning); emmy attacks the **setup** invariance axis (does a metric travel across
  deployments?). Orthogonal and composable; emmy's distinct contribution is the estimation
  rigor — noise floor, phantom calibration — their position paper leaves open, applied to
  collectives. Construct validity and the capabilities/propensities distinction are shared
  primitives, which makes this a framing cite rather than prior art to differentiate from.
- _MARL evaluation protocols_ (Gorsane et al., _Standardised Evaluation Protocol for
  Cooperative MARL_, NeurIPS 2022) are the methodological ancestor: they standardize how to
  report _scalar return_. Emmy ports variance-aware evaluation to _structured collective
  metrics_ and adds a calibration phantom, an estimator-noise floor, and a cross-setup
  invariance axis those protocols don't carry.
- _Precision-aware RL reporting_ (Agarwal et al., _Deep RL at the Edge of the Statistical
  Precipice_, NeurIPS 2021; `rliable`) makes scalar scores variance-aware across few seeds.
  Emmy ports that discipline to non-scalar collective metrics, the re-sample floor playing
  rliable's interval-estimate role.
- _Information-theoretic structure_ (Williams & Beer PID; Lizier transfer entropy; Rosas et
  al. O-information, 2019) supplies candidate descriptors and tools for the interaction-graph
  and contagion metrics. Emmy treats estimability as a first-class gate — these enter only
  behind the Stage-0 phantom; where not estimable at this N and sample budget, that is
  reported, not papered over.
- _Standards bodies_ — NIST AI RMF, NIST CAISI, AISI-UK, CEN-CENELEC JTC 21 — are
  explicitly looking for measurement primitives; none yet exist in mature form for
  multi-agent systems.

## 8. Governance and standards relevance

Audit-friendly architecture is a founding design constraint. The library computes metrics
from action–observation streams with no privileged-access requirement; this matches the
interface conditions under which third-party auditors and regulators must work.

Specific technical-to-policy bridges:

- _Per-metric invariance profiles → comparability standards._ Which collective metrics a
  regulator can compare across two deployments, and which it cannot.
- _Setup-class declarations → transparency standards._
- _No-privileged-access metrics → architecture-agnostic regulation._
- _Negative-space reporting → standards humility._

Coordination targets: AISI-UK methodology team, NIST CAISI staff, CEN-CENELEC JTC 21,
Cooperative AI Foundation, OECD AI standards working groups.

## 9. AI safety relevance

The program's safety relevance is downstream option value — a precondition for other safety
work, not a bet on where risk will concentrate.

**The inspection gap (the headline).** Powerful models from many developers are increasingly
composed into _collectives_ by agent frameworks: multiple model-agents acting on a shared
task or environment. AI-control and single-model evaluation assess the trustworthiness of
_individual_ models; they do not characterize the collective's emergent behavior, and no
external inspection layer for it exists. That surface is already a demonstrated concern, not
a thought experiment: Anthropic's alignment team finds that multi-agent _AI organizations_
can be more effective but _less aligned_ than the individual agents composing them, and
argues such systems need testing for misalignment across organizational structures — exactly
the measurement no single-model check provides
([Anthropic, 2026](https://alignment.anthropic.com/2026/ai-organizations/)).

**Why a comparability instrument is the precondition.** Today a collective-(mis)alignment
metric reported by a vendor on its own setup cannot be trusted by an external auditor on a
_different_ deployment — there is no evidence the number travels. Emmy's map says, per metric,
whether it travels, and on the no-privileged-access surface a third party actually has. That
is the external-inspection layer AI-control (white-box, owner-side) does not provide, and the
precondition for ever comparing one collective to another. It complements AI-control rather
than competing with it: control protocols increasingly involve multiple agents (e.g.
untrusted monitors), but they evaluate the trust relationship, not the collective dynamics
those protocols run inside.

**Falsifiable, composable claims.** Robustness, alignment, and emergent-failure-mode claims
about multi-agent systems are uninterpretable without measurement that travels across testing
setups; with it, they become comparable across labs and falsifiable. This value does not
depend on any particular theory of where multi-agent risk concentrates.

**Early warning — a hypothesis held in reserve.** Whether multi-agent collapses carry the
slow-precursor signatures (rising autocorrelation, variance, recovery time) that precede
regime shifts in climate, ecology, and finance is an open empirical question — some collective
failures may be discrete or adversarial, with no slow precursor. It is a complement to the
headline for catastrophic-risk-focused audiences, pursued only if a planted construct sweep
produces a qualitative transition, and explicitly not part of the first paper.

## 10. What success looks like

**First milestone:** the Stage-0 phantom map — every battery metric's estimator calibrated
against known ground truth, with its trueness bound and noise floor (Gate A). This is the
de-risk gate the rest of the program runs behind, at ~$0 compute.

**Next:** the invariant × construct-valid map produced on small LLM-agent teams — which of
the field's metrics travel across the core nuisances, which only profile across setup axes,
and which track a planted construct that joint task-performance cannot separate (including the
ones that do neither). Library v0.1 + data released so the map is reproducible. At least one
standards-body methodology conversation produces actionable feedback.

**Then:** Paper 1 submitted to an ML venue (NeurIPS / ICML / TMLR), reporting the full map.
Library v0.2 + data released. At least one named external group re-using the library.

The bar held throughout: the whole grid reported, profiles not verdicts, code and data
shared.

## 11. Document map

This is currently the only public document in `docs/`. Additional documentation will be
added as it is finalized for public release.
