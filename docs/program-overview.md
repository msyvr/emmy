# Measurement Foundations for Multi-Agent AI: Program Overview

**Status:** Current synthesis, 2026-05-25. Pre-experiment. This document is the
canonical reference for the program; other documents will be added to `docs/`.

**Summary.** Emmy builds measurement foundations for multi-agent
AI: evaluation-invariant observables for collective behavior — quantities computed
from action–observation streams, without privileged model access, designed to stay
stable across changes in evaluation setup. Where such observables can be found, two
payoffs follow — cross-paper claims about coordination, robustness, and failure become
comparable, and external evaluators gain an inspection layer for deployed agent
collectives that single-model methods aren't equipped to provide. The candidate
observables are multi-scale and include dynamic response functions, so a collective's
fragility under stress can be a measured quantity. The orientation is drawn from
physics, systems biology, and statistical mechanics, and complements
benchmark-based evaluation. This is pre-experiment work: the first deliverable is
a paper plus open-source library — an open, reproducible characterization of which
observables travel across evaluation setups and which discriminate, not a set
asserted in advance.

---

## 1. Motivation

Multi-agent AI systems — large-language-model agent ensembles,
reinforcement-learning agent collectives — are moving from research into
deployment faster than their measurement practices are maturing. The tools we
have were built for other things: single-model evaluation and interpretability
characterize one model at a time, and multi-agent results are dominated by joint
return on benchmarks and ad-hoc behavioral indicators chosen per paper. None of
these characterize what a collective does in a way that travels beyond the setup
that produced it. Even applied practice shows the same shape: Anthropic's engineering
guide to evaluating AI agents is organized entirely around _single_ agents — coding,
conversational, research, computer-use — and names multi-agent collaboration as a
frontier its techniques will still need to adapt to ([Anthropic Engineering,
2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

Four consequences:

- **Cross-paper claims do not compose.** Robustness findings reported in one
  evaluation setup do not transfer because the only quantities reported in
  common are insufficient to characterize underlying collective behavior.
- **Evaluation-setup dependence is implicit.** Every claim is conditional on
  algorithm, seed, reward shape, observation layout, and environment
  parameters, but these dependencies are not characterized. Reproducibility
  is point-estimate-within-setup, not class-of-setup.
- **Safety-relevant questions are unreachable.** Whether a controlled
  experiment maps to deployment, whether a collective is fragile under stress,
  whether diversity helps or hurts robustness — none answerable cleanly
  without measurement that travels across evaluation setups.
- **The clock is capability, not convention.** Reporting conventions are forming
  now, but the binding deadline is the trajectory: the field is racing toward
  autonomous AI R&D, carried out by agent collectives — "synthetic teams" of AI
  systems managing one another. Anthropic co-founder Jack Clark, in his _Import
  AI_ newsletter, puts ~60% on AI systems autonomously running their own R&D by
  the end of 2028
  ([Import AI 455](https://importai.substack.com/p/import-ai-455-automating-ai-research)).
  The inspection layer for collective behavior has to exist before collectives do
  high-stakes autonomous work — and once humans leave the loop, external
  measurement is the oversight handle that remains.

The gap is foundational, and closing it starts with finding out whether observables
with explicit invariance properties exist for collective behavior at all — and which.

## 2. Thesis

The program's first project is empirical and diagnostic. It does not assume a
canonical set of evaluation-invariant observables exists and set out to validate it;
it asks whether such observables exist at all, and which. Take a battery of candidate
observables for collective behavior — information-theoretic, dynamical, and
structural quantities, and their combinations — and measure them across existing
multi-agent environments and policies under a defined group of evaluation-setup
transformations (seed, algorithm, reward scale, observation layout, environment
reparameterization). For each candidate, characterize two things jointly: how
invariant it is across those transformations, and how well it discriminates
collective behaviors that joint return cannot. The deliverable is the resulting
map — which observables travel across setups, which discriminate, and where the two
trade off — released as an open library of measured observables with their
invariance-and-discrimination profiles. Whether a usefully-invariant,
usefully-discriminating set exists is the question the study answers, not a premise
it assumes; a finding that none exists, or that invariance and discrimination are
fundamentally in tension, is itself a result worth publishing and a guide for anyone
building multi-agent evaluations.

**The stance is measurement-first.** Most multi-agent and safety work starts from a
human-meaningful concept — cooperation, deception, empowerment, power-seeking — and
looks for a proxy that captures it in some environment. This program inverts that: it
starts from quantities that are operationally defined and measurable from behavior,
and treats both their invariance structure and their relationship to safety as
empirical questions to be characterized, not settled by naming. The discrimination
target is deliberately left open — the aim is to surface emergent collective
properties without pre-deciding which matter or under what conditions they show up.
The lens throughout is safety/alignment rather than capability (the two are confounded
in practice, so both will appear); the cooperative–neutral–adversarial range of team
objectives is one slice that gets spanned in passing, not a privileged axis. The move
is not new — it is the discipline evolutionary biology brought to traits once assumed
adaptive (Gould & Lewontin's "spandrels," 1979): build the controlled alternative and
measure, rather than read function into what you observe. The approach is deliberately
unusual for AI safety — its payoff depends on the bridge from measured observables to
safety-relevant behavior being real, which the program treats as something to
demonstrate (see §9), not assert.

The framework rests on six commitments.

1. **Multi-scale observables.** Component-level, pairwise, _and_ system-level
   quantities in the candidate battery from the start. Not reductionist. System-
   level observables (correlation length, integrated information, response
   functions) validate the component-level ones by their predictive
   relationship — not the other way around.
2. **Explicit invariance groups.** Each observable is published with a stated
   set of evaluation-setup transformations under which it is invariant.
   Invariance is demonstrated empirically and, where tractable, analytically.
3. **Dynamic observables as instrumentation.** Response functions and their
   curvature (fragility / anti-fragility) are part of the candidate battery from
   the start, defined as perturbation-response quantities. Perturbation
   protocols are how dynamic observables are _measured_, not a separate
   empirical program. Anti-fragility becomes a measurable curvature.
4. **Setup reflexivity.** Every empirical claim reports the evaluation-setup
   class under which it was produced _and_ the negative space it admits but
   does not measure. Reproducibility means setup-class reproducibility.
5. **Cross-substrate transfer.** Observable definitions are substrate-agnostic.
   First demonstrations are in cooperative MARL; framework design admits
   later extension to LLM-multi-agent and active-inference systems with the
   same observables.
6. **Equilibrium-grounded measurement scaffolding.** Equilibrium statistical
   mechanics and renormalization-group theory provide the load-bearing
   machinery for defining observables that compose across scales and stay
   stable under evaluation-setup changes. The same machinery extends to
   non-equilibrium regime shifts: critical-slowing-down (CSD) indicators —
   autocorrelation, variance, recovery time, correlation length — emerge from
   the physics of systems near critical points and are computable from the same
   rollout data. That extension is the validated pattern used in climate,
   ecology, and finance; here it is one application of the scaffolding, pursued
   as a hypothesis under test (see §9), not a foundational claim.

## 3. Methodological discipline

The discipline matched to a discovery study is not pre-registration — that is a
tool for *confirmatory* hypothesis-testing, and forcing it onto exploratory
characterization buys rigidity and perverse design incentives (a test no one wants
to fail gets designed to be easy to pass) more than rigor. The discipline here is
**systematic coverage, complete reporting, and reproducibility**:

- **Declared scope, swept completely.** The candidate battery, the transformation
  group, and the environments/policies are fixed and stated up front, then swept as
  a full grid. The guard against cherry-picking is that the *whole* grid is reported
  — including observables that neither travel nor discriminate — not a pre-committed
  go/no-go on a favored hypothesis.
- **Profiles, not verdicts.** Invariance and discrimination are reported as
  continuous quantities with seed-variance bands, not binary "is/isn't invariant"
  labels. Validity is not binary; "how invariant, under which transformations,
  within what bounds" is the right unit.
- **Reproducibility as the primary contribution.** Code, configs, environments, and
  the measured-observable library are released so the community can re-analyze,
  re-run, and build on the map — the open-science investment that compounds far more
  than a pre-registration does.
- **Negative results are inherent, not bolted on.** Because the deliverable is the
  map, an observable that fails to travel or discriminate is a reported data point,
  not a "failed" experiment — no pre-registration is needed to make null findings
  publishable.

This is not a purely academic concern. Production eval practice hits the same wall from
the other side: shared state across trials — leftover files, cached data, resource
contention — produces _correlated_ failures that look independent, and the numbers
become unreliable for measuring the system ([Anthropic Engineering,
2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). Harness
hygiene handles that case; the observables here go one step further — when the
correlation comes from the agents themselves sharing structure, it is treated as a
quantity to measure, not only a confound to isolate.

Pre-registration is held in reserve for a later, *specific* confirmatory claim — if
and when the program makes one (e.g. "observable O is invariant under transformation
T within bound ε"). That is the regime it was built for; this first, map-making phase
is not.

## 4. First project: paper 1 + library v0.1

**Working title:** _Which observables of multi-agent collective behavior travel across
evaluation setups, and which discriminate? An empirical characterization._

The paper characterizes four things, each reported as a measured profile across the
battery rather than asserted:

1. **Discrimination.** Which candidate observables distinguish configurations that
   joint return cannot — reported per observable.
2. **Invariance.** The measured invariance profile of each candidate across the named
   transformation group, with seed-variance bands. The battery deliberately includes
   observables expected _not_ to travel; reporting them is what keeps the
   discrimination finding non-vacuous.
3. **Transfer.** Whether observable definitions and interpretations carry across at
   least two environments, even where numerical values are environment-specific.
4. **System-validation.** Whether component-level and pairwise observables track
   system-level ones (correlation length, response-function curvature) under
   controlled perturbation — the evidence for their role in the measured set.

**Candidate observables (the battery):**

- _O1: Conditional joint policy entropy._ H(a₁,…,a_N | s).
- _O2: Normalized pairwise mutual information._ I(a_i; a_j | s), normalized.
- _O3: Response function and fragility curvature._ R(σ) measured under
  perturbation of stress parameter σ; fragility index = ∂²R/∂σ² at operating
  point. Sign convention: negative = fragile, near zero = robust, positive =
  anti-fragile.
- _O_neg (illustrative non-traveler): per-step reward variance._ References reward, so
  it cannot be setup-invariant; a sharper non-traveler would fail a _non-trivial_
  transformation, not reward rescaling.

**Evaluation-setup transformation group:**

- _T1 (seed swap)_ — invariance expected, near-perfect.
- _T2 (algorithm swap)_ — invariance partial; deviations themselves
  informative. Pair to be confirmed (see §6).
- _T3 (reward rescaling)_ — a near-trivial case (any action-only observable is
  invariant, any reward-referencing one is not), so it carries little evidential
  weight; included for completeness, not as a discriminating test.
- _T4 (observation channel permutation)_ — invariance expected.
- _T5 (minor environment reparameterization)_ — invariance _not_ expected;
  included to characterize the boundary of the invariance group.

**Library v0.1.** Open-source Python library computing the candidate
observables from action-observation streams. Audit-friendly architecture by
construction: no privileged access to weights, training data, or internal
representations required. Initial scope is observables-and-API; reference
environments and analysis examples populate post-pilot.

**Plan.** Pipeline and first sweeps on existing environments, then the full
characterization grid, then library v0.1 + data release, then the writeup and public
framing post — sequenced so partial results are publishable at each step rather than
gated on a single final deliverable.

## 5. Later directions

Beyond the first project, in roughly increasing order of ambition:

- **Phenomenology.** A second paper applying the observables — likely a
  characterization of intermittent adversarial injection, with response curves and
  fragility profiles as the primary findings.
- **Scaling laws and phase diagrams.** How the observables scale with N, bandwidth,
  compute, and reward dimensionality; where in dimensionless-quantity space collective
  behaviors qualitatively change; the perturbation taxonomy as a phenomenology of
  fragility profiles.
- **Comparative and applied.** Active-inference contrast (the same observables on
  multi-agent active-inference systems); diversity and collective robustness;
  real-world mapping conditions.

The first project's paper + library is the load-bearing deliverable — it stands on its
own if the broader program goes no further.

## 6. Open operational decisions

The framing is mature; the operational commitments below need to be locked
in before pilot.

1. **Algorithm pair for T2.** Initial candidate MAPPO + IPPO is too
   within-family (both PPO). Adding at least one structurally different
   algorithm (QMIX or similar value-decomposition method) is under
   consideration to make the invariance claim non-vacuous.
2. **Environment pair.** Both cooperative is potentially too within-class.
   Either add a mixed-motive environment or be explicit that paper 1 is
   scoped to cooperative-only and the transfer claim is tested in a follow-on.
3. **R(σ) commitment.** Time-to-recovery as primary; joint return as
   secondary check. Both reported in paper 1.
4. **Paper 1 scope.** Four claims across two environments × two algorithms is
   ambitious for a first project. Considering whether a tighter scoping
   would land harder; because the deliverable is the map, partial results are
   publishable regardless.
5. **Position paper (yes/no).** Establishes presence and anchors framing before
   paper 1 ships.
6. **Library v0.1 scope.** Minimal (O1, O2, O_neg) at v0.1; add O3 in v0.2
   after dynamic-observable empirical work.
7. **Co-author search.** Solo program at this ambition is a real risk.
   A MARL-credible co-author early would significantly de-risk both
   technical work and adoption.

## 7. Landscape position

**Closest proximate work**: Polani / Tiomkin, multi-agent empowerment
(April 2026) — the nearest methodological neighbors, information-theoretic and
MARL-native. But empowerment is a _quantity_ (an action→future channel capacity),
not an invariance framework: a candidate observable to characterize, not a
competing standard. The proximity matters for collaboration and credibility — a
MARL-native co-author or adopter — more than for competition.

**Adjacent communities and where the gap remains**:

- _MARL theory and evaluation_: Joel Leibo (Melting Pot) is the closest to
  thinking about measurement at the substrate-vs-population level; others
  (Tuyls, Foerster, Whiteson, Stone, Bowling) are benchmark-focused.
- _Info-theoretic MARL measurement_: Lizier, Williams & Beer PID,
  transfer-entropy decompositions, synergy/redundancy frameworks.
  Sophisticated but not organized around evaluation-setup invariance.
- _Critical-slowing-down_: Scheffer/Lenton tipping-point literature has 20+
  year empirical track record across climate, ecology, finance. Application
  to multi-agent AI is novel; the methodological transfer is established.
- _Renormalization group in ML_: present but scattered; no programmatic use
  for measurement standards.
- _Standards bodies_: NIST AI RMF, NIST CAISI, AISI-UK, CEN-CENELEC JTC 21
  — explicitly looking for measurement primitives; none yet exist in mature
  form for multi-agent systems.

**Positioning vs. Friston's manifesto.** Measurement-theoretic complement to
the FEP / active-inference framework. Not competing foundational text;
substrate-invariant measurement that applies to active-inference agents as
one of several substrates. The active-inference arc is visible in the program
but does not load-bear the first paper.

## 8. Governance and standards relevance

Audit-friendly architecture is a founding design constraint. The library
computes observables from action-observation streams with
no privileged-access requirement; this matches the interface conditions
under which third-party auditors and regulators must work.

Specific technical-to-policy bridges:

- _Fragility curvature → deployment-readiness criteria._
- _Setup-class declarations → transparency standards._
- _Cross-substrate observables → architecture-agnostic regulation._
- _Diversity / role observables → systemic-risk analysis._
- _Negative-space reporting → standards humility._
- _CSD indicators → early-warning monitoring for deployed AI systems._

Coordination targets: AISI-UK methodology team, NIST CAISI staff, CEN-CENELEC
JTC 21, Cooperative AI Foundation, OECD AI standards working groups.

## 9. AI safety relevance

The program's safety relevance is a precondition for other safety work, not a bet on
where risk will concentrate.

**The inspection gap (the headline).** Powerful models from many developers are
increasingly composed into _collectives_ by agent frameworks: multiple model-agents
acting on a shared task or environment. AI-control and single-model evaluation assess
the trustworthiness of _individual_ models; they do not characterize the collective's
emergent behavior, and no external inspection layer for it exists. That collective
surface is already a demonstrated concern, not a thought experiment: Anthropic's
alignment team finds that multi-agent _AI organizations_ can be more effective but
_less aligned_ than the individual agents composing them, and argues such systems
need testing for misalignment across organizational structures — exactly the
measurement no single-model check provides
([Anthropic, 2026](https://alignment.anthropic.com/2026/ai-organizations/)). The
framework builds that layer — rigorous, characterized measurement of collective behavior from
action–observation streams with no privileged model access, which is exactly the
surface a third-party evaluator or regulator can observe. It complements AI-control
rather than competing with it: control protocols increasingly involve multiple agents
(e.g. untrusted monitors), but they evaluate the trust relationship, not the
collective dynamics those protocols run inside.

**Falsifiable, composable claims.** Robustness, alignment, and emergent-failure-mode
claims about multi-agent systems are uninterpretable without measurement that travels
across testing setups; with it, they become comparable across labs and falsifiable.
This is the precondition for downstream evaluation, audit, and standards work, and its
value does not depend on any particular theory of where multi-agent risk concentrates.

**Anti-fragility and cross-substrate generalization.** The dynamic observables turn
"how a collective degrades under stress" into a measurable, comparable object;
substrate-agnostic definitions let claims survive architectural change.

**Early warning for catastrophic transitions — a hypothesis under test.** The framework
includes critical-slowing-down indicators (autocorrelation, variance, recovery time,
correlation length) that are validated precursors of catastrophic regime shifts in
climate, ecology, and finance and are computable on rollout data without privileged
access. _Whether multi-agent AI transitions carry the same signatures is an open
empirical question_ — some collective failures may be discrete or adversarial, with no
slow precursor. The program treats early warning as a hypothesis to test, not a
guarantee: where the signatures exist, the indicators apply; where they don't, that
boundary is itself a finding. This is the natural bridge for catastrophic-risk-focused
audiences — a complement to the headline, not the foundation.

## 10. What success looks like

**First milestone:** First characterization map produced across the declared battery ×
transformation grid on at least one environment — which observables travel, which
discriminate, and where they trade off (including the ones that do neither). Library
v0.1 + data released so the map is reproducible. At least one standards-body
methodology conversation produces actionable feedback.

**Next:** Paper 1 submitted to a MARL or ML venue (NeurIPS / ICML / TMLR),
reporting the full map across two environments. Library v0.2 + data released. At least
one named external group re-using the library. Paper 2 (phenomenology of a specific
observable) underway.

The bar held throughout: the whole grid reported, profiles not verdicts, code and data
shared.

## 11. Document map

This is currently the only public document in `docs/`. Additional
documentation will be added as it is finalized for public release.
