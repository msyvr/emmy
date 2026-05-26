# Measurement Foundations for Multi-Agent AI: Program Overview

**Status:** Current synthesis, 2026-05-25. Pre-experiment. This document is the
canonical reference for the program; other documents will be added to `docs/`
as they are finalized for public release.

**One-paragraph framing.** Emmy builds measurement foundations for multi-agent
AI: a small, canonical set of evaluation-invariant observables for collective
behavior, computed from action–observation streams without privileged model
access. Two payoffs follow — cross-paper claims about coordination, robustness,
and failure become comparable, and external evaluators gain an inspection layer
for deployed agent collectives that single-model methods do not provide. The set
is multi-scale and includes dynamic response functions, so a collective's
fragility under stress is a measured quantity. The orientation is drawn from
physics, systems biology, and statistical mechanics, and complements
benchmark-based evaluation. This is pre-experiment work: the first deliverable is
a paper plus open-source library with pre-registered, falsifiable claims.

---

## 1. Why this work

Multi-agent AI systems — large-language-model agent ensembles,
reinforcement-learning agent collectives — are moving from research into
deployment faster than their measurement practices are maturing. The tools we
have were built for other things: single-model evaluation and interpretability
characterize one model at a time, and multi-agent results are dominated by joint
return on benchmarks and ad-hoc behavioral indicators chosen per paper. None of
these characterize what a collective does in a way that travels beyond the setup
that produced it.

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
- **Standards are crystallizing now.** Reporting conventions are being formed
  through accretion. The window for foundational influence is the next
  12–18 months.

The gap is foundational, not phenomenological. Closing it requires committing
to a canonical set of observables with explicit invariance properties, not
producing another phenomenology paper.

## 2. Thesis

The program establishes a measurement framework: a small canonical set of
observables for collective state and collective response, each defined with
an explicit invariance group under evaluation-setup transformations,
demonstrated to discriminate behaviors that joint return cannot, and packaged
as community infrastructure that compounds across the field rather than
accumulating as silo'd benchmarks.

The framework rests on six commitments.

1. **Multi-scale observables.** Component-level, pairwise, _and_ system-level
   quantities in the canonical set from the start. Not reductionist. System-
   level observables (correlation length, integrated information, response
   functions) validate the component-level ones by their predictive
   relationship — not the other way around.
2. **Explicit invariance groups.** Each observable is published with a stated
   set of evaluation-setup transformations under which it is invariant.
   Invariance is demonstrated empirically and, where tractable, analytically.
3. **Dynamic observables as instrumentation.** Response functions and their
   curvature (fragility / anti-fragility) are part of the canonical set from
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

The single most distinctive feature of the program is the **falsification
phase** built into Year 1.

- A pre-registered set of go/no-go criteria for the foundational claims —
  posted to OSF before pilot experiments begin.
- A negative-control observable (per-step reward variance, currently)
  included specifically because it is expected to fail the T3 invariance
  test. Demonstrates that the invariance claim is discriminating, not
  vacuous.
- Three predefined pivots, one for each likely failure mode, so that
  negative results trigger a defined next step rather than ad-hoc reframing.
- Three falsification streams: Layer 2 empirical invariance, Layer 4
  policy-relevance, and an adoption test seeking concrete commitments to
  use the library from named labs.
- Willingness to publish negative results, including a "null result protocol"
  paper that documents what didn't work and why.

Pre-registered go/no-go criteria with predefined pivots are standard in
physics and clinical trials, almost unheard of in ML. The discipline pays
off twice — as epistemic insurance against years of wasted work, and as a
credibility differentiator that few research programs at this stage carry.

## 4. First project: paper 1 + library v0.1

**Working title:** _Evaluation-invariant observables for multi-agent
collectives: invariance, discrimination, transfer._

The paper makes four falsifiable claims:

1. **Discrimination.** Configurations indistinguishable by joint return are
   distinguishable by the canonical observables.
2. **Invariance.** Each canonical observable is invariant under the named
   evaluation-setup transformation group, within seed-variance bounds. A
   negative-control observable is included that is _not_ invariant.
3. **Transfer.** Observable definitions and interpretations transfer across
   at least two cooperative environments, even though numerical values are
   environment-specific.
4. **System-validation.** Component-level and pairwise observables track
   system-level ones (correlation length, response-function curvature) under
   controlled perturbation, validating their place in the canonical set.

**Canonical observables (current candidates):**

- _O1: Conditional joint policy entropy._ H(a₁,…,a_N | s).
- _O2: Normalized pairwise mutual information._ I(a_i; a_j | s), normalized.
- _O3: Response function and fragility curvature._ R(σ) measured under
  perturbation of stress parameter σ; fragility index = ∂²R/∂σ² at operating
  point. Sign convention: negative = fragile, near zero = robust, positive =
  anti-fragile.
- _O_neg (negative control): per-step reward variance._ Expected to fail T3.

**Evaluation-setup transformation group:**

- _T1 (seed swap)_ — invariance expected, near-perfect.
- _T2 (algorithm swap)_ — invariance partial; deviations themselves
  informative. Pair to be confirmed (see §6).
- _T3 (reward rescaling)_ — invariance expected for O1–O3, violation
  expected for O_neg.
- _T4 (observation channel permutation)_ — invariance expected.
- _T5 (minor environment reparameterization)_ — invariance _not_ expected;
  included to characterize the boundary of the invariance group.

**Library v0.1.** Open-source Python library computing the canonical
observables from action-observation streams. Audit-friendly architecture by
construction: no privileged access to weights, training data, or internal
representations required. Initial scope is observables-and-API; reference
environments and analysis examples populate post-pilot.

**Timeline:** 6 months. M1–M2 falsification phase; M3–M4 main empirical work;
M4–M5 library v0.1 release; M5–M6 writeup and public framing post.

## 5. Phasing beyond Year 1

- **Year 1 (months 7–12).** Paper 2: first phenomenology paper using the
  observables. Likely target: characterization of intermittent adversarial
  injection with response curves and fragility profiles as the primary
  findings.
- **Years 2–3.** Scaling laws (how do canonical observables scale with N,
  bandwidth, compute, reward dimensionality), phase diagrams (where in
  dimensionless-quantity space do collective behaviors qualitatively
  change), perturbation taxonomy as a phenomenology of fragility profiles.
- **Years 4–5.** Active-inference contrast (same observables on
  multi-agent active-inference systems); diversity and collective robustness;
  real-world mapping conditions.

The program is structured to deliver foundational results in Year 1 even if
the multi-year program loses funding or scope. Year 1's paper + library is
the load-bearing deliverable.

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
4. **Paper 1 scope.** Four claims across two environments × two algorithms
   in six months is ambitious. Considering whether a tighter scoping
   would land harder; the pre-registration handles publishability of
   partial results regardless.
5. **Position paper (yes/no).** Companion to pre-registration; establishes
   presence and anchors framing before paper 1 ships.
6. **Library v0.1 scope.** Minimal (O1, O2, O_neg) at v0.1; add O3 in v0.2
   after dynamic-observable empirical work.
7. **Co-author search.** Solo program at this ambition is a real risk.
   A MARL-credible co-author in Year 1 would significantly de-risk both
   technical work and adoption.

## 7. Landscape position

**Closest proximate work**: Polani / Tiomkin, multi-agent empowerment
(April 2026). Info-theoretic depth, MARL substrate, three steps from this
program's framework. Contribution-distinctness window is months, not years.

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

## 10. What success looks like at month 6 and month 12

**Month 6:** Falsification-phase outcome resolved. Either the foundational
invariance claims survive empirical pressure (proceed to writeup of full
paper), or they don't (publish null-result paper, revise framework). Library
v0.1 released with observables-and-API surface even if scope is minimal. At
least one Layer 4 conversation with a standards-body methodology team
produces actionable feedback.

**Month 12:** Paper 1 submitted to a MARL or ML venue (NeurIPS / ICML /
TMLR). Library v0.2 released with O3 added. At least one named adoption
commitment from an external research group. Paper 2 (phenomenology) underway.

The bar held throughout: transparent reporting of what the experiments show.

## 11. Document map

This is currently the only public document in `docs/`. Additional
documentation will be added as it is finalized for public release.
