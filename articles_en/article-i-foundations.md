---
title: "Mathematical Foundations of the Computational Representation of Biological Systems"
subtitle: "An Axiomatic Meta-Model for Computational Precision Medicine"
author: "Computational analysis — BioSpace framework"
date: "July 2026"
lang: en
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.15
toc: true
toc-depth: 2
numbersections: true
colorlinks: true
---

# Abstract

In computational medicine, a patient's representation is typically treated as a preprocessing step, subordinate to the inference algorithm that follows it — its quality assessed only indirectly, through the performance of a downstream classifier or regressor. This article defends and formalizes a different position: the computational representation of a biological system is a legitimate scientific object, whose properties can and should be defined, tested, and refuted independently of any specific inference algorithm.

We present a meta-model with nine formal entities, five fundamental axioms (four of which restate properties already tested empirically as software contracts), a formalization of the representation space as a metric space — with an honest account of which implemented geometries satisfy that condition and which do not —, an algebraic characterization of admissible operators as a monoid, fundamental theorems connecting the framework to classical results in mathematics (the Banach fixed-point theorem, continuity), a catalog of invariants — some confirmed empirically, one refuted — and a categorical organization demonstrated by two independent realizations of a functor between representation categories of distinct data sources.

This is the first of a planned series of articles. The following ones report empirical validation on Type 2 Diabetes Mellitus, using two structurally independent real public data sources (NHANES, a cross-sectional metabolic health survey; and UCI Diabetes 130-US Hospitals, longitudinal administrative hospitalization records).

# 1. Introduction

Most of the machine-learning-for-healthcare literature treats patient representation as an engineering artifact: a feature table, chosen pragmatically to feed a predictive model. When the representation changes — a different set of variables, a different normalization, a different missing-data encoding — this is typically logged as a different version of the pipeline, not as a change in a mathematical object with its own properties, verifiable independently of whichever model comes afterward.

The BioSpace framework starts from a different central hypothesis: the computational representation of a biological system — not the algorithm that operates on it — is the correct unit of scientific rigor. If this hypothesis is correct, it should be possible to: (i) formally define what counts as an admissible representation; (ii) declare properties that every admissible representation must satisfy, in the form of contracts verifiable by automated test, not merely described in prose; (iii) show that these properties hold — or honestly report when they do not — against synthetic data with known ground truth and, decisively, against real clinical data; and (iv) demonstrate that the same representation infrastructure remains valid when the data source and the inference algorithm both change.

This article establishes the formal apparatus — the meta-model, the axioms, the space's structure, the theorems, the invariants, the categorical organization, and the contracts. Extensive empirical validation — on Type 2 Diabetes Mellitus, using two real, independent public sources (NHANES and UCI Diabetes 130-US Hospitals), structurally very different from each other (cross-sectional versus longitudinal administrative) — is reported in the articles that follow in this series, to which this one serves as the foundation.

# 2. The Formal Meta-Model

We define a biological system as a pair

**B = (E, T)**

where E represents the organism's physiological state and T its temporal evolution. This definition precedes any clinical observation: the biological system exists independently of exams, sensors, or computational models. Consequently,

**B ∉ X**

that is, the biological system does not belong to the computational space X — it belongs to the physical world. This distinction, although it may seem obvious, has a direct practical consequence: no property of the representation space can be used to make claims about the biological system itself, only about its computational image.

An observation O(B) is a set of measurements M taken on B at an instant t. Each measurement is processed by a semantic domain D — a clinically coherent grouping of variables (for example, a glycemic domain, a cardiovascular domain) — which encodes it into one or more characteristics (Features), preserving provenance (which raw measurement each characteristic came from) and completeness (whether the measurement was in fact available, or was imputed).

A representation R aggregates one or more semantic domains. Applied to an observation, it produces a representation vector — a point in the representation space X:

**R : O(B) → X**

The complete composition, from the biological system to the point in the space, is:

**B → O(B) → R(O(B)) ∈ X**

Over X, a geometry G assigns a notion of distance or similarity between points (Section 4). When the same biological system is observed repeatedly over time, the resulting sequence of points forms a trajectory Γ — the fundamental object becomes the entire trajectory, not an isolated instant, whenever the scientific question is about dynamics, not state. A phenotype F is a subset of X defined by a membership rule; a cohort C is a collection of biological systems, each with its own trajectory, all represented in the same space X.

A complete meta-model is, therefore, the tuple M = (B, O, D, R, X, G, Γ, F, C) — nine entities, each with a precise formal definition, serving as shared vocabulary for everything that follows.

# 3. Axiomatic Foundations

The previous sections introduced the meta-model's entities definitionally. We now state five axioms that govern how these entities relate to one another. Each was chosen because it corresponds to a property already implemented as an executable formal contract (Section 9) — the axioms are not free-standing mathematical decoration; one of them, as reported below, was violated by a real implementation bug before being fixed.

## Axiom 1 (Existence)

For every observed biological system, at least one admissible computational representation exists:

**∀ B with O(B) ≠ ∅, ∃ R(B) ∈ X**

The qualification O(B) ≠ ∅ is deliberate: existence is guaranteed constructively by R being a total function of the available observations, not of the biological system itself. A never-observed system has no representation — there is, trivially, nothing yet for R to act on.

## Axiom 2 (Observational Determinism)

Under the same set of observations, the representation is unique:

**O₁ = O₂ ⟹ R(O₁) = R(O₂)**

This axiom restates the Reproducibility Contract (Section 9). It is tested, not merely assumed: the regression suite confirms bit-identical output across repeated builds from the same input data.

## Axiom 3 (Semantic Preservation)

The representation preserves a declared biological equivalence relation ~:

**bᵢ ~ bⱼ ⟹ R(bᵢ) ~ R(bⱼ)**

This axiom restates the Semantic Preservation Contract. It is important to state explicitly that ~ must be declared per domain — the meta-model provides no universal biological equivalence relation of its own. The axiom constrains whatever equivalence a domain author chooses to declare; it is not evidence that a privileged notion of biological equivalence exists a priori.

## Axiom 4 (Temporal Consistency)

If tᵢ < tⱼ, the representation at tᵢ cannot depend on observations that occurred after tᵢ:

**tᵢ < tⱼ ⟹ R(B, tᵢ) is independent of O(B, tⱼ)**

This axiom restates the Temporality Contract — and it is, as far as we know, the only one of these five axioms whose violation was caught empirically, not merely anticipated in theory: an incorrectly propagated `as_of` parameter allowed, in an earlier version of the implementation, a later observation to leak into an earlier time cut — a real bug, fixed, now covered by a dedicated regression test. We report this episode because it is direct evidence that declaring a property axiomatically does not, by itself, guarantee that an implementation satisfies it — the axiom is a specification against which code can fail, not a description of code that cannot fail.

## Axiom 5 (Closure)

Every admissible operator maps the representation space to itself:

**T : X → X**

Unlike Axioms 1–4, this one is not independently falsifiable by testing an existing implementation — it is instead a definitional boundary of the theory: an operator whose output leaves X is, by construction, outside the scope of this framework. We declare it as an axiom because it is the condition that makes the algebraic structure of Section 5 possible, not because it is itself an empirical claim.

# 4. Metric Structure of the Representation Space

We formalize the representation space as a triple

**(X, d, Σ)**

where X is the set of admissible representations, d is a distance function, and Σ is the semantic structure that assigns each coordinate to a named domain — already implemented as a mapping from domain name to a list of characteristics, distinct from the flattened vector that a geometry or algorithm actually consumes.

A function d: X × X → ℝ≥0 is a metric on X if, for all x, y, z ∈ X, it satisfies non-negativity, identity of indiscernibles (d(x,y)=0 iff x=y), symmetry, and the triangle inequality (d(x,z) ≤ d(x,y) + d(y,z)).

Not every implemented geometry satisfies all four properties, and we report this as a real limitation, not a footnote to be avoided. Euclidean distance and approximate geodesic distance over a k-nearest-neighbor graph are true metrics by construction. Dynamic Time Warping (DTW) is well documented in the sequence-alignment literature as generally not satisfying the triangle inequality — and, resolving a question that earlier informal attempts in this project had left open, this failure was confirmed for the specific implementation used here: using the published counterexample of Tralie et al. (2022) — three sequences a = (−1,…,−1,0), b = (−1,0,1), c = (0,1,…,1), each element repeated more than once —, we verified that this project's DTW implementation violates the triangle inequality, both with and without its optional path-length normalization. We therefore treat DTW as a dissimilarity, not a metric, and restrict the Section 6 theorems that depend on completeness to the geometries confirmed as true metrics.

# 5. Algebra of Operators

Let T = {T : X → X} be the set of admissible operators under Axiom 5. Composition T₂∘T₁, an identity operator I(x)=x for all x ∈ X, and associativity ((T₃∘T₂)∘T₁ = T₃∘(T₂∘T₁)) give (T, ∘, I) the structure of a monoid. Associativity requires no proof specific to this framework — it is inherited for free from function composition in general, once closure is guaranteed by Axiom 5.

We stop short of claiming a group structure, because that would be false: most operators implemented in this project — phenotyping, dimensionality reduction, aggregation — are not invertible. A K-Means assignment or a PCA projection loses information and admits no T⁻¹ that recovers the original representation. The algebra of admissible operators in this framework is a monoid, not a group, and we state this precisely rather than reaching for a stronger structure the implementation does not support.

# 6. Fundamental Theorems

## Theorem 1 (Existence of a stationary state under mean-reverting dynamics)

Consider a single coordinate of the representation space evolving under a mean-reverting operator, xₜ₊₁ = φxₜ + (1−φ)μ + εₜ. Discarding the stochastic term εₜ, the deterministic map T(x) = φx + (1−φ)μ satisfies |T(x) − T(y)| = |φ|·|x − y| for all x, y. If |φ| < 1, T is a contraction on (ℝ, |·|), and by the Banach fixed-point theorem, T has a unique fixed point x* = μ, to which every trajectory converges under iteration.

The scope of this result is stated precisely so as not to be read more broadly than it is: it holds per coordinate, because the fitted model is diagonal — each characteristic's φ is estimated independently —, not as a claim of a single global contraction over the entire multidimensional space. The condition |φ| < 1 is exactly the stability criterion this project already computes and tests for each characteristic, which makes the theorem's hypothesis directly verifiable against real data, not merely assumed: in empirical validation on real multi-encounter hospital trajectories (UCI Diabetes 130-US Hospitals, 16,773 patients), the thirteen characteristics with a fitted dynamic satisfy |φ| < 1, guaranteeing a unique stationary mean for each.

## Theorem 2 (Continuity of the representation, with documented exceptions)

If R is continuous, small biological perturbations produce small representational perturbations: for every ε > 0 there exists δ > 0 such that d_B(B₁,B₂) < δ implies d_X(R(B₁),R(B₂)) < ε.

We do not claim that this holds as a general property of the meta-model, because it demonstrably does not. Continuous-valued domains under fixed completeness weight are Lipschitz-continuous — a z-score is an affine, hence Lipschitz, function of the raw measurement. But the completeness-weighted z-score has a genuine discontinuity at the missing-data boundary: moving from "present" to "absent" changes a characteristic's weight discretely, not continuously. Ordinal and categorical encodings are inherently discrete and are not continuous in the input variable in any useful sense. Theorem 2, therefore, holds on the subset of domains that use purely continuous encoding under fixed completeness, and does not hold across a missing-data transition nor within categorical domains. We report this as a real structural limitation of the current representation, not a caveat to be swept under the rug.

## Corollary 1 (Indistinguishability under identical representation)

If R(B₁) = R(B₂), then M(R(B₁)) = M(R(B₂)) for any function M defined over X.

We call this a corollary, not a theorem, because it is an immediate consequence of the well-definedness of functions — M applied to equal arguments produces equal results, by the very definition of "function" — not a result requiring proof specific to this framework. Its value is not in mathematical depth; it lies in making explicit a claim left implicit throughout the entire architecture: an inference algorithm M has access only to R(B), never to B directly, so that any two biological systems the meta-model represents identically are, by construction, indistinguishable to any downstream algorithm. This is the formal grounding for treating representation and inference as independent methodological degrees of freedom — the central thesis of this article series.

# 7. Invariants

## Domain-order invariance

A system's representation is indexed by domain name, not by sequential position: two representations built from the same observations are equal as elements of X regardless of the order in which the domains were registered in a Representation's constructor. This invariance does not automatically extend to the flattened vector consumed by geometries and algorithms — that operation requires an explicit order as an argument, and comparing two vectors built with different orders compares different coordinates under the same index. The implementation avoids this failure by requiring the order to be propagated consistently through every downstream call, verified by a dedicated test — not by any property of X that makes the failure structurally impossible.

## Scale re-parametrization equivariance — confirmed empirically

The completeness-weighted z-score is equivariant, not invariant, under an affine change of the underlying unit (for example, kilograms to pounds): re-fitting the reference statistics in the new unit reproduces the identical z-score, because the z-score transformation commutes with an affine change of variable. This is a weaker and more precise claim than "invariance" — the representation does not ignore units; it is stable under the simultaneous change of the reported unit and the re-fitting of the reference distribution in that unit.

This property was CONFIRMED by a dedicated regression test, not merely deduced from the algebraic form of the transformation: identical to floating-point precision (difference < 10⁻⁹) across the base case, partial missingness in the reference population, sign-inverted domains, and ten independent random scale factors.

## Temporal discretization — tested, and the invariance does not hold

Unlike the two previous properties, this one was tested and refuted, not confirmed. We subsampled real multi-encounter trajectories from the UCI hospital cohort (keeping only even-indexed encounters per patient, which exactly doubles the time interval between the remaining points) and re-fit the Theorem 1 evolution operator. The per-day contraction rate φ is not invariant to this change: it increases systematically under subsampling for nearly every characteristic tested — for instance, length of stay goes from φ ≈ 0.21 at daily resolution to φ ≈ 0.43 at the coarser resolution, nearly doubling, not a rounding-level discrepancy. For characteristics already close to the instability threshold, the fitted mean μ diverges far more severely, consistent with the well-documented near-unit-root estimation problem in the time-series literature.

The practical conclusion is stated plainly: φ, as currently estimated, should not be treated as a discretization-independent physical rate constant — comparing φ values fitted at different observation frequencies is not, today, a valid operation in this framework.

# 8. Categorical Organization

The operator algebra of Section 5 can be organized as a category C: objects are representation spaces X_d, one per data source or physiological scope (for example, NHANES's cross-sectional metabolic representation and UCI's longitudinal hospital-utilization representation — two structurally very different sources, not two variations of the same thing); morphisms are admissible operators T: X_d → X_d within a single object; identity is I_Xd; composition is as defined in Section 5, associative by inheritance from function composition.

Since every category defined here has exactly one object, a functor between two such categories is precisely a monoid homomorphism — a map φ from the admissible operators of one category to those of the other, preserving identity and composition. We report two concrete candidates, not just one, each tested against real data, with scope stated precisely in each case:

The first candidate: the evolution operator (Theorem 1) satisfies the one-parameter semigroup law T_{s+t} = T_s∘T_t — verified algebraically by direct substitution, then tested numerically against genuine real longitudinal data: 16,773 real multi-encounter patients from the UCI hospital cohort. The law holds exactly, to floating-point precision, for the Features with a fitted dynamic. This validates the law on one real object of the category — not yet, by itself, a demonstration of a functor between two different categories, since that requires a second, equally longitudinal source to test whether the same law holds identically on another object; the cross-sectional source (NHANES) used in the second candidate below has no per-patient longitudinal structure and cannot be used for this specific test.

The second candidate, more direct, and the one that actually realizes the mapping BETWEEN categories: a projection function takes the representation of any data source to a dictionary {process_name: aggregated value}, indexed by shared physiological process — not by source-specific domain. Real demonstration: a NHANES patient and a UCI patient, projected onto the same shared process (glucose homeostasis, present in both representations under the same label), despite zero raw variables in common between the two representations. We report a real limitation encountered while building this example, not hidden away: direct numeric comparability between the two projected values depends on the source Features being under the same scale convention — which holds for NHANES (completeness-weighted z-score) but not, today, for UCI's glycemic-testing domain, which uses a raw ordinal category (0, 1, or 2) by design, not a z-score. The functorial mapping — the same named process, aggregating Features from structurally distinct representations into a shared target — is genuinely demonstrated; direct numeric comparability of the aggregated value, in this specific instance, is not, and we register this distinction precisely rather than erasing it.

The two candidates are complementary, not competing: the first proves an algebraic law about operators, validated on a real object; the second constructs a concrete object-level mapping BETWEEN two structurally very different data sources, toward a shared target. What we still do not have — named here as future work, not claimed as a result — is the first candidate's semigroup law demonstrated as a genuine functor between two categories (which would require a second, real, longitudinal source beyond UCI), and a systematic check that the identity and composition laws hold for any pair of sources, not only the one tested.

# 9. Formal Contracts

The previous sections defined the meta-model's structure. Defining entities, however, is not sufficient to characterize a formal theory — it is necessary to specify which properties every implementation must satisfy. Formally, each contract is a predicate over the representation space,

**Cᵢ : X → {0, 1}**

where Cᵢ(x) = 1 if representation x satisfies property i. The scientifically admissible subspace is

**X\* = {x ∈ X : Cᵢ(x) = 1 ∀ Cᵢ ∈ K}**

Only representations belonging to X\* — not X in general — are eligible for the inference operators; a representation that fails even one contract is, by this definition, outside the theory's admissible domain, regardless of whether an implementation eventually produces an output for it.

Eleven contracts compose K today: Traceability, Semantic Preservation, Compositionality, Continuity, Extensibility, Algorithmic Independence, Temporality, Reproducibility, Versionability, Interoperability, and an eleventh — Population Injectivity — which did not receive its own number in the original convention. All eleven now have a dedicated empirical test, not merely a formal declaration; the last four to gain a test — Compositionality, Algorithmic Independence, Versionability, and Interoperability — were closed in a single dedicated round, each with a positive case and a decisive negative case (for example, an algorithm with a hard-coded dependency on a specific domain correctly fails the Algorithmic Independence contract when applied to a representation lacking that domain, rather than the test merely confirming success on well-behaved inputs).

# 10. Discussion and Conclusion

This article established the formal apparatus of the BioSpace framework: a meta-model of nine entities, five axioms — four of them reformulations of already-tested contracts, one whose real violation was caught by a fixed implementation bug —, a metric structure with an honest accounting of which geometries satisfy that condition, an operator algebra characterized precisely as a monoid, theorems connecting the framework to classical results with scope stated without exaggeration, a catalog of invariants where two were confirmed empirically and one was tested and refuted, and a categorical organization demonstrated by two independent functor realizations.

The discipline running through every section is deliberate: every strong claim comes with the test that supports it, and every limitation found — the triangle inequality violated by DTW, the discontinuity at missing-data transitions, the temporal-discretization invariance that does not hold — is reported with the same weight as the positive results. A representation theory is only as trustworthy as the tests that could have refuted it.

This is the first article in a series. The second establishes geometry and representation as independent methodological axes in the detection of computational structure — validated, like this one, without relying on any single data source to carry the argument. The third demonstrates generalization across data sources with empirical validation on two independent public sources of Type 2 Diabetes Mellitus (NHANES and UCI Diabetes 130-US Hospitals). A fourth article positions this architecture relative to the existing normative-modeling literature. The foundation established here is what makes these later articles comparable to one another — the same theory, tested against genuinely different data sources and inference algorithms.
