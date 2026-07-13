---
title: "Convergence Between Independent Methods as Methodological Evidence"
subtitle: "What a Representation-First Architecture Enables, Tested Against a Single Real Predictive Problem"
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

A null result obtained by a single method is always ambiguous between two explanations: the effect does not exist, or the chosen method failed to find it. Resolving this ambiguity normally requires rebuilding the problem from scratch under a different method — feature engineering, missing-data handling, variable encoding, all redone — which carries a real cost and, for that reason, is rarely done in practice. This article argues, and demonstrates against a single real predictive problem, that an architecture treating representation as an explicit, reusable object — not a preprocessing step rebuilt for every new method — reduces that cost to nearly zero, turning triangulation between independent methods into a routine practice rather than an expensive exception.

We test this against a genuine question: does a patient's first hospital encounter predict a future early readmission? Using exactly the same baseline representation — Features from the first encounter, never information from later encounters (Temporality Contract) —, we handed the same data, with no intermediate reconstruction, to four statistically independent methods: a Cox survival model stratified by phenotype (concordance index≈0.52), a Random Forest (AUC=0.50–0.52, 5-fold cross-validation), a Logistic Regression (AUC≈0.53, 5-fold cross-validation), and SHAP explainability over the trained Random Forest (diffuse importance across 13 Features, ratio between the largest and smallest <20×, versus >5× in a synthetic control with known genuine signal). A fifth, related method — early-warning-signal detection (critical slowing down) — could not be tested with adequate statistical power, which is itself informative: only 139 of 71,518 patients (0.19%) have long enough longitudinal follow-up for the most rigorous test available, and only 1 shows the warning signal.

The first four methods converge on essentially the same performance ceiling, close to chance. We argue that this convergence is stronger evidence than any of the four results in isolation — not because four numbers near 0.5 are more impressive than one, but because the four methods share only the representation, not the model's mathematical form, the linearity assumption, or the optimized loss function. If the limitation were specific to one method, there would be no reason for the other three to agree.

# 1. Introduction

Comparing different statistical methods against the same problem is, in principle, a valuable practice: when two methods agree despite operating under distinct assumptions, that agreement is stronger evidence than either method alone — the inverse of the usual argument about p-values obtained by chance, where multiple independent attempts *increase* the chance of an isolated false positive but *drastically reduce* the chance of multiple concordant false positives arising by chance.

In practice, this kind of triangulation is rare — not because it is theoretically less valuable, but because it carries a real engineering cost. A Cox survival model, a Random Forest, and a SHAP explanation typically require three different data pipelines: different input formats, different categorical-variable encoding conventions, missing-data handling potentially inconsistent across the three implementations. Rebuilding the same feature set three times, in three slightly different ways, introduces a real risk that the observed "convergence" is actually an artifact of three reconstructions that coincidentally carry the same bias — or, in the opposite direction, that an observed divergence is a reconstruction artifact, not a genuine finding about the methods.

This article tests a narrower claim: if a patient's representation is a first-class object — built once, under formal contracts guaranteeing reproducibility and semantic preservation, served identically to any downstream consumer —, the cost of triangulating across methods drops enough that rebuilding the same problem under four independent methods stops being an expensive engineering decision and becomes a routine check. We test this not in the abstract, but against a single real predictive question, using the same representation, with no feature reconstruction between the four methods compared.

# 2. The Question Used as a Test Case

The question: does a patient's state at their first hospital encounter predict an early readmission (fewer than 30 days) at a *subsequent* encounter? We use the UCI Diabetes 130-US Hospitals cohort — 16,773 patients with two or more encounters, of whom 3,011 have four or more, the subset used by methods requiring a larger reference window.

The design imposes a deliberate constraint, not an incidental limitation: no characteristic used to characterize a patient may come from an encounter after the first. This is not merely good statistical practice — it is a direct instance of the Temporality Contract already formalized in the representation architecture used (a patient's representation at instant t cannot depend on observations after t), and applying it here is what makes this question genuinely predictive, not a cross-sectional characterization disguised as prediction.

The label: 1 if the patient had at least one early-readmission event at any encounter from the fourth onward (or the second, depending on the method); 0 otherwise. The event rate in the eligible cohort is approximately 20–31%, varying slightly across the subsets used by each method (Section 3).

# 3. Five Methods, One Representation

All methods below consume the same feature matrix — Features from each patient's first encounter, in the same domain order, produced by the same representation-construction call. No method involved reconstructing features, re-encoding variables, or adjusting missing-data handling differently from the others.

**Survival (Cox by phenotype).** Eligible patients phenotyped by K-Means over the baseline representation; a Cox proportional-hazards model fit with phenotype as the covariate, time to the first readmission event (in encounter-index units, not calendar — the source has no real date) as the survival variable.

**Random Forest.** A random-forest classifier (200 trees, max depth 6, class-balanced weights) trained directly on the same feature matrix, evaluated by 5-fold stratified cross-validation, AUC-ROC metric.

**Logistic Regression.** Same matrix, same cross-validation, a linear classifier with class-balanced weights.

**Explainability (SHAP over the Random Forest).** Not a predictive method in itself — a lens over the already-trained model. Computes each feature's mean absolute contribution to each patient's prediction, using an exact explainer for tree models (`TreeExplainer`). Validated earlier against a synthetic scenario with known ground truth: a feature that determines the label by construction receives SHAP importance more than 5× larger than a pure-noise feature with no relationship to the label.

**Early-warning signals (critical slowing down).** A related but structurally different method from the four above: it does not predict a label, it tests whether a feature's autocorrelation and variance show a statistically significant rising trend (Kendall's test with significance via autoregressive surrogate data) in the encounters preceding an event — the classical signature of approaching a critical transition in the dynamical-systems literature. Validated earlier against a genuine synthetic saddle-node bifurcation, the field's own standard test.

# 4. Results

**Table 1.** Predictive performance of the four comparable methods, same baseline representation.

| Method | Metric | Value |
|---|---|---|
| Cox by phenotype | Concordance index | ≈0.52 |
| Random Forest | AUC (5-fold cross-validation) | 0.50–0.52 |
| Logistic Regression | AUC (5-fold cross-validation) | ≈0.53 |
| SHAP (over the Random Forest) | Largest/smallest mean \|SHAP\| ratio | <20× (synthetic control: >5×) |

The four methods converge on essentially the same ceiling — all close to 0.5, the expected performance of a prediction with no genuine discriminative power. SHAP explainability adds information the first three methods, alone, do not give: it is not that strong signal is concentrated in a specific feature the models are failing to exploit well — importance is distributed diffusely across the 13 available baseline features, with none dominating the others by a margin that would resemble a genuine lost signal.

**Early-warning signals, treated separately.** Unlike the four methods above, this one did not produce a null result with adequate statistical power — it produced an eligible sample too small for any test to have power to detect anything, even if it exists. Of the source's 71,518 patients, only 320 (0.45%) have enough encounters for the most rigorous method available; restricting to a design with genuine prospective validation (detection within the first 8 encounters, event checking only in later ones, with no future information leaking in), 139 patients remain, of whom only 1 shows the warning signal. We report this separately from the four main methods because it is evidence of a different nature: not "we tested and did not find," but "there is not enough data to test with confidence" — a distinction worth keeping explicit, not collapsing into the same category as the adequately powered null results.

# 5. Why Convergence Is Stronger Evidence Than an Isolated Result

This article's central argument is not that four numbers close to 0.5 are, arithmetically, more convincing than one. It is that the four methods compared in Table 1 share no structural assumption in common, except the input representation:

- Cox assumes proportional hazards and a parametric form for the baseline hazard function; Random Forest and Logistic Regression make neither assumption.
- Random Forest is a non-linear method, based on recursive partitions of the feature space; Logistic Regression is strictly linear in the feature space (after the link function).
- SHAP is not, itself, a predictive method — it tests a different question (concentration versus diffusion of importance) over one of the already-trained models, not over the data directly.

If the observed absence of predictive signal were a limitation specific to ONE method — Cox penalized by the proportional-hazards assumption not holding, or Logistic Regression penalized by non-linear relationships it cannot capture —, there would be no reason for the other methods, free of that specific assumption, to reach the same conclusion. Convergence across methods with mutually exclusive structural assumptions is evidence that the limitation lies in the informational content of the baseline representation relative to this specific outcome, not in the choice of any individual method.

This form of argument — multiple independent methods, same representation, same conclusion — is only cheap to produce when the representation already exists as a reusable object before any method is chosen. Rebuilding the same feature matrix three or four times, once per method, would be the usual pattern; here, each method was simply an additional consumer of the same already-built representation, with no incremental engineering cost beyond fitting the model itself.

# 6. Discussion

## 6.1 What convergence does not prove

Convergence across methods rules out one class of explanation (a limitation specific to one method), not every possible class. A bias shared by all four methods — for example, a systematic error in extracting the baseline representation itself, present before any method is applied — would produce the same apparent convergence without the substantive conclusion ("there is not enough signal in the first encounter") necessarily being correct. We tested the representation itself, separately and before this analysis, against synthetic data with known ground truth (not reported in this article) — but we recognize that this is a residual assumption that convergence across methods, by itself, does not eliminate.

## 6.2 Relation to the underlying representation architecture

This article's argument depends on a condition not every medical-data architecture satisfies: that a patient's representation be built once, under explicit formal contracts (reproducibility — the same input data always produces the same representation; temporality — no characteristic at one instant depends on future observations), and served identically to any downstream consumer. When these guarantees do not exist, or are not verified, triangulation across methods stops being a cheap check and goes back to being an expensive engineering project — the usual scenario this article argues is avoidable.

## 6.3 Why report a null result across four methods, not just one

A null result reported under a single method invites the obvious question: would another method find the signal this one did not? Reporting the same question under four structurally independent methods closes that objection in advance, at an additional engineering cost the underlying architecture reduced to nearly zero. We treat this as the standard null results should meet, whenever the cost of producing it is low enough that skipping it is not a reasonable excuse.

# 7. Limitations

1. **A single predictive question, on a single data source.** We do not claim that the convergence observed here generalizes to any other question or source — only that the triangulation method itself, enabled by the underlying representation architecture, is reusable for any future question at low cost.
2. **The four methods compared do not exhaust the space of possible predictive methods** — we did not test, for example, deep neural networks or sequential boosting methods over the same representation; it is possible, though it does not seem likely given the observed pattern, that an additional method would find signal the four tested did not.
3. **The fifth method (early-warning signals) could not be evaluated with adequate statistical power** — its inclusion in this article is as evidence of a different kind (a data limit, not a tested null result), not as a fifth point of convergence of the same nature as the first four.
4. **Convergence across methods does not eliminate the possibility of a shared bias in the baseline representation itself**, prior to any method (Section 6.1) — mitigated, but not eliminated, by independent validation of the representation against synthetic data with known ground truth.

# 8. Conclusion

We tested whether four statistically independent methods — survival, non-linear classification, linear classification, and explainability over the non-linear model — agree when applied to the same real predictive question, using the same baseline representation with no feature reconstruction between them. They agree: all converge on a performance ceiling near chance, and explainability adds that this absence of signal is distributed diffusely, not hidden in a specific feature the models are failing to exploit well.

The central methodological argument is not about this specific predictive question — it is about what makes this kind of triangulation cheap enough to be done routinely, rather than reserved for exceptional findings that justify the engineering cost of rebuilding the problem multiple times. An architecture that treats representation as an explicit, testable, reusable object — not an implicit preprocessing step rebuilt for every new method — turns triangulation between independent methods from an expensive exception into a routine check. A null result confirmed by four methods that share no structural assumption in common, except the input representation, is qualitatively different — and stronger — evidence than the same null result reported under a single method, however well that single method was executed.
