---
title: "Geometry as an Independent Methodological Axis in the Detection of Computational Structure"
subtitle: "A Formal Decomposition of Representation, Geometry, and Inference, Tested on Two Independent Real Public Data Sources"
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

The lack of reproducibility of clustering-derived subtypes is one of the most persistent obstacles in computational phenotyping in medicine — documented in psychiatric disorders, chronic respiratory disease, sepsis, and autism spectrum disorder, among other conditions, reviewed by groups with no obvious disciplinary overlap. Part of the field has responded to this problem by moving away from discrete clustering toward continuous, model-based alternatives such as normative modeling. We do not dispute that diagnosis. We ask a narrower, complementary question: when a clustering-based pipeline *is* used, how much of the instability commonly attributed to "the algorithm" or "the disease" is actually attributable to the representation and geometry chosen upstream of the algorithm — a design decision rarely varied, and almost never reported, in the applied literature?

We formalize this question as a decomposition F = A(G(R(B))), separating the biological system (B), its representation (R), the geometry imposed on that representation (G), and the inference algorithm (A) as four independent methodological degrees of freedom. We test this decomposition directly on two structurally distinct real public data sources — NHANES (national metabolic health survey, adults ≥20 years) and UCI Diabetes 130-US Hospitals (administrative hospital admission and readmission records) — without relying on any additional data source to carry the central argument.

On NHANES, holding the representation fixed, the standard geometry (Euclidean distance over the feature vector, feeding K-Means) finds highly stable clusters (ARI=0.957 at K=2). An alternative geometric diagnostic over the SAME representation — structural curvature of the similarity graph — initially **does not corroborate** that structure (p=0.36, null), until the representation, not the geometry, is enriched with an additional physiologically motivated domain — at which point the same curvature comparison becomes significant (p=0.0012). On UCI, two distinct geometric operationalizations of "cohort of similar patients" — K-Means cluster membership versus a k-nearest-neighbor query to that same cluster's centroid, both over the SAME representation and the same underlying Euclidean metric — recover substantially different populations (Jaccard index≈0.11, ~20% overlap), even though both capture real hospital-readmission signal above the population baseline. In neither source did a relational geometry (a graph, explored by a graph neural network) produce a benefit over an equivalent point-wise representation — a negative finding, and a real tension with the common intuition that relational structure should always help.

We do not interpret these findings as evidence that clustering "works" as long as the right geometry is chosen — the point is narrower and, we think, more useful: representation and geometry are measurable, independent contributors to the structure detectable in a computational space, and comparisons that vary only the algorithm, holding representation and geometry fixed and implicit, are underspecified relative to the real space of methodological decisions at play.

# 1. Introduction

A growing body of work has documented, independently and across disciplines with no obvious overlap, that computational subtypes derived from clustering frequently fail to replicate: across different research groups, across resamplings of the same population, and across small pipeline variations. This convergence is, in itself, a finding — it suggests that whatever is driving the instability is not particular to any one disease's biology, but something closer to a shared methodological substrate underlying how "phenotype discovery" is typically performed.

One response to this state of affairs is to abandon discrete clustering entirely, replacing it with continuous, model-based deviation scores relative to a normative reference. This is a well-grounded response to a real problem, and nothing that follows is offered as competing with it.

We pursue a narrower and, we think, complementary question. Papers reporting unstable clusters, and papers reporting stable ones, typically vary the clustering *algorithm* (K-Means versus Gaussian mixtures versus hierarchical methods) while holding each patient's *representation* — which variables, under what normalization, under what notion of distance — essentially fixed, and almost never reported as a design decision. If representation and geometry are themselves independent sources of variance in the outcome, then comparing algorithms while holding representation implicit is not a controlled comparison — it is a comparison confounded by an unexamined variable.

We test this directly, on two real public data sources that are structurally very different from each other — which makes any finding that holds across both more defensible than a finding isolated to a single cohort: NHANES (a cross-sectional metabolic health survey) and the UCI Diabetes 130-US Hospitals dataset (longitudinal administrative hospitalization records, with no real calendar dates, only encounter order). Neither was chosen for convenience or because it already supported the argument — both were adopted in this project for independent reasons (validating generalization across diseases), and the geometry findings reported here emerged from the same infrastructure already used for other purposes, not from data collected specifically for this argument.

# 2. A Formal Decomposition

We propose representing a biological system B through an explicit function R: B → X, producing a representation space X, and we separate what is usually called "the clustering pipeline" into four independent components:

**F = A(G(R(B)))**

where R is the representation, G is the geometry (the notion of distance or similarity imposed on R(B)), A is the inference algorithm, and F is the resulting phenotype assignment. This is not a new claim about biology; it is a claim about where methodological degrees of freedom live. R and G are typically treated as fixed background assumptions — a vector of z-scored clinical variables under Euclidean distance — rather than as hypotheses in their own right, comparable in status to the choice of A. Our claim is that R and G deserve the same explicit scrutiny, and the same systematic variation across candidate choices, that the clustering algorithm A conventionally receives.

This decomposition makes a specific, falsifiable prediction: if instability is substantially attributable to G, rather than solely to the underlying population structure or to A, then holding a single representation R fixed and varying only G should change the detectability of stable structure — even though the patients, the variables collected on them, and the available algorithm family are identical across conditions. Section 4 tests this prediction in two complementary ways: by varying the *geometric diagnostic* applied to the same representation (Section 4.1), and by varying the *geometric grouping mechanism* applied to the same representation and the same base metric (Section 4.2).

# 3. Methods

## 3.1 Data sources and base representations

**NHANES** (National Health and Nutrition Examination Survey, pre-pandemic cycle): 9,232 adults ≥20 years, represented by a vector organized into clinically motivated semantic domains (glycemic, anthropometric, cardiovascular, renal, lipid, comorbidity, treatment), each variable normalized by completeness-weighted z-score against the population reference. Cross-sectional — a single point per patient.

**UCI Diabetes 130-US Hospitals**: 71,518 patients, 101,766 administrative hospital encounters, represented by a 3-domain vector (hospital utilization, glycemic testing, medication intensity). 16,773 patients have two or more encounters, enabling longitudinal analysis; the source contains no real calendar dates, only encounter order.

In both sources, the representation R used in each geometry comparison is held identical across the compared conditions — no variable is added or removed when switching geometry within the same results section; where the representation is deliberately changed (Section 4.1), this is explicitly stated as the manipulated variable.

## 3.2 Geometries and geometric mechanisms compared

1. **Point-wise Euclidean + K-Means** — the default geometry: unweighted Euclidean distance over the feature vector, feeding cluster assignment via Voronoi partition among K simultaneous centroids.
2. **Structural curvature (Ollivier-Ricci)** — over a k-nearest-neighbor graph built on the same vector representation, comparing edge curvature within the same phenotype against edges crossing the boundary between phenotypes.
3. **Geometric proximity query (k-nearest-neighbors to a query point)** — the same Euclidean distance underlying K-Means, but applied as a query to a SINGLE reference point (the centroid of an already-identified cluster), not as a simultaneous partition among multiple centroids.
4. **Relational geometry (patient similarity graph)** — the same base vector space, converted into a graph whose nodes are patients and whose edges connect k-nearest-neighbors, explored by a semi-supervised graph neural network.

## 3.3 Evaluation metrics

**Cluster stability**: Adjusted Rand Index (ARI) between partitions obtained on independent random halves of the cohort, conventional stability threshold ARI ≥ 0.7.

**Structural curvature**: Ollivier-Ricci curvature per edge of the similarity graph, compared between edges within the same phenotype and edges crossing the phenotype boundary (Mann-Whitney test).

**Overlap between geometric grouping mechanisms**: Jaccard index and asymmetric overlap fractions between the membership set of a K-Means cluster and the membership set of a k-nearest-neighbor query to that same cluster's centroid, with identical k between the two definitions.

**Semi-supervised classification performance**: accuracy of a graph neural network (Graph Convolutional Network) trained on the relational representation, compared against an architecturally identical network with message passing disabled, across a range of label-availability-fraction conditions.

All methods were validated against synthetic data with known ground truth before any analytic application — for example, the Ollivier-Ricci curvature implementation was checked against textbook results (exactly zero curvature on a cycle graph, strongly positive curvature on a complete graph) before any analytic use.

# 4. Results

## 4.1 NHANES: the standard geometry finds stable structure, but an alternative geometric diagnostic initially does not corroborate it

Under the standard geometry (Euclidean + K-Means), NHANES's metabolic representation produces highly stable clusters: ARI=0.957 at K=2, remaining above the conventional stability threshold through K=7. By this criterion, the standard geometry "works" — the opposite of what is usually reported in the phenotype-reproducibility literature.

We applied, over the SAME representation and the same sample, a second geometric diagnostic: structural curvature of the similarity graph at phenotype boundaries. If the structure found by K-Means were robust in any broader geometric sense, we would expect edges crossing the phenotype boundary to show systematically different curvature from edges within the same phenotype. The initial test found no such difference (p=0.36, null, K=2, sample of 1,500) — a result that, taken in isolation, would suggest that the boundary found by K-Means, despite being statistically stable under resampling, corresponds to no genuine structural tension detectable by a different geometric criterion.

We did not treat this null as final. Enriching the representation with an additional physiologically motivated domain — full lipid profile (total cholesterol, HDL, triglycerides) — and repeating the identical curvature comparison, over the same population and the same graph geometry, the finding reversed: **the same boundary, now with a richer representation, shows significant curvature discrimination (p=0.0012)**. Neither the geometry nor the population changed between the two tests — only R.

**Table 1.** Structure detected in NHANES, by geometric diagnostic and representation.

| Condition | Geometric diagnostic | Result |
|---|---|---|
| 6-domain representation | Cluster stability (ARI, K=2) | Stable (0.957) |
| 6-domain representation | Boundary curvature | Not significant (p=0.36) |
| 7-domain representation (+ lipid) | Boundary curvature | Significant (p=0.0012) |

We interpret this as a direct instance, not anticipated when this article's central argument was formulated, of the decomposition in Section 2 itself: "detectable structure" is not a fixed property of either geometry or representation alone — it is a joint property of the two, and changing either one, holding the other fixed, can reverse the conclusion.

## 4.2 UCI: two geometric operationalizations of "cohort," the same representation, substantially different populations

A high-risk phenotype for early hospital readmission was identified by K-Means (K=4) over UCI's canonical 3-domain representation — 6,091 patients, with a 30-day readmission rate of 8.75% (versus 4.51% in the general population, ~1.9× baseline).

We built a second geometric operationalization of "patients similar to this group," over the SAME representation and the same Euclidean metric underlying K-Means: a k-nearest-neighbor query to that same phenotype's centroid — not to a real patient, but to an arbitrary query point in the representation space — with k equal to the original phenotype's size. If the two operationalizations were capturing essentially the same geometric notion of "proximity," we would expect high overlap between the two patient sets.

The observed overlap was low: Jaccard index≈0.11, about 20% overlap between the two sets, despite identical size and identical base representation and metric. This has a direct mathematical explanation, not an artifact: K-Means partitions the space by Voronoi cells among the K centroids SIMULTANEOUSLY — a point belongs to the cluster of the nearest centroid, among all K, not only to the reference centroid —, while the k-nearest-neighbor query considers only distance to ONE reference point, ignoring the other centroids. These are two genuinely different geometric mechanisms for defining "similar group," not two approximations of the same thing, even when operating over identical representation and metric.

**Table 2.** Two geometric operationalizations of the same high-risk phenotype, UCI.

| Cohort definition | Geometric mechanism | Size | 30-day readmission rate |
|---|---|---|---|
| Original K-Means cluster | Voronoi partition among 4 centroids | 6,091 | 8.75% (~1.9× baseline) |
| k-nearest-neighbor query | Distance to a single reference point (the centroid) | 6,091 | 6.70% (~1.5× baseline) |
| — | Overlap between the two | Jaccard≈0.11 (~20%) | — |

Both operationalizations capture real signal above the population baseline — neither is spurious —, but they define substantially different populations. The methodological point is that "using a geometry" is not a binary choice: within what would generically be called a "geometric approach," distinct grouping mechanisms produce distinct results on identical data.

## 4.3 Relational geometry (graph): no benefit detected in either source

We tested whether exploiting relational geometry — the patient similarity graph, via a semi-supervised graph neural network — improves classification relative to a purely point-wise representation (the same network, with graph message passing disabled), in both sources, across a range of available-label fractions from 1.5% to 50%.

In neither source did the graph produce a consistent benefit. In NHANES, the relational geometry hurts or does not help across nearly the entire range tested; a possible positive signal appears only at an extreme fraction (~1.5% labeled, n=22), with too small a margin (+0.9 percentage points) to be reported as conclusive rather than as a direction worth dedicated follow-up. In UCI, the graph does not help at any fraction tested, with no crossover detected.

This is a negative finding, and it is worth recording with the same weight as the positive findings of Sections 4.1 and 4.2: the common intuition that relational structure should always add value over a point-wise representation did not hold in either source tested here. A plausible interpretation, consistent with the findings of the previous sections: in both sources, the point-wise representation under standard geometry already resolves substantial structure on its own (Section 4.1: NHANES has very stable clusters; Section 4.2: UCI has phenotypes with clear readmission signal) — leaving relatively little unresolved structure for a relational geometry to recover.

# 5. Discussion

## 5.1 Relation to the reproducibility critique

We read our results as consistent with, and offering a partial mechanistic account for, the reproducibility problem already established across multiple diseases and multiple research groups — but with a nuance the reproducibility literature typically does not emphasize: geometry is not a one-dimensional axis of "better" or "worse." NHANES shows stable clustering under the standard geometry, but an alternative geometric diagnostic initially disagrees with that conclusion on the SAME representation — and the disagreement only resolves by varying the representation, not the diagnostic geometry. UCI shows that even within "geometric approaches," distinct grouping mechanisms (multi-centroid partition versus single-point query) produce substantially different populations over identical representation. Neither finding would be visible in a comparison that varies only the clustering algorithm, holding representation and geometry fixed and unexamined.

## 5.2 Relation to normative modeling

Normative modeling responds to the reproducibility problem by changing what the output of phenotyping *is* — replacing discrete cluster membership with a continuous deviation score against a reference distribution, sidestepping the requirement that stable discrete boundaries exist at all. Our approach responds to a narrower version of the same problem without changing the output type: we ask whether, *given* that discrete clustering is being used, the representation and geometry feeding into it receive the same scrutiny as the algorithm itself. These are not competing answers to the same question, but answers to two different questions — "should we be clustering at all" and "if we are clustering, what upstream choices does the result depend on" — and we would expect a well-specified account of computational phenotyping to eventually need both.

## 5.3 On the risk of "cherry-picked geometry"

If representation and geometry can be chosen to reveal structure that a default choice conceals, the same flexibility could in principle be used to manufacture apparent structure that does not exist, simply by trying representations and geometries until one produces a stable result. The four geometric mechanisms tested here were chosen for stating a distinct, nameable structural hypothesis — stability under multi-centroid partition, structural tension at a boundary, proximity to a single reference point, relational adjacency — not by searching among many candidates until one produced the desired result. We report the geometries and representations that revealed structure (stable K-Means on NHANES; both geometric cohorts on UCI with real signal) alongside those that did not, or revealed structure only partially (null curvature until R was enriched; the relational graph in neither source), with the explicit purpose of making this point verifiable, not merely asserted.

## 5.4 What would falsify this position

It is worth explicitly stating what evidence would count against the claim made here, since the claim is only useful if it is falsifiable. If, across more data sources and more diseases, alternative geometric diagnostics applied to the same representation always agreed with each other, and different geometric grouping mechanisms always recovered the same population — that is, if varying G (holding R fixed) or varying the specific geometric mechanism (holding R and the base metric fixed) produced no detectable difference at all — that would be evidence that geometry is not, in fact, doing independent work, and that the instability documented in the literature has other, geometry-independent causes. We would consider that a more important result than the one reported here, and we would want to know about it.

# 6. Limitations

1. **Two data sources, no more.** Although structurally very different from each other (cross-sectional versus longitudinal administrative; population metabolic health versus hospital utilization), both come from United States healthcare contexts; generalization to other healthcare systems or populations was not tested.
2. **The set of geometric mechanisms tested does not exhaust the space of possible geometries** over the same representation — other choices may reveal additional structure not captured here.
3. **None of the comparisons was validated against independent hard clinical outcomes** (cardiovascular events, mortality) beyond the hospital readmission already used as an outcome in Section 4.2; the NHANES curvature comparison (Section 4.1) is entirely about internal statistical structure, with no associated clinical outcome.
4. **The representation enrichment that reversed the NHANES curvature finding (Section 4.1) was a post-hoc decision**, motivated by the initial null result, not pre-registered. We report this transparently because the alternative — discarding the null without investigating, or not reporting the reversal — would be less honest, but we recognize that a pre-registered replication of the representation enrichment would strengthen the claim.
5. **The graph neural network tested in Section 4.3 is one architecture among many possible ones**; we do not rule out that a different relational architecture could find benefit where this one did not.

# 7. Conclusion

We tested, on two structurally very different real public data sources, whether representation and geometry function as genuinely independent methodological degrees of freedom in the detection of computational structure — not only in principle, but with measurable practical consequence. On NHANES, the same representation and the same population produce opposite conclusions under two different geometric diagnostics (cluster stability versus structural curvature), until the representation, not the geometry, is changed. On UCI, the same representation and the same base Euclidean metric produce substantially different populations under two distinct geometric cohort-definition mechanisms (multi-centroid partition versus single-point proximity query), both capturing real signal, but not the same signal. In neither source did a relational geometry produce the benefit that common intuition would attribute to it.

We think the honest summary of these results is not "geometry matters" in a generic sense, but the more precise and more useful claim the decomposition in Section 2 was built to make from the start: representation and geometry are separate, independently manipulable degrees of freedom, and the detectability of structure in a computational phenotype is a joint property of both — tested here without relying on any single data source to carry the argument, and available for replication on any third source that satisfies the same standard of methodological transparency.
