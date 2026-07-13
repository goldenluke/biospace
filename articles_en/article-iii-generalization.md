---
title: "Generalization Across Structurally Independent Data Sources: Type 2 Diabetes Mellitus as a Clinical Interpretation Over a Generic Metabolic Representation"
subtitle: "Architecture, Synthetic Validation, and Validation on Two Structurally Incompatible Real Data Sources"
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

"Disease-agnostic" — the claim that a computational representation architecture does not depend on any specific disease — is easy to assert and hard to prove. This article documents how Type 2 Diabetes Mellitus was modeled within the BioSpace framework not as its own representation package, but as a **clinical interpretation** applied over a generic representation of the endocrine-metabolic system: no domain, no normalized variable, and no data structure in this work "knows" what diabetes is — only pure functions read an already-built vector and return a clinical label. We report validation of this architecture across three stages of increasing evidentiary strength, culminating in two real public data sources that are **structurally incompatible with each other**: NHANES (a cross-sectional metabolic health survey, N=9,232 adults, continuous physical-exam and laboratory variables) and the UCI Diabetes 130-US Hospitals dataset (N=101,766 administrative encounters, 71,518 patients, with none of NHANES's continuous variables, and age recorded in 10-year bins).

On NHANES, laboratory-criterion diabetes classification (the American Diabetes Association's glycemic criterion) reached 75.0% sensitivity and 95.0% specificity against self-reported diagnosis (n=7,848 comparable) — clinically plausible, reflecting real, well-documented underdiagnosis — and coherence among variables of the same physiological process was confirmed statistically (p=0.0022), unlike a synthetic generator used in an earlier validation stage, which did not reproduce that property. On UCI — a source with none of NHANES's continuous variables, which required a genuinely different representation, not an adaptation of the same one — a phenotype of elevated prior hospital utilization, without using age or diagnosis code, showed nearly double the 30-day readmission rate.

The most important finding of this article, however, is neither of these two on its own — it is what happened when we tested whether that readmission finding was robust to the representation used to find it. Adding a fourth domain to UCI's representation (ICD-9 diagnostic categories, the same grouping used in the original article that introduced this dataset) and re-running the same phenotyping over the same population, the structure found changed: instead of isolating prior hospital utilization, the new representation isolates a small group with extremely long hospital stays, and the association with readmission weakens substantially (from ~2.2× to 1.49×). Neither representation is wrong — they capture different structures in the same real data. This is not a theoretical argument: it is a live demonstration, within a single data source, that representation is not neutral — the same central thesis that motivates treating representation as an object of scientific scrutiny independent of the inference algorithm.

# 1. Introduction

The usual practice of computational disease modeling starts from the disease: "diabetes" is chosen, the variables that are "of diabetes" are decided, and a representation is built around them. This work argues, and implements, the opposite: the starting point is the biological system — in this case, the endocrine-metabolic system — and diabetes emerges as one of several possible clinical interpretations over the same representation, not as modeling's fundamental unit.

The motivation is not merely theoretical. A representation package built around "diabetes" tends to incorporate, implicitly and unexamined, assumptions about what counts as a relevant variable, which geometry to use, and how to interpret missing data — decisions that should be explicit and, ideally, reusable for other conditions that share the same physiological substrate (metabolic syndrome, chronic kidney disease, obesity). Separating representation from clinical interpretation is what makes that reuse possible — and testable.

"Disease-agnostic," however, is a claim that is proved or refuted, not a property that is declared. The most direct proof available is not to show that the architecture *could*, in principle, work across more than one disease — it is to apply it to two real data sources that differ from each other enough that a representation built for one would be unusable on the other without genuine redesign, and to show that the same theory, the same formal contracts, and the same vocabulary hold up under both. That is this article's central test.

# 2. Modeling

## 2.1 From the "diabetes" package to the "metabolic" package

The project began with a `plugins/diabetes/` package, containing domains, biological system, and representation named around diabetes. We refactored this into `plugins/metabolic/` — a package that represents the endocrine-metabolic system with no reference to diabetes whatsoever — and reduced `plugins/diabetes/` to a thin layer: it re-exports the generic representation under the old names (`DiabetesSystem is MetabolicSystem`, verified by object identity, not merely behavioral equality) and contributes only what is genuinely diabetes-specific — the clinical criterion and a synthetic generator that simulates a diabetes scenario, not a representation of diabetes.

## 2.2 Representation: seven physiological domains

`MetabolicRepresentation` composes seven semantic domains, each named by physiological meaning, not by diagnostic category: `GlycemicDomain` (fasting glucose, HbA1c), `AnthropometricDomain` (age, sex, BMI, waist circumference), `CardiovascularDomain` (blood pressure, heart rate), `RenalDomain` (creatinine, eGFR via CKD-EPI 2021), `LipidDomain` (total cholesterol, HDL, triglycerides), `ComorbidityDomain`, and `TreatmentDomain`. Each variable is normalized by completeness-weighted z-score against a population reference — missing data never discards a patient, only reduces the weight of that coordinate.

## 2.3 Clinical interpretation as a pure function

The central distinction: diabetes is not a property of the representation R, it is a transformation applied over it.

**B → R(B) → N(R(B))**

`classify_diabetes_status(vector)` reads the raw HbA1c and fasting-glucose values already present in `GlycemicDomain` — it introduces no new variable — and applies the ADA glycemic criterion (HbA1c ≥ 6.5% or glucose ≥ 126 mg/dL → diabetes; HbA1c ≥ 5.7% or glucose ≥ 100 mg/dL → pre-diabetes; absence of both measurements → indeterminate, never normal by omission; otherwise, normal). `N` occupies the position of an inference algorithm in this architecture, never that of a representation — preserving that distinction is what allows a second interpretation, `classify_metabolic_syndrome_risk_full` (full NCEP ATP III criterion, sex-specific: central adiposity, triglycerides, HDL, blood pressure, glucose), to operate over exactly the same vector with no modification to the representation.

## 2.4 Physiological processes: an optional layer between observation and domain

Every `Observable` may optionally declare a `process` — the name of a biological mechanism it measures, independent of which domain consumes it (`HbA1cObservable.process = "glucose_homeostasis"`). This enables two queries: `Representation.processes()` (which processes the representation covers) and `Representation.features_by_process(vector)` (grouping Features from *different* domains that measure the same process).

The layer is strictly additive, tested within the package itself: of the three domains in the representation used on UCI (Section 5), only one (`GlycemicTestingDomain`) declares a process; the other two (`HospitalUtilizationDomain`, `MedicationIntensityDomain`) never do, and `Representation.processes()` works correctly in both cases, without requiring every domain to participate.

## 2.5 Derived variables: from observation to trajectory

`DerivedVariable` is an entity parallel to `SemanticDomain` — not a subclass, because `encode()` only sees a single instant, and a derived variable needs the entire trajectory. We implemented three: `HbA1cSlopeVariable` (linear trend, %/year), `HbA1cVariabilityVariable` (standard deviation of the series), and `GlycemicBurdenVariable` (sum of HbA1c excess above 7.0% across the trajectory — reusing exactly the mechanism already used in the synthetic generator for renal decline, not a new mechanism invented for this layer). Tested against trajectories with hand-computed results (precision of 1e-3) and against the relevant edge cases: a single-point trajectory returns `None` for slope/variability (never a fabricated value), but still computes burden, which makes sense with a single point.

# 3. Synthetic Validation

## 3.1 The generator and a real mechanism

The synthetic generator simulates three severity groups (controlled, moderate, decompensated) with longitudinal progression. One mechanism was deliberately built to be testable: renal decline (falling eGFR, rising creatinine) correlated with **cumulative glycemic exposure**, not just baseline severity — chronic hyperglycemia damages the kidneys, a real, not arbitrary, mechanism. Tested: positive correlation between mean HbA1c and eGFR decline (ρ≈0.21).

## 3.2 Confounding by indication, correctly detected

`check_baseline_balance` applied to insulin initiation in the synthetic generator found 13 of 16 variables imbalanced between those who start treatment and those who do not — the correct behavior: more severe patients are preferentially indicated for treatment, and the contract should detect this, not hide it.

## 3.3 Proof of genericity: two interpretations, expected result

We built four clinical profiles with known outcome — diabetes without metabolic syndrome, syndrome without diabetes, both, neither — and confirmed that `classify_diabetes_status` and `classify_metabolic_syndrome_risk_full`, operating over the same representation, disagree exactly where they should disagree. If they always agreed, metabolic syndrome would just be diabetes under another name, not a genuine second lens.

## 3.4 A real negative finding: the generator does not reproduce process coherence

`check_process_coherence` tests whether Features of the same physiological process correlate more strongly with each other, across the population, than Features of different processes — validated earlier on two synthetic scenarios with known ground truth (confirms when real correlation exists; does not confirm on pure noise). Applied to the synthetic diabetes generator: **coherence does not confirm** (|r| same-process = 0.42 vs. different-process = 0.43, p=0.65). Investigated: the generator draws each variable independently within the severity class (one `rng.normal()` call per key, with no shared latent factor) — the existing population correlation comes from the class label and treatment effects, not from a real physiological mechanism linking HbA1c and glucose beyond that. We report this as a real limitation of the generator, not hidden behind a passing test.

# 4. Real-Data Validation: NHANES

## 4.1 Two real bugs, found only because real data arrived

The tests using fabricated data mocked the entire file-reading step, testing only the merge logic — they never exercised real `pandas.read_sas`. Testing against the real `.XPT` files (Pre-pandemic cycle, August 2017–March 2020, `P_` prefix), two real problems appeared: (1) `pandas.read_sas` requires `format="xport"`, not `format="xpt"` as the original implementation assumed; (2) the blood-pressure file changed methodology (`P_BPXO`, oscillometric, variables `BPXOSY1`/`BPXODI1`), not `P_BPX`/`BPXSY1`/`BPXDI1` as in the isolated 2017–2018 cycle. A test was written specifically to close this coverage gap, exercising the real file reader, not only the merge logic.

**A third, more consequential bug, found in a later audit**: the helper function that extracts a Feature's raw value had a silent fallback problem — for a genuinely missing Feature, it returned 0.0 instead of `None`, making the "indeterminate" branch of the diabetes classification unreachable. In practice, 1,140 adults with neither HbA1c nor glucose on record were being classified as "normal" instead of correctly excluded — inflating the analysis denominator with uninformative cases. This bug affected a number already published (see Section 4.2); fixed, with a correction note reported transparently, not hidden.

## 4.2 Diabetes classification against real self-reported diagnosis

In 7,848 adults (age ≥20) with an HbA1c/glucose value and a valid response about prior diabetes diagnosis, comparing `classify_diabetes_status` (ADA criterion) against the response to "has a doctor ever told you that you have diabetes?": sensitivity 75.0%, specificity 95.0%, accuracy 91.8%. Clinically plausible — diabetes underdiagnosis is a real, documented phenomenon, not a classifier artifact. Among those classified as pre-diabetes by laboratory criterion (n=3,141), 91.5% had no self-reported diabetes diagnosis — also consistent with the literature, which describes pre-diabetes as widely underdiagnosed, and a figure unaffected by the bug described in Section 4.1.

## 4.3 Process coherence: confirmed on real data, unlike the synthetic

Applying `check_process_coherence` to the same real population: |r| same-process = 0.782 vs. different-process = 0.151, p=0.0022 — **confirmed**. The contrast with Section 3.4 is not an inconsistency in the contract; it is the validation that was missing. The same test, applied to two sources with genuinely different properties (a synthetic generator with no real shared mechanism; a real population with the physiological structure the generator failed to reproduce), gave two different, correct answers.

## 4.4 An unsurprising, but recorded, finding

The minimum age in the full sample is close to zero — NHANES samples the entire population, including children, by design. We filtered to ≥20 years before any clinical diabetes interpretation: applying an adult diagnostic criterion to an infant makes no clinical sense, and this filtering is documented explicitly in the code, not implicit.

# 5. Real-Data Validation: UCI Diabetes 130-US Hospitals

## 5.1 A genuinely different structure requires a genuinely different representation

We inspected the structure before mapping anything — a discipline that proved necessary: this source (101,766 encounters, 71,518 patients) **has no** continuous HbA1c or glucose (only categories — `A1Cresult`, 83.3% missing; `max_glu_serum`, 94.8% missing), no BMI, waist circumference, blood pressure, or creatinine. Age comes in 10-year bins. Forcing this source into the `MetabolicRepresentation` built for NHANES would have produced an artificially impoverished representation, or missing data disguised as real data.

We built, instead, a representation appropriate to what this source actually measures: `HospitalUtilizationDomain` (eight hospital-utilization variables, 100% complete — the exact opposite of NHANES's completeness pattern), `GlycemicTestingDomain` (A1Cresult/max_glu_serum recoded ordinally, sparse by design), and `MedicationIntensityDomain` (insulin status, medication change). The connection to NHANES did not disappear: `GlycemicTestingDomain` declares the same `process="glucose_homeostasis"`, formally recording that the two sources measure the same biological mechanism through instruments of very different granularity — one continuous and dense, the other ordinal and sparse by design.

## 5.2 An unforeseen opportunity: real trajectories

23% of patients (16,773 of 71,518) have multiple encounters — up to 39 at the observed maximum. `encounter_id` is not a real date (the source has no date), but it grows monotonically; we used this as a chronological-order proxy, not a real interval, explicitly documented wherever this assumption is used. This produced the project's first genuinely real (not synthetic, not cross-sectional) trajectory.

## 5.3 The strongest finding under the original representation: a phenotype with neither age nor diagnosis associates with readmission

K-Means phenotyping (K=4; only 3 non-empty clusters emerged, a finding recorded without being hidden) over hospital utilization, glycemic testing, and medication — without age, without diagnosis code — produced a phenotype (6,091 patients) with nearly double the 30-day readmission rate of the other two (8.75% vs. 3.97% and 4.64% — a ratio of approximately 2.2×). We characterized the group before reporting: it is not the phenotype with the most insulin or medication change (that is a different, intermediate-risk group) — it is the group with elevated **prior** hospital utilization (prior outpatient visits: 2.24 vs. 0.13–0.23; emergency visits: 0.74 vs. 0.07–0.15; prior inpatient stays: 1.85 vs. 0.30–0.48). This is consistent with the hospital-readmission prediction literature, in which prior utilization is among the strongest and most consistently replicated predictors.

## 5.4 The most important finding of this article: the same finding, under a different representation, weakens — and this is not a bug

Section 5.3 reports a real finding, but it is not absolute — it depends on the representation used to find it, and we tested that directly rather than merely hypothesizing it. We added a fourth domain to UCI's representation, `DiagnosisCategoryDomain`: 9 ICD-9 categories (diabetes, circulatory, respiratory, digestive, injury, musculoskeletal, genitourinary, neoplasms, other), the same grouping used by the article that originally publicly introduced this dataset (Strack et al., 2014), extracted from the `diag_1`/`diag_2`/`diag_3` fields.

With this fourth domain included, and K-Means phenotyping re-run over the same population, the same methodology, the same K — the structure found changes. Instead of organizing primarily by prior hospital utilization, the new representation isolates a small group (212 patients, 0.30% of the cohort) defined by extremely long stays (13.41 days on average, versus 4.22–4.57 for the remaining groups) — and the association between the highest-risk phenotype and 30-day readmission becomes substantially weaker: a ratio of 1.49×, not the roughly 2.2× above.

Neither representation is wrong. They capture different structures in the identical real data — one organizes patients primarily by prior utilization pattern, the other by a combination of utilization and diagnostic burden that ends up dominated by an extreme subgroup with prolonged hospitalization. This is the same central thesis motivating this entire work — separating representation from inference precisely because representation is not a neutral choice —, now in a real, unplanned instance, found by testing Section 5.3's own finding against a second, equally reasonable representation of the same source, within a single dataset, not across two different sources. Both representations remain available side by side in the code — neither was discarded in favor of the other, and which one to use depends on the question being asked, not on which gives the cleaner result.

# 6. Discussion

## 6.1 What the three "diabetes" have in common, and what they do not

Throughout this work, "diabetes" was never a single thing. In the synthetic generator, it is a progression scenario generated with deliberate mechanisms. In NHANES, it is a laboratory criterion applied to a representation of continuous physical-exam and laboratory variables. In UCI Diabetes 130-US Hospitals, it does not even appear directly in the representation used — what appears is hospital utilization and sparse glycemic testing, and the most clinically relevant phenotype that emerged did not even need the diabetes variable itself to be found. These three representations are not interchangeable, and should not be treated as such — each answers a different question, with the data it actually had available. The same theory — the same meta-model, the same formal contracts, the same vocabulary of domains/processes/trajectories — holds all three without modification.

## 6.2 Negative findings with the same weight as positive ones

We reported, with the same level of detail, that the synthetic generator does not reproduce process coherence, that `check_process_coherence` could not be meaningfully tested on UCI's original representation (insufficient declared process to form comparison pairs), that K=4 produced only 3 non-empty clusters, and that a published finding (Section 5.3) weakens substantially under an equally defensible alternative representation (Section 5.4). None of these is a "bad" result to be hidden — each is real information about the limits of what was built and tested.

## 6.3 Two bugs, one pattern

The real bugs found in the loaders (file format, blood-pressure variable name, and the silent missing-value fallback that affected a published number) share a common cause: earlier tests validated the transformation logic with fabricated data, never the real file reader nor the edge cases of total absence. This is not an isolated failure — it is a general risk pattern in data pipelines: testing the logic is not the same as testing against the real source, and the difference only shows up once the real source arrives.

## 6.4 On the risk of "cherry-picked representation"

If representation can be chosen to reveal or hide a finding (Section 5.4), the same flexibility could, in principle, be used to pick the representation that produces the desired result, reporting only that one. Our defense against this risk is not an abstract rule — it is the actual content of Section 5.4: we report both representations side by side, with their two different results, and both remain available in the code, not only the one that "worked better." A skeptical reader should expect, and should look for, the same kind of transparency in any claim of representation robustness — reporting what changes when the representation changes, not only the result under the representation that was published first.

# 7. Limitations

1. **None of the three sources used here was validated against a long-term clinical outcome** (mortality, cardiovascular events) — UCI Diabetes 130-US Hospitals comes closest (readmission), but it is a short-term outcome.
2. **`classify_metabolic_syndrome_risk_full` uses the five NCEP ATP III criteria**, but depends on data completeness (triglycerides and HDL, fasting subsample, ~30–42% completeness in NHANES) — results are conditioned on that availability, explicitly documented in the function's return, not hidden.
3. **The chronological-order proxy in UCI Diabetes 130-US Hospitals (`encounter_id`) is not a real date** — absolute time intervals between encounters are not reliable, only the relative order.
4. **`check_process_coherence` was not meaningfully tested on UCI's original representation** (Sections 5.1–5.3) — this would require declaring a process on more than one domain, which we did not do for lack of sufficient mechanistic confidence to justify the label.
5. **The comparison in Section 5.4 uses two representations, not a systematic sweep** of all reasonable representations of the same source — we do not know whether a third representation (for example, including age) would produce a third, still different structure, though we consider this likely given the pattern already observed.
6. **The three data sources used (synthetic, NHANES, UCI) differ in population, design, and granularity** — no direct comparison of effect magnitude across them is appropriate; the validation here concerns the architecture and the behavior of the formal contracts, not a single number reproducible across sources.

# 8. Conclusion

Type 2 Diabetes Mellitus, in this work, was never the starting point of the modeling — it was treated, at every validation stage, as a clinical interpretation applicable over physiological representations built independently of it. This architectural choice made it possible to test the same thesis repeatedly, under different conditions: a synthetic generator revealed a real limitation of the generator itself (absence of process coherence); real NHANES data confirmed the same property the synthetic generator lacked, produced a diabetes classification with clinically plausible sensitivity and specificity, and exposed a real bug that had affected an already-published number; a second real source, structurally incompatible with the first — with none of NHANES's continuous variables, with age in coarse bins — required, and received, a genuinely different representation, and still produced this work's clinically strongest finding, without needing the diabetes variable itself.

The most important result, however, was not any single number — it was testing whether that very finding was robust to the representation that produced it, and discovering that it is not: a fourth variable added to the same source, the same population, the same methodology, changes the structure found and substantially reduces the strength of the reported association. This does not weaken this work's central claim; it is the most direct demonstration of it available. "Disease-agnostic" was not treated, in this article, as a property declared once and assumed thereafter — it was treated as a claim subject to the same scrutiny as any clinical finding, tested against two sources that differ as much as two real clinical data sources plausibly can differ, with negative findings and reversals reported with the same weight as the positive ones.
