# BioSpace

**A computational representation layer for biological systems — not an algorithm, not a disease-specific tool.**

BioSpace formalizes the idea that a patient's computational representation deserves the same scientific scrutiny as the inference algorithm applied to it. Representation, geometry, and inference are treated as three independent methodological degrees of freedom (`F = A(G(R(B)))`), each testable on its own, each capable of changing the final result while the other two stay fixed.

This isn't a slogan applied after the fact. It's tested directly, on three real data sources that share no population, design, or granularity — a clinical sleep-apnea cohort, the NHANES national metabolic health survey, and the UCI Diabetes 130-US Hospitals administrative dataset — and the tests include real negative and mixed results, reported with the same weight as the positive ones.

## Start here

- **[`manuscript.html`](manuscript.html)** — the full formal theory: meta-model, axioms, contracts, geometry, categorical structure, and every empirical validation, in one document.
- **[`index.html`](index.html)** — a shorter, visual tour of the same material, with a live triangulation demo and links to the article series.
- **The article series** (5 short papers, English, no references section by request) — from foundations to a single applied methodological question:
  1. Mathematical Foundations — the meta-model, five axioms, and two functor demonstrations
  2. Geometry as a Missing Axis — representation held fixed, geometry varied, on two independent real sources
  3. Generalization Across Diseases — Type 2 diabetes on two structurally incompatible real sources
  4. Representation Before Inference — a position paper engaging with normative modeling (Marquand et al., 2019)
  5. Convergence Between Independent Methods as Evidence — the triangulation methodology paper
- **[`biospace/METABOLISM_FINDINGS.md`](biospace/METABOLISM_FINDINGS.md)** — an 18-section, chronological log of every real finding (positive, negative, and reversed) produced against real data this project.
- **[`biospace/HISTORY.md`](biospace/HISTORY.md)** — the module-by-module implementation log, phase by phase.

## Installation

```bash
pip install -r requirements-dev.txt --break-system-packages
```

The dashboards each have their own `requirements.txt` (Streamlit + Plotly + the shared scientific stack); install those separately if you only want to run a dashboard, not the full test suite.

## Quickstart

```python
from biospace.plugins.metabolic import MetabolicSystem, MetabolicRepresentation, classify_diabetes_status
from biospace.core import Observation
from datetime import datetime

system = MetabolicSystem(identifier="patient_001")
system.observe(Observation(
    timestamp=datetime(2024, 1, 15), source="lab",
    values={"hba1c": 7.2, "glicemia_jejum": 145, "idade": 58, "imc": 31.2},
))

representation = MetabolicRepresentation()
vector = representation.transform(system, at=datetime(2024, 1, 15))

status = classify_diabetes_status(vector)   # a clinical interpretation, not part of the representation
print(status)  # "diabetes"
```

The point of this example: `classify_diabetes_status` never touches raw measurements. It reads `vector`, a representation already built by domain code that has never heard of diabetes. Swap it for `classify_metabolic_syndrome_risk_full` and nothing about `representation` or `system` changes.

## Package structure

```text
biospace/
├── core/                    # meta-model: BiologicalSystem, Observation, Feature, Representation,
│                             # RepresentationSpace, Geometry, Trajectory, Phenotype, Cohort, Operator
├── geometry/                 # Euclidean · DTW · Gromov-Wasserstein · approximate Riemannian ·
│                             # learned metric (NCA) · relational graph · geometric cohort queries
├── phenotyping/              # K-Means · HDBSCAN · GMM · Spectral
├── dynamics/                 # mean-reverting evolution operator, stability, semigroup law
├── early_warning/            # critical slowing down (Dakos et al. 2012 methodology)
├── survival/                 # Kaplan-Meier · Cox · log-rank, ordinal-time (no calendar dates)
├── causal/                   # baseline balance, propensity, digital-twin counterfactuals
├── gnn/                      # graph convolutional network, pure NumPy
├── graph/                    # patient similarity graph construction
├── representation_learning/  # autoencoders vs. PCA, honestly compared
├── foundation/                # architectural prototype for a shared multi-domain foundation model
├── prediction/                # any sklearn-compatible classifier over a RepresentationSpace
├── explainability/            # SHAP over any trained predictor
├── risk/                      # transparent linear risk scores
├── latent/                    # genuine factor analysis, not an arbitrary weighted index
├── anomaly/                   # Isolation Forest · LOF · One-Class SVM
├── topology/                  # Mapper · persistent homology (Betti numbers, persistence diagrams)
├── intervention/              # what-if / counterfactual queries over a fitted digital twin
├── longitudinal/              # trajectory updates, transition analysis, survival (calendar-time)
├── ontology/                  # auto-generated ontology from domain/observable declarations
├── datasets/                  # NHANES and UCI Diabetes 130-US Hospitals real-data loaders
└── plugins/
    ├── metabolic/             # the real representation: 7 physiological domains, disease-agnostic
    ├── diabetes/               # a thin re-export layer over metabolic/ — see below
    └── sleep/                  # the original OSA representation (8 domains, real clinical cohort)
```

Every module has at least one dedicated test file under `tests/`. There is no module in this package with zero test coverage.

### Why `plugins/diabetes/` is nearly empty

This is the clearest illustration of the project's central claim. `plugins/metabolic/` represents the endocrine-metabolic system with no reference to diabetes anywhere — seven domains (glycemic, anthropometric, cardiovascular, renal, lipid, comorbidity, treatment), each named by physiological meaning. `plugins/diabetes/` re-exports that representation under old names (`DiabetesSystem is MetabolicSystem`, verified by object identity) and contributes only what is genuinely diabetes-specific: the ADA glycemic classification criterion, and a synthetic generator that simulates a diabetes scenario. Diabetes is a **clinical interpretation** applied over the representation — `B → R(B) → N(R(B))` — never a property of the representation itself. The same representation supports a second, independent interpretation (`classify_metabolic_syndrome_risk_full`, the full NCEP ATP III criterion) with zero modification.

## Data sources

| Source | Kind | Size | Notes |
|---|---|---|---|
| SAOS (sleep apnea) | Real clinical cohort | 355 patients, 1,556 exams | The original application case; longitudinal, real calendar dates |
| NHANES | Real public survey | 9,232 adults ≥20y | Cross-sectional, dense continuous lab variables |
| UCI Diabetes 130-US Hospitals | Real public administrative records | 71,518 patients, 101,766 encounters | Longitudinal by encounter order (no real dates); none of NHANES's continuous variables |
| Synthetic diabetes generator | Fabricated, known ground truth | Configurable | Used to validate contracts before any real-data claim |

No finding in this project is reported from a single source without at least being framed against what a second, structurally different source would need to look like to replicate it.

## Formal contracts

Eleven contracts, each with a dedicated empirical test (not just a formal declaration): Traceability, Semantic Preservation, Compositionality, Continuity, Extensibility, Algorithmic Independence, Temporality, Reproducibility, Versionability, Interoperability, and Population Injectivity. Only representations satisfying all eleven are considered scientifically admissible — a representation that fails even one contract is outside the theory's domain, regardless of whether the code still produces output for it. Full definitions and test references: `manuscript.html`, §5, and `biospace/README.md`.

## Dashboards

Four Streamlit dashboards, 49 pages total, each with an automated per-page smoke-test script under `scripts/`:

| Dashboard | Pages | Source |
|---|---|---|
| `biospace_dashboard/` | 19 | SAOS (real clinical cohort) |
| `biospace_dashboard_diabetes/` | 12 | Synthetic diabetes generator |
| `biospace_dashboard_nhanes/` | 8 | NHANES (real) |
| `biospace_dashboard_uci/` | 10 | UCI Diabetes 130-US Hospitals (real) |

Run any of them with `streamlit run App.py` from inside the corresponding directory. Verify a dashboard's pages load and its buttons don't throw with `python3 scripts/check_dashboard<_name>.py` before assuming a change works — several real bugs in this project were only caught by clicking a button in a test, not by the page loading successfully.

## Tests

361 tests, `pytest tests/`. A handful require real uploaded data files (NHANES `.XPT` files, the UCI `diabetic_data.csv`) and are skipped automatically if those files aren't present — see `tests/test_uci_diabetes_real_data.py`, `tests/test_nhanes_real_data.py`, `tests/test_survival.py`, and `tests/test_critical_slowing_down.py`. Every module built this project was validated first against synthetic data with known ground truth, then — where a real source existed — against real data, with the real-data result reported honestly whether or not it confirmed the synthetic expectation.

## Known limitations

- No real long-term mortality or cardiovascular-event outcome in any of the three real sources — UCI's readmission is the closest thing to a hard outcome, and it's short-term.
- Temporal-discretization invariance for the evolution operator's φ does **not** hold — confirmed by direct test, not assumed. Comparing φ fitted at different observation frequencies is not, today, a valid operation.
- DTW, as implemented here, violates the triangle inequality — confirmed against a published counterexample (Tralie et al., 2022), not merely cited from the general literature. Treated as a dissimilarity, not a metric, throughout.
- The relational-geometry (graph) advantage found on SAOS did **not** replicate in the same direction on NHANES or UCI — reported as a genuinely mixed result in Article II, not smoothed over.
- Roughly half of the AI-technique taxonomy this project has been checked against (deep learning proper, knowledge-graph embeddings, RNN/LSTM forecasting, reinforcement learning, federated learning, few-shot, multimodal, foundation models, formal ontology linkage) remains unimplemented — mostly because the data this project has doesn't support them yet (no sequential treatment decisions, no multi-institution data, no imaging/text/genomics), not because they were forgotten.

See `biospace/METABOLISM_FINDINGS.md` for the complete, dated account of every finding — including the bugs found along the way and the published numbers they corrected.
