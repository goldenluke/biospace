# BioSpace

*[Leia em português](README.pt-BR.md)*

Computational infrastructure for representing biological systems — not a
library of medical algorithms. The role NumPy plays for numerical
computing, or scikit-learn for machine learning: a foundational layer
on top of which disease-specific plugins (today, only OSA/sleep apnea)
are built.

Implements the meta-model M = (B, O, D, R, X, G, Γ, F, C):

| Symbol | Entity | Class |
|---|---|---|
| B | Biological system | `BiologicalSystem` |
| O | Observation | `Observation`, `Observable`, `Measurement` |
| D | Semantic domain | `SemanticDomain`, `LatentDomain` |
| R | Representation | `Representation`, `RepresentationVector` |
| X | Representation space | `RepresentationSpace` |
| G | Geometry | `Geometry`, `TrajectoryGeometry`, `DynamicGeometry` |
| Γ | Trajectory | `Trajectory` |
| F | Phenotype | `Phenotype` |
| C | Cohort | `Cohort` |

**For the complete history of how each piece was built — real
findings, bugs found and fixed, design decisions and why — see
[HISTORY.md](HISTORY.md).** This file is usage reference only.

## Installation

```bash
pip install -r requirements-dev.txt   # numpy, scipy, scikit-learn, pandas, POT, pytest
```

No `setup.py`/`pyproject.toml` yet — use directly from the repository
root (`import biospace`) or add it to `PYTHONPATH`.

## CI

`.github/workflows/ci.yml` runs on every push/PR: the test suite,
`examples/*.py`, `check_install.py`, `demo_sleep.py`, and
`scripts/check_dashboard.py` (verifies ALL dashboard pages via
`AppTest` with synthetic data — catches exactly the class of error that
already broke a real deployment on Streamlit Cloud: one file out of
sync with another). Run `python3 scripts/check_dashboard.py` locally
before pushing dashboard changes.

## Quickstart

```python
from datetime import datetime
from biospace.plugins.sleep import SleepSystem, SleepRepresentation, exam
from biospace.core import Cohort

representation = SleepRepresentation()
cohort = Cohort()

system = SleepSystem(identifier="patient_1")
system.observe(exam({"ido": 22.0, "idade": 55, "imc": 31.0}, timestamp=datetime(2024, 1, 1)))
cohort.update(system, representation, timestamp=datetime(2024, 1, 1))

space = cohort.snapshot()
matrix, ids = space.matrix()   # ready for any Geometry/Phenotyper/Predictor
```

Fields omitted from the `exam(...)` dict become structural absence
(imputed as z=0, weighted by completeness — not an error). See
`examples/` for complete cases with all fields and multiple exams.

## Package structure

```
biospace/
├── core/            fundamental entities (never disease- or algorithm-aware)
│   ├── biological_system.py, observation.py, measurement.py, feature.py
│   ├── domain.py, latent_domain.py, representation.py, composite_representation.py
│   ├── representation_space.py, trajectory.py, cohort.py, phenotype.py
│   ├── geometry.py (interface), operator.py (interface), distribution.py
│   └── contracts.py            empirical verification of formal contracts (see table below)
│
├── geometry/        concrete implementations of Geometry/TrajectoryGeometry/DynamicGeometry
│   ├── euclidean.py, mahalanobis.py, wasserstein.py, information.py, cosine.py
│   ├── dtw.py, gromov_wasserstein.py       (TrajectoryGeometry)
│   ├── riemannian.py                        (approximate geodesic via k-NN graph)
│   ├── learned.py                            (NCA — learned metric)
│   └── dynamic.py                             (d(x,y,t) — phenotype-conditioned metric)
│
├── phenotyping/     Operator[list[Phenotype]]
│   ├── kmeans.py, clinical_kmeans.py, hdbscan.py, gaussian.py, spectral.py
│   └── contracts.py             check_phenotype_stability (Contract 8.5)
│
├── prediction/      Predictor (wrapper around any sklearn estimator)
├── risk/            RiskOperator (LinearRiskOperator)
├── intervention/    InterventionOperator — τ: X -> X (FeatureShiftIntervention)
├── early_warning/   EarlyWarningOperator — CriticalSlowingDownDetector
├── longitudinal/    SurvivalAnalyzer (Kaplan-Meier), TransitionAnalyzer, TrajectoryUpdater
├── dynamics/        EvolutionOperator, StabilityOperator, DynamicSystem
│                    (learned spontaneous dynamics — Trajectory -> Future State)
├── causal/          check_baseline_balance, ObservationalEffectEstimator,
│                    estimate_propensity/match_on_propensity/estimate_matched_effect,
│                    DigitalTwin, Scenario
├── latent/          FactorAnalysisLatentDomain (generic Factor Analysis)
├── ontology/        Ontology (data dictionary), run_contract_suite (all contracts)
│
└── plugins/
    ├── sleep/       disease plugin #1 — OSA, validated on real data
    │   ├── domains.py                8 domains (Anthropometric, Apnea, Hypoxia, SleepArchitecture,
    │   │                              Cardiovascular, Comorbidity, Symptoms, Treatment)
    │   ├── latent.py                  InflammationProxyDomain, FrailtyProxyDomain, AutonomicBalanceProxyDomain
    │   ├── hierarchical.py             HierarchicalSleepRepresentation (grouping by physiological system)
    │   ├── loader.py                   load_from_excel/load_from_dataframe (groups by patient)
    │   └── representation.py, system.py, builders.py, clinical_maps.py
    │
    └── diabetes/    disease plugin #2 — Type 2, entirely synthetic (architectural rigor, not clinical)
        ├── domains.py                6 domains (Glycemic, Anthropometric, Cardiovascular, Renal, Comorbidity, Treatment)
        ├── latent.py                  InsulinResistanceProxyDomain
        ├── synthetic.py                longitudinal generator (renal decline via accumulated glycemic exposure)
        └── loader.py, representation.py, system.py, builders.py, reference.py
```

## Reference by module

### `core` — never disease- or algorithm-aware

- `BiologicalSystem(identifier)` — accumulates `Observation`s; never recreated, only updated.
- `Observation(timestamp, source, values)` / `Observable` (subclassed per quantity) / `Measurement` (with provenance + optional uncertainty via `Distribution`).
- `SemanticDomain(observables)` — `encode(measurements) -> list[Feature]`; `name` required (validated at construction).
- `LatentDomain(source_domains)` — a domain with no Observables of its own; requires a declared `hypothesis` and defaults to `is_validated=False`.
- `Representation(domains)` — `transform(system, timestamp=None) -> RepresentationVector`; rejects domains with colliding names.
- `CompositeRepresentation(name, children)` — groups domains (or other groups) under a name, behaves like an ordinary domain.
- `RepresentationSpace` / `Cohort` / `Trajectory` — storage; `Cohort.snapshot()` produces a cross-sectional `RepresentationSpace`.
- `Phenotype(name, membership_fn, interpretation)` — a region of X, never an algorithm.
- `Operator[TOutput]` / `LongitudinalOperator[TOutput]` — marker interfaces (only `describe()` is universal).
- `Distribution` / `Normal` / `PointMass` — probabilistic observations (`std < 0` rejected at construction).
- `contracts.py` — see the formal contracts table below.

### `geometry`

| Class | What it measures |
|---|---|
| `Euclidean`, `Mahalanobis`, `Wasserstein`, `InformationGeometry`, `Cosine` | Distance between points in X |
| `DTW`, `GromovWasserstein` | Distance between whole trajectories (`TrajectoryGeometry`) |
| `RiemannianGeometry` | Approximate geodesic via k-NN graph — non-flat space |
| `LearnedGeometry` | Metric learned via NCA from labels |
| `PhenotypeConditionedGeometry` | `d(x,y,t)` — local per-phenotype metric (Ledoit-Wolf) |

### `dynamics` — Trajectory → EvolutionOperator → Future State

`MeanRevertingEvolutionOperator.fit(cohort)` fits a discrete
Ornstein-Uhlenbeck process per Feature; `.predict(x, delta_t_days)`
extrapolates. `StabilityOperator` summarizes how many Features diverge
(φ≥1). `DynamicSystem` combines a `Trajectory` with a fitted
`EvolutionOperator` for `predict()`/`simulate()`.

### `causal` — adjusted observational association, NEVER causal proof

`do()` (on `DigitalTwin`) applies a transformation on the
representation space — it is not identified causal inference in
Pearl's sense. Recommended sequence:

```python
from biospace.causal import check_baseline_balance, estimate_propensity, match_on_propensity, estimate_matched_effect

balance = check_baseline_balance(cohort, "treatment", "aam", order=order)   # ALWAYS first
model = estimate_propensity(cohort, "treatment", "aam", order=order)         # L2-regularized logistic regression
match_result = match_on_propensity(cohort, "treatment", "aam", order=order)  # caliper matching
effect = estimate_matched_effect(cohort, match_result, order=order)          # difference-in-differences
```

### `plugins/sleep`

The only disease plugin fully implemented. `load_from_excel(path)` /
`load_from_dataframe(df)` group rows by patient (`id_column="paciente"`)
and sort by date (`order_column="inicio"`), producing one `SleepSystem`
per patient with a full trajectory — never one system per row.

## Formal contracts (empirical verification, not mathematical proof)

| # | Contract | Function |
|---|---|---|
| 5.2 | Semantic Preservation | `core.contracts.check_semantic_preservation` |
| 5.3 | Compositionality | `core.contracts.check_compositionality` |
| 5.4 | Continuity | `core.contracts.check_lipschitz_continuity` |
| 5.5 | Extensibility | `core.contracts.check_extensibility` |
| 5.6 | Algorithmic Independence | `core.contracts.check_algorithmic_independence` |
| 5.7 | Temporality | `core.contracts.check_temporality` |
| 5.8 | Reproducibility | `core.contracts.check_reproducibility` |
| 5.9 | Versionability | `core.contracts.check_representation_schema_compatibility` |
| 5.10 | Interoperability | `core.contracts.check_interoperability` |
| 8.5 | Phenotype Stability | `phenotyping.contracts.check_phenotype_stability` |
| — | Injectivity (population-level) | `core.contracts.check_injectivity` |

All 11 run together via `ontology.run_contract_suite(...)` — each one
is only checked if the necessary data is supplied.

**Genuinely pending** (not implementable as an automatic checker, by
nature): 5.1 Traceability — today only implicit via
`Feature.provenance` (every Feature carries where it came from), but
there is no formal "test" for traceability beyond manually inspecting
that field.

**5.3, 5.6, and 5.10 were closed in a dedicated investigation round**
(see "Closed scientific gaps" section below) — 5.9 was already covered
by `check_representation_schema_compatibility`, it just wasn't labeled
as such.

## Closed scientific gaps (dedicated investigation)

### `hypoxia.tempo_total_em_hipoxemia_min`: the "genuine divergence" was 1 outlier, not a real signal

For several sessions this was recorded as the one Feature with genuine
divergence (φ=1.0039) detected by `EvolutionOperator` on the real
data — "would deserve dedicated investigation." Investigated: the
Feature has a strongly skewed distribution (median=0, 75th
percentile=0, maximum=135, against most patients at zero). Removing
**a single patient** (`sleep_ls_000035`, the outlier with value 135,
out of ~296 patients with a trajectory), φ drops to 0.984 —
**stable**. The divergence was never real population-level
progression, it was sensitivity to a single isolated outlier.

Formalized as a reusable diagnostic:
`biospace.dynamics.check_feature_stability_robustness(cohort, feature_name, order)`
— leave-one-patient-out on a specific Feature, reports whether the
stability conclusion depends on any individual patient. Tested both in
the positive case (recovers the real finding above) and in a synthetic
counter-case (a genuinely stable Feature stays robust to removal of
any patient).

### Phenotype stability: a sweep of 4 algorithms confirms the instability is real, not a choice of K

ARI=0.42 (K=4, `ClinicalKMeansPhenotyper`) was recorded as "worth
revisiting with a different K or algorithm." Done: a sweep of ~28
configurations — KMeans (K=2..8), GaussianMixture (K=2..6), Spectral
(K=2..6), HDBSCAN (min_cluster_size=5..30) — **none crosses the
stability threshold** (ARI≥0.7, Hennig 2007). The closest was
GaussianMixture K=2 (ARI=0.583). Spectral clustering performed worse
than random at small K (ARI≈0 or slightly negative). Conclusion: the
OSA population forms a severity CONTINUUM, not sharply separated
clusters — this is not an artifact of K or of the algorithm. See
`tests/test_phenotype_stability_sweep.py`.

## Tests

```bash
pytest              # tests/ (pytest.ini already points there)
```

Covers the 11 formal contracts, one test per real bug found during
development (see `tests/test_core_hardening.py` and
`tests/test_regressions.py`-equivalents scattered across modules), and
proof that the core is genuinely generic
(`tests/test_core_disease_agnostic.py` — builds a new disease from
scratch, with no dependency on the sleep plugin). Most tests do not
depend on the real Excel file (patient data does not belong in the
repository); the few that do (findings specific to the real data, like
the two above) use `pytest.mark.skipif` and are skipped when the file
is not present (e.g., in CI).

Also see `examples/` — standalone, commented scripts meant to be read
by humans (not to lock in regressions like the tests do).

## Dashboard

`biospace_dashboard/` — Streamlit, 19 pages, built entirely on top of
`biospace` (never recomputes representation/phenotype itself). See its
own README for details. Runs with real data (.xlsx upload) or a
longitudinal synthetic cohort generated locally.

## Known limitations (summary — details in HISTORY.md)

- Two disease plugins: OSA (`plugins/sleep/`, validated with real
  data) and Type 2 Diabetes (`plugins/diabetes/`, entirely synthetic —
  see section below). The genericity of the core is tested by two
  independent examples, but no third disease with real data has yet
  exercised this in production.
- No adjustment for unmeasured confounders in `causal/` — propensity
  matching reduces, never eliminates, confounding by indication.
- `RiemannianGeometry` and `GromovWasserstein` are computationally
  expensive (do not scale to large populations without optimization).
- Latent domains (`InflammationProxyDomain` etc.) are explicitly
  unvalidated statistical hypotheses (`is_validated=False`) — no
  independent biomarker in this dataset to confirm them.

## Phase 5 — Representation Learning (`representation_learning/`)

```
Today:  System -> Representation
Later:  System -> Representation -> Representation Learning
```

Central architectural difference: learning happens **on top of the
`RepresentationSpace` already computed by the physiological domains** —
never from raw `Observation`/`Measurement` data. `RepresentationLearner.fit(space)`
is the central contract of the interface (not an implementation
detail): this means learning can never rediscover structure that the
domains have not already exposed as an axis — it reorganizes/compresses
what the domains defined, it does not replace that definition.

`AutoencoderRepresentationLearner` — a NON-LINEAR autoencoder via
`sklearn.neural_network.MLPRegressor` (input = output = X, with a
narrow intermediate layer = the embedding), with manual extraction of
the bottleneck activation. **Why not PyTorch/TensorFlow**: tested —
`pip install torch` downloads ~1GB of CUDA dependencies even for
CPU-only use, incompatible with a framework that must stay lightweight
and testable in CI.

```python
from biospace.representation_learning import AutoencoderRepresentationLearner, compare_reconstruction_error

ae = AutoencoderRepresentationLearner(embedding_dim=5, hidden_dim=16)
ae.fit(space)  # never cohort/raw data — always the already-computed RepresentationSpace
embedding = ae.transform(space.get(some_id).as_vector(order))

result = compare_reconstruction_error(space, embedding_dim=5)  # automatically compares against PCA
```

### Real finding #1: on a KNOWN non-linear latent structure, the autoencoder clearly wins

Built a controlled synthetic scenario — 8 Features observed from a 2D
latent, mixing linear and NON-linear combinations (sine, product,
difference of squares). Reconstruction error: PCA (linear) = 0.562;
Autoencoder (non-linear) = 0.125 — **~4.5x better**. Confirms that
non-linearity captures something real when it exists.

### Real finding #2 (the most important one): on the real OSA data, PCA won in EVERY configuration tested

On the 355 real patients (52 dimensions), I tested the autoencoder
against PCA at several embedding dimensions (2, 5, 10) and several
hyperparameter combinations (`hidden_dim`, `alpha`, `max_iter`) — **PCA
won in all of them**. This is not a tuning problem: it is a real lesson
about the difference between linear methods (analytic solution, few
parameters) and non-linear ones (gradient-trained, far more
parameters) on a small sample — 355 patients is too few for a neural
network to find a better solution than PCA's optimal linear solution,
even if the true underlying structure had some non-linearity.

**This is why `compare_reconstruction_error()` exists and runs both
automatically**: the discipline of "never trust the non-linear method
without comparing it to the linear one" is built into the API itself,
not left as something the user has to remember.

### Real finding #3 (while validating the test): comparing by RATIO against a near-perfect PCA is unstable

In a third scenario (EXACTLY linear data), PCA reached near-zero error
(0.0018 — the analytic solution). The autoencoder, gradient-trained,
had an optimization gap (0.04) — small in absolute terms, but 22x worse
in RATIO, only because the denominator (PCA's error) is nearly zero. I
adjusted the corresponding test to compare absolute error, not ratio,
precisely because of this instability.

## Phase 10 — Foundation Model: architectural prototype (`foundation/`)

```
Millions of patients -> BioSpace -> Foundation Model
(not trained on text, trained on physiological states)
```

Explicitly marked as "**in the future**" in the original request —
treated as such. With 355 real patients and synthetic cohorts, this
project is ~4 orders of magnitude away from "millions of patients": the
difference is not one of degree, it is one of scale. Nothing here is,
or pretends to be, a foundation model.

What was built: `MaskedFeaturePredictor` — an ARCHITECTURAL proof of
concept that this project's representation (Features with semantic
provenance, built by `SemanticDomain`) is a valid substrate for
self-supervised pretraining, in the same pattern as BERT's Masked
Language Modeling (Devlin et al., 2019) — but masking physiological
Features instead of words.

```python
from biospace.foundation import MaskedFeaturePredictor

model = MaskedFeaturePredictor(hidden_dim=32, mask_fraction=0.15)
model.fit(space)  # RepresentationSpace of ONE disease — never raw data
result = model.masked_reconstruction_error(space)  # honest: error only at masked positions
```

### Decisive validation: known synthetic correlation structure

Tested BEFORE trusting real data — Features with known correlation
(f2=2·f1, f3=-f1) should reconstruct much better than the naive
baseline (predicting the mean); pure-noise Features (f4, f5, unrelated
to anything) should NOT "pretend" to reconstruct better than that same
baseline:

| Feature | Reconstruction MSE | Variance (baseline) | Ratio |
|---|---|---|---|
| f1, f2, f3 (correlated) | 0.016 – 0.040 | 1.04 – 4.22 | **0.01 – 0.04** |
| f4, f5 (pure noise) | 1.17 – 1.37 | 0.98 – 1.06 | **1.20 – 1.29** |

The model learns real structure where it exists (ratio «1) and does
not invent structure where there is none (ratio ≈1, honestly no better
than the baseline).

### On the real OSA data: an honest finding about physiological redundancy

Comorbidities/treatments reconstruct very well from the rest of the
profile (MSE/variance ratio of 0.02–0.06 for `irritabilidade`,
`cancer`, `doenca_coronaria`, `cpap`) — plausible: comorbidities tend
to co-occur in more severe patients. Age (`anthropometric.idade`) and
some hypoxemia extremes (`hypoxia.tempo_spo2_90`) come out with a
ratio near or above 1.0 — the model does not beat the baseline — also
plausible: age carries information relatively INDEPENDENT of momentary
physiological state. Genericity confirmed: the same
`MaskedFeaturePredictor`, with no changes, runs identically on the
diabetes plugin.

### What is deliberately left out

Training with a "pure" masked loss (only at masked positions — true
MLM) was not implemented; scikit-learn's `MLPRegressor` does not
accept a per-sample masked loss, so training reconstructs the whole
vector (a documented simplification). EVALUATION, however, always
measures error only at the positions actually masked — the reported
metric is honest even with this training simplification. Combining
multiple diseases into a single model (the step that would make this
more like a real foundation model) was not attempted — each plugin's
Feature space has a different dimensionality, and solving that
properly is a design problem in itself, not a line of code.

## Phase 9 — Simulation: `twin.simulate_ensemble()` (many futures, not just one)

```
twin = patient.clone()
twin.simulate(...)
```

`DigitalTwin.clone_from(trajectory)` (= `patient.clone()`) and
`DigitalTwin.simulate()` (deterministic) already existed, built during
the Causal Inference work. What was missing: ENSEMBLE simulation — a
real digital twin reports uncertainty about the future, not a single
deterministic point.

```python
result = twin.simulate_ensemble(evolution_operator, horizon_days=2000, step_days=50, n_samples=500)
result["mean"]   # mean of the predictive distribution, per timestep
result["std"]    # standard deviation of the predictive distribution, per timestep
result["paths"]  # all n_samples simulated trajectories
```

Uses `EvolutionOperator.sample()` (the STOCHASTIC counterpart of
`predict()`, which already existed with the theoretically correct
variance formula for a discrete Ornstein-Uhlenbeck process) —
`simulate_ensemble()` only needed to chain calls to `sample()`
correctly.

### Real finding and fix: wrong noise scale inflated variance by ~7x

Validating `simulate_ensemble()` against the KNOWN THEORETICAL
STATIONARY variance of a synthetic OU process (target_var=4.0), the
simulated variance converged to **29.2** — nearly 7.5x the true value.

**Root cause**: `residual_std` (the residual standard deviation already
fitted by `MeanRevertingEvolutionOperator`) is on the scale of the
AVERAGE Δt of the pairs used in fitting (in real clinical data,
typically tens to hundreds of days) — but the OU process's stationary
variance formula needs the noise on a **1-day** scale, consistent with
`phi_per_day`. `sample()` was using `residual_std` directly as if it
were already that 1-day scale.

**Fix**: added `FeatureDynamics.mean_dt_days` (the average Δt of the
fitting pairs) and `FeatureDynamics.sigma_eps_per_day`, which inverts
the stationary variance relationship:

```
residual_std² = σ_day² · (1 - φ^(2·mean_dt)) / (1 - φ²)
   =>  σ_day = residual_std / sqrt((1 - φ^(2·mean_dt)) / (1 - φ²))
```

After the fix, the same validation converged to **3.60** — within 10%
of the true theoretical value (4.0), within what is expected for
sampling noise with 500 simulations. Formalized as a permanent
regression test
(`tests/test_simulation.py::test_simulate_ensemble_variance_converges_to_known_stationary_variance`)
— if the noise scale regresses back to the original bug, this test
fails.

## Phase 8 — Geometry: Manifold, Curvature, Stability, Metastability

```
Today:  Patient -> Representation
Later:  Patient -> Representation -> Manifold -> Trajectory -> Curvature -> Stability
```

Most of this chain already existed before this item: `RiemannianGeometry`
(Manifold — approximate geodesic via k-NN graph), `Trajectory`,
`StabilityOperator`/`EvolutionOperator` (Stability). I also found
substantially advanced work already in place (from a previously
interrupted session) covering Curvature and Metastability — I verified
all of it, added a third complementary perspective, and document all
three together here for the first time.

### Three INDEPENDENT ways of estimating curvature

| Source | Function | What it measures |
|---|---|---|
| Temporal | `FeatureDynamics.curvature` (`biospace.dynamics`) | k = -ln(φ) — directly from the φ already fitted by `MeanRevertingEvolutionOperator`; HIGH curvature = narrow, deep well = fast recovery = high resilience |
| Population density | `estimate_density_curvature`, `detect_metastability` (`geometry/curvature.py`) | U''(x) at the mode of an effective potential U(x)=-log(density(x)) reconstructed by KDE over the cross-sectional population |
| Structural (graph) | `ollivier_ricci_curvature`, `graph_curvature_summary` (`geometry/graph_curvature.py`) | Ollivier-Ricci curvature (optimal transport between neighborhoods) over the k-NN graph of the MANIFOLD — the only one of the three that actually uses the manifold itself |

These are three measurements from different sources (dynamics of a
single patient over time; the shape of one Feature's population
distribution; the relational structure of the whole space) — they are
not expected to numerically coincide, but agreement in DIRECTION is
evidence they capture something real.

**Validation of temporal curvature**: synthetic Ornstein-Uhlenbeck
process with a known true k — `FeatureDynamics.curvature` recovered k
almost exactly (ρ=0.99+ with the real k, relative error <25%).

**Real finding (limitation) of density-based curvature**:
differentiating a KDE curve twice greatly amplifies imprecision —
tested with 4 Features of IDENTICAL true curvature by construction,
the returned value varied nearly 5x (0.14 to 0.71). Documented as a
QUALITATIVE indicator, not for precise numerical comparison.

**Validation of structural curvature (my contribution)** — 3 synthetic
cases with results known from the Ollivier-Ricci literature, tested
BEFORE trusting real data:
- Complete graph → +0.625 (strongly positive, as expected).
- Large cycle → exactly 0.000 (a classic, well-cited result).
- Binary tree → global average +0.033, initially surprising (expected
  negative); investigating by edge type: internal (backbone) edges =
  -0.310 (negative, as expected — structural bottleneck), leaf edges =
  +0.333 (positive — a known artifact of the lazy walk at degree-1
  nodes). The global average just mixes the two.

**On the real OSA data** (similarity graph, 355 patients): edges that
cross different phenotypes have more negative curvature (-0.146) than
edges within the same phenotype (-0.076) — Mann-Whitney p=5.7e-19.
Boundaries between phenotypes are structurally more fragile
("bottlenecks"), exactly the interpretation from the Ollivier-Ricci
literature as a signal of reduced resilience.

### Metastability

`detect_metastability(space, feature_name)` counts how many distinct
WELLS exist in the potential landscape of a Feature — more than one
well = multiple genuine stable states (not just "there appears to be
clustering," but "there is a real energy barrier between the groups,"
quantified by `Well.escape_barrier`). Validated: unimodal population →
1 well; well-separated bimodal population → 2 wells with a substantial
barrier (>1.0); noise within a single Gaussian does not spuriously
fragment into multiple wells.

### Connecting to what already existed: Early Warning Signals, Resilience, Critical Slowing Down

The four cited concepts already had (or gained, with this item) a
concrete place in the project:

| Concept | Where it lives |
|---|---|
| Early Warning Signals | `early_warning.CriticalSlowingDownDetector` |
| Critical Slowing Down | same — increasing variance/autocorrelation/skewness in a sliding window |
| Resilience | `FeatureDynamics.resilience_score` (a direct alias for `curvature`) |
| Metastability | `detect_metastability` (potential wells) + structural graph curvature (phenotype boundaries) |

## Phase 7 (part 2) — GNN: `SimpleGCN` in pure NumPy (`gnn/`)

Direct continuation of the graph (`graph/`) — the "then GNN" we had
explicitly left pending. `SimpleGCN`: a real Graph Convolutional
Network (Kipf & Welling, 2017), forward AND backward (gradient)
derived and hand-written in pure NumPy — the same decision not to use
PyTorch/TensorFlow already made in `representation_learning/` and
`graph/` (~1GB of CUDA even for CPU, and PyTorch's CPU-only index
isn't even in the allowed network domain list here).

**Critical step, done BEFORE trusting any result**: the analytical
gradient (manual backward pass) was checked against finite differences
— relative error of ~1e-8 across all parameters. A manual backprop
written incorrectly generates no runtime error at all; the model
"trains" toward the wrong direction silently. Without this check,
nothing else here would have any value.

```python
from biospace.graph import build_cohort_similarity_graph
from biospace.gnn import SimpleGCN, prepare_node_classification_data
from biospace.geometry import Euclidean

G = build_cohort_similarity_graph(space, Euclidean(), k=8)
data = prepare_node_classification_data(space, G, labels=phenotypes, labeled_ids=train_ids)

gcn = SimpleGCN(hidden_dim=16)
gcn.fit(data["A"], data["X"], data["y"], data["labeled_mask"], epochs=300)
predictions = gcn.predict(data["A"], data["X"])
```

### Real finding: the classic crossover pattern from the semi-supervised GCN literature

Tested on the 355 real patients, predicting phenotype (transductive,
semi-supervised — Kipf & Welling 2017), comparing the GCN (uses the
graph) against the SAME model with no message passing
(`A=identity` — isolates what the graph contributes):

| Labeled fraction | With graph | Without graph | Difference |
|---|---|---|---|
| 50% (177 patients) | 0.882 | 0.927 | **-0.045** (graph hurts) |
| 20% (71 patients) | 0.817 | 0.845 | -0.028 |
| 10% (35 patients) | 0.794 | 0.750 | +0.044 |
| 5% (17 patients) | 0.757 | 0.580 | **+0.178** (graph helps a lot) |

**Why this makes sense, and is not noise**: the phenotypes come from
`ClinicalKMeansPhenotyper` over the SAME X space used as Features — a
direct classifier on X already reconstructs those boundaries almost
perfectly when there is enough data, and neighbor smoothing from the
graph can blur the exact boundary near the divide between clusters. But
when labels become scarce (17 of 355 patients), "borrowing" information
from the graph's structure (similar patients tend to share phenotype —
validated earlier in `graph/`: 74.9% vs. 27.4% random) compensates
substantially for the lack of labeled examples — exactly the original
motivation Kipf & Welling gave for proposing semi-supervised GCNs.

## Phase 7 (part 1) — Knowledge Graph: the patient stops being a vector (`graph/`)

```
Today:  patient.vector() -> x ∈ X (a point)
Later:  patient.graph()  -> G = (V, E) (a network)
```

Scope deliberately limited to the GRAPH — the GNN itself ("then GNN")
was explicitly set as the next step by the original request, not
implemented here. Two levels:

**`build_patient_graph(system, representation, feature_correlations=None)`**
— the INTERNAL network of a single patient: nodes for the patient,
each domain, each Feature, and each comorbidity/treatment PRESENT;
edges `OBSERVED`/`BELONGS_TO`/`HAS`, and `CORRELATES_WITH` between
Features whose REAL POPULATION correlation (`compute_feature_correlations`,
not an invented ontology) exceeds a threshold. Tested on real data: the
strongest correlation found was `apnea.ido` × `apnea.no_de_dessaturacoes`
(+0.971) — makes complete clinical sense, both measure the frequency
of obstructive events.

**`build_cohort_similarity_graph(space, geometry, k=5)`** — the network
of PATIENTS (not a single patient's internal network) — nodes =
patients, edges = k nearest neighbors under any `Geometry` already in
the project. This is the structure a future GNN would consume (message
passing between similar patients). **Validated on real data**:
neighbors in the graph share phenotype in 74.9% of cases, against
27.4% for a random baseline — nearly 3x better, confirms the network
captures real structure.

**`to_pyg_arrays(graph)`** — exports `node_features`/`edge_index` in
the raw format that PyTorch Geometric/DGL expect, WITHOUT depending on
any GNN framework (`pip install torch` downloads ~1GB of CUDA even for
CPU — the same decision already made in `representation_learning/`) —
leaves this next step ready to plug in, without pretending to already
be that step.

```python
from biospace.graph import build_patient_graph, build_cohort_similarity_graph, compute_feature_correlations, to_pyg_arrays
from biospace.geometry import Euclidean

correlations = compute_feature_correlations(space)
G_patient = build_patient_graph(system, representation, feature_correlations=correlations)
G_cohort = build_cohort_similarity_graph(space, Euclidean(), k=5)
arrays = to_pyg_arrays(G_patient)  # node_features, edge_index -- ready for a GNN, whenever it comes
```

**Finding while validating the test itself**: in a synthetic
population with 2 well-separated groups, two Features generated
INDEPENDENTLY within each group still correlated >0.9 across the whole
population — an "ecological" correlation (Simpson-like: both jump
together between groups, even though they are independent within
each). Not a bug — a real lesson about interpreting population-level
correlation; the test was adjusted to isolate what it actually meant
to check.

## Second plugin: Type 2 Diabetes (`plugins/diabetes/`)

Built entirely synthetic (no real patient data) — the goal is
architectural rigor, not clinical validation. Same discipline as the
sleep plugin: 6 semantic domains (`GlycemicDomain`,
`AnthropometricDomain`, `CardiovascularDomain`, `RenalDomain`,
`ComorbidityDomain`, `TreatmentDomain`), completeness-weighted z-score,
one latent domain with a declared hypothesis
(`InsulinResistanceProxyDomain`, `is_validated=False`), a realistic
longitudinal generator (`generate_synthetic_dataframe`), and a
`loader.py` that groups by patient.

```python
from biospace.plugins.diabetes import generate_synthetic_dataframe, load_from_dataframe

df = generate_synthetic_dataframe(n_per_group=30, seed=42)
cohort, representation = load_from_dataframe(df)
```

**Deliberate rigor finding in the generator**: renal decline
(eGFR/creatinine) correlated with ACCUMULATED GLYCEMIC EXPOSURE over
time, not just baseline severity — chronic hyperglycemia damages the
kidneys, a real mechanism. Tested: positive correlation between mean
HbA1c and eGFR decline (ρ≈0.21 — positive as expected, modest because
of the individual noise also deliberately injected).

Validated with NO changes whatsoever to the core or the rest of the
toolkit: formal contracts (Reproducibility, Semantic Preservation,
Temporality), `KMeansPhenotyper`, and `check_baseline_balance` (correctly
detects confounding by indication in insulin adoption — 13/16 Features
imbalanced). See `tests/test_diabetes_plugin.py` (11 tests) and
`examples/05_diabetes_toy_disease.py`.
