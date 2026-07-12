# BioSpace

**A computational representation framework for biological systems**

BioSpace is an open-source framework for modeling biological systems as explicit computational representations rather than collections of independent variables.

Instead of treating patients as rows in a table, BioSpace organizes observations into biological domains, producing structured representations that can be explored using machine learning, artificial intelligence, graph algorithms, topology, geometry, and systems medicine.

The project separates four fundamental concepts:

- Biological system
- Computational representation
- Representation geometry
- Inference algorithms

This separation allows the same biological representation to be analyzed using different computational methods without changing the underlying model.

---

# Motivation

Most biomedical AI pipelines follow the same architecture:

```text
Clinical Data
      ↓
Feature Matrix
      ↓
Machine Learning
      ↓
Prediction
```

Although highly successful, this approach usually leaves one important question unanswered:

> **What exactly is being represented before machine learning begins?**

BioSpace proposes that computational representation should become an explicit scientific object.

Instead of beginning with algorithms, BioSpace begins by representing biological systems.

```text
Biological System
        ↓
Computational Representation
        ↓
Representation Space
        ↓
Geometry
        ↓
Inference
```

This architecture allows representation to evolve independently from prediction algorithms.

---

# Core Philosophy

BioSpace is built around one central idea:

> **Representation comes before inference.**

Machine learning should operate on biologically meaningful representations rather than directly on heterogeneous clinical tables.

The framework therefore models:

- biological systems
- physiological domains
- observations
- measurements
- computational representations
- cohorts
- trajectories
- phenotypes

as first-class computational objects.

---

# Biological Systems

A biological system is the primary object of the framework.

Instead of representing patients as data frames,

```python
X = patient_features
```

BioSpace represents an actual biological entity.

```python
patient = BiologicalSystem("P001")
```

The biological system becomes responsible for organizing observations collected throughout time.

---

# Physiological Domains

Observations are grouped according to physiology rather than spreadsheet columns.

Example:

```text
Patient

├── Metabolism
│      ├── Glucose
│      ├── HbA1c
│      └── Insulin
│
├── Cardiovascular
│      ├── Blood Pressure
│      └── Heart Rate
│
├── Respiratory
│      ├── ODI
│      ├── AHI
│      └── SpO₂
│
└── Anthropometry
       ├── BMI
       ├── Weight
       └── Waist Circumference
```

Domains can be independently developed and combined into complete biological representations.

---

# Computational Representation

Each biological system is transformed into an explicit computational representation.

```python
representation = BioSpace.encode(patient)
```

The representation becomes the primary object consumed by downstream algorithms.

Algorithms no longer define the representation.

They operate on it.

---

# Representation Space

Every encoded biological system occupies a position inside a common representation space.

```text
          ● Patient A

     ● Patient B

               ● Patient C

 ● Patient D
```

Distances inside this space represent biological similarity rather than merely numerical similarity.

---

# Cohorts

BioSpace treats cohorts as mathematical objects.

Traditional approach:

```sql
SELECT *
FROM patients
WHERE HbA1c > 6.5
```

BioSpace:

```python
cohort = space.region(
    center=metabolic_cluster,
    radius=0.8
)
```

A cohort becomes a region of the representation space.

---

# Computational Phenotypes

Phenotypes emerge from the representation.

```python
phenotypes = space.discover_phenotypes()
```

Different clustering algorithms may explore the same representation.

For example:

- HDBSCAN
- K-Means
- Spectral Clustering
- Gaussian Mixture Models
- Graph Community Detection

The representation remains unchanged.

Only the inference operator changes.

---

# Temporal Representation

BioSpace natively supports longitudinal biological systems.

Instead of representing a patient as a single point,

the framework represents trajectories.

```python
patient.observe(
    date="2022",
    glucose=110
)

patient.observe(
    date="2023",
    glucose=128
)

patient.observe(
    date="2024",
    glucose=165
)
```

The resulting representation becomes

```text
R(B, t)
```

allowing longitudinal analysis, trajectory mining and disease progression studies.

---

# Scientific Contracts

BioSpace introduces the concept of scientific contracts.

Representations are automatically verified to ensure desirable mathematical properties.

Current contracts include:

- Continuity
- Temporal consistency
- Provenance
- Interoperability
- Causality
- Contract preservation

Rather than relying on implementation conventions, these properties are continuously tested.

---

# Architecture

```text
                    Biological System
                            │
                            ▼
                  Physiological Domains
                            │
                            ▼
                     Observations
                            │
                            ▼
                     Measurements
                            │
                            ▼
             Computational Representation
                            │
                            ▼
                Representation Space
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
         Cohorts       Phenotypes      Trajectories
            │               │                │
            └───────────────┼────────────────┘
                            ▼
                     AI Algorithms
```

---

# Current Applications

BioSpace has already been applied to several biomedical domains.

## Metabolism

Type 2 Diabetes

Datasets:

- NHANES
- UCI Diabetes Dataset

Applications include:

- computational phenotyping
- cohort construction
- metabolic representation
- exploratory analysis
- dashboards

---

## Sleep Medicine

Obstructive Sleep Apnea Syndrome (OSA)

Applications include:

- respiratory representation
- computational phenotypes
- transition analysis
- trajectory analysis
- patient similarity

---

# Artificial Intelligence Applications

Because BioSpace separates representation from inference, virtually any AI algorithm can operate on the representation.

Examples include:

## Machine Learning

- Random Forest
- XGBoost
- CatBoost
- LightGBM
- Logistic Regression
- SVM

## Deep Learning

- MLP
- TabNet
- FT-Transformer
- Neural Networks

## Representation Learning

- Autoencoders
- Variational Autoencoders
- Contrastive Learning
- Self-Supervised Learning

## Geometric Deep Learning

- Graph Neural Networks
- Graph Attention Networks
- GraphSAGE
- Graph Transformers

## Clustering

- HDBSCAN
- K-Means
- Spectral Clustering
- Gaussian Mixture Models

## Topological Data Analysis

- Persistent Homology
- Mapper
- Persistence Diagrams

## Survival Analysis

- Kaplan-Meier
- Cox Models
- Random Survival Forest

## Causal AI

- Structural Causal Models
- DAGs
- Counterfactual Analysis

## Digital Twins

- Intervention Simulation
- Treatment Response
- What-if Analysis

---

# Future Directions

BioSpace is currently evolving toward several advanced capabilities.

## Biomedical Ontologies

Automatic integration with:

- LOINC
- SNOMED CT
- ICD
- FHIR

Example:

```python
Glucose(
    value=98,
    loinc="2345-7"
)
```

---

## Knowledge Graphs

Physiological domains naturally induce semantic graphs connecting:

- biological systems
- domains
- observables
- measurements
- ontologies

---

## Multimodal Representation

Future versions aim to integrate:

- laboratory data
- medical imaging
- ECG
- polysomnography
- genomics
- transcriptomics
- proteomics
- wearable devices
- clinical notes

---

## Systems Medicine

BioSpace aims to represent multiple physiological systems simultaneously.

Examples:

- metabolism
- respiratory system
- cardiovascular system
- renal system
- endocrine system
- neurological system

allowing integrated computational models of human physiology.

---

# Scientific Vision

The long-term objective of BioSpace is not simply to provide another biomedical AI framework.

Instead, it seeks to establish computational representation as an independent scientific discipline.

Future research questions include:

- What is the best representation for a biological system?
- Which geometry best describes disease organization?
- How should biological similarity be defined?
- How should computational phenotypes be represented?
- Can cohorts become permanent mathematical objects?
- Can representation be validated independently from prediction?

---

# Documentation

Project documentation:

https://goldenluke.github.io/biospace/

---

# Citation

If you use BioSpace in your research, please cite the associated publications once available.

---

# Contributing

Contributions are welcome.

Possible areas include:

- biomedical ontologies
- representation learning
- graph learning
- geometric machine learning
- systems medicine
- causal inference
- multimodal representations
- digital twins
- visualization
- documentation
- benchmarking
- R implementation
- Julia implementation

Please open an issue before implementing major architectural changes.

---

# License

This project is released under the MIT License.

---

# Status

BioSpace is under active research and development.

The framework is evolving toward a general computational theory for biological system representation, where biological systems, representations, geometry, and inference become explicitly separated scientific concepts.
