# BioSpace

**A Framework for Computational Representation of Biological Systems**

BioSpace is an open-source framework that introduces a formal architecture for representing biological systems before applying Artificial Intelligence, statistical inference, or computational analysis.

Rather than modeling patients as rows in a table, BioSpace models them as structured biological systems composed of observations, physiological domains, representations, trajectories, and phenotypes.

The project separates **representation** from **inference**, allowing AI algorithms to operate on an explicit computational model instead of directly on heterogeneous clinical data.

---

# Theoretical Foundation

BioSpace is grounded in an ongoing theoretical framework that formalizes the computational representation of biological systems.

The theory introduces concepts such as:

- Biological systems as computational objects
- Representation functions
- Physiological domains
- Scientific contracts
- Computational phenotypes
- Representation spaces
- Computational geometry for biological systems
- Trajectories and longitudinal representations

The complete theoretical manuscript is available at:

https://goldenluke.github.io/patient-representation/

This manuscript provides the mathematical and conceptual foundations that motivate the architecture implemented in BioSpace.

# Motivation

Modern biomedical AI typically follows the pipeline:

```text
Patient
    ↓
DataFrame
    ↓
Machine Learning
    ↓
Prediction
```

Although successful, this approach leaves an important question unanswered:

> **What is the computational representation of a biological system?**

BioSpace proposes that this question should be answered before selecting algorithms.

The framework introduces an explicit representation layer between raw observations and computational inference.

```text
Biological System
        ↓
Observations
        ↓
Physiological Domains
        ↓
Representation
        ↓
Geometry
        ↓
Artificial Intelligence
        ↓
Inference
```

---

# Core Principles

BioSpace is built around a small number of architectural principles.

## Biological systems are first-class objects

Instead of treating patients as collections of variables, BioSpace models biological systems explicitly.

```python
patient = BiologicalSystem()
```

---

## Representation is independent of algorithms

Machine learning algorithms should consume representations rather than raw datasets.

```text
Biological System

↓

Representation

↓

Machine Learning
```

This separation allows different algorithms to operate on the same biological representation.

---

## Physiological domains organize knowledge

Clinical observations are grouped according to biological meaning.

Examples include:

- Anthropometry
- Respiratory Function
- Sleep Architecture
- Glucose Metabolism
- Cardiovascular Function
- Medication
- Symptoms
- Laboratory Tests

Domains become reusable building blocks for different diseases.

---

## Diseases reuse the same architecture

The framework does not implement diseases directly.

Instead, diseases are modeled by composing physiological domains.

Current implementations include:

- Obstructive Sleep Apnea (OSA)
- Diabetes Mellitus

Future implementations may include:

- Cardiovascular diseases
- COPD
- Cancer
- Neurological disorders
- Chronic kidney disease

---

# Architecture

```text
BiologicalSystem
        │
        ├──────── Observations
        │
        ├──────── Measurements
        │
        ├──────── Domains
        │
        ├──────── Representation
        │
        ├──────── Phenotypes
        │
        ├──────── Trajectories
        │
        └──────── Computational Geometry
```

---

# Main Components

## BiologicalSystem

Represents an individual biological system.

It aggregates all observations across time.

---

## Observation

Represents a measurable biological quantity.

Examples:

- Glucose
- HbA1c
- SpO₂
- Blood Pressure
- BMI

---

## Measurement

Represents an observation obtained at a specific point in time.

Each measurement preserves provenance information.

---

## Physiological Domain

Organizes biologically related observations.

Examples:

```text
Respiratory Domain

├── AHI
├── ODI
├── SpO₂
└── Hypoxic Burden
```

---

## Representation

Transforms biological systems into computational objects suitable for inference.

Formally,

```
R : B → X
```

where

- **B** is a biological system
- **X** is the representation space

---

## Phenotype

Represents a region of the representation space rather than a predefined diagnostic label.

---

## Trajectory

Represents the temporal evolution of a biological system.

```
Γ(t)
```

---

# Scientific Contracts

One of the central ideas of BioSpace is the concept of **Scientific Contracts**.

Unlike traditional unit tests, scientific contracts verify mathematical and scientific properties of the representation.

Current contracts include:

- Continuity
- Traceability
- Temporality
- Interoperability

These contracts are continuously executed during development to ensure that implementation remains consistent with the theoretical model.

---

# Current Applications

## Obstructive Sleep Apnea

The OSA implementation models domains such as:

- Anthropometry
- Respiratory Events
- Hypoxemia
- Symptoms
- Comorbidities
- Treatment

Interactive dashboard:

https://biospace-saos.streamlit.app/

---

## Diabetes

The diabetes implementation reuses the same architecture while replacing only the physiological domains.

Interactive dashboard:

https://biospace-diabetes.streamlit.app/

---

# Future Roadmap

The long-term goal is to build a complete computational infrastructure for biological systems.

Planned features include:

## Biomedical Ontologies

Support for:

- LOINC
- SNOMED CT
- ICD-10 / ICD-11
- HL7 FHIR
- openEHR

---

## Automatic Ontology Generation

Generate biomedical ontologies directly from computational models.

Supported formats:

- RDF
- OWL
- JSON-LD

---

## Knowledge Graphs

Represent biological systems as semantic graphs.

Applications:

- Graph Neural Networks
- Graph Embeddings
- Clinical Reasoning

---

## Representation Learning

Learn latent biological representations using:

- Autoencoders
- Variational Autoencoders
- Contrastive Learning
- Self-Supervised Learning

---

## Computational Geometry

Explicit geometric structures for biological representation spaces.

Potential geometries include:

- Euclidean
- Mahalanobis
- Information Geometry
- Learned Geometries

---

## Digital Twins

Simulation of hypothetical interventions using computational representations.

Examples:

- Weight loss
- Medication changes
- Disease progression
- Treatment response

---

## Multimodal Systems

Native support for multiple data modalities.

Examples:

- Laboratory tests
- Medical imaging
- Clinical text
- Wearables
- Omics
- Time series

---

# Scientific Vision

BioSpace aims to establish a new computational layer between biological observations and artificial intelligence.

Rather than asking:

> Which algorithm performs best?

BioSpace asks:

> How should biological systems be represented computationally?

This shift places computational representation at the center of biomedical AI.

---

# Publications

Current research topics include:

- Computational Representation of Biological Systems
- Scientific Contracts for Biomedical AI
- Patient Representation and Phenotype Stability
- Geometry of Biological Representation Spaces
- Representation Learning for Clinical Systems
- Computational Phenotyping
- Biomedical Knowledge Graphs

---

# Contributing

BioSpace is an open-source research project.

Contributions are welcome in areas including:

- Python
- R
- Biomedical Informatics
- Machine Learning
- Computational Geometry
- Knowledge Graphs
- Representation Learning
- Biomedical Ontologies
- Clinical Data Modeling
- Software Engineering

Please open an issue before submitting major architectural changes.

---

# Citation

If you use BioSpace in your research, please cite the associated theoretical publications once available.

---

# License

This project is released under the MIT License.

---

# Contact

Contributions, discussions, collaborations, and research partnerships are welcome.

BioSpace is developed as an open scientific framework for computational representation of biological systems.
