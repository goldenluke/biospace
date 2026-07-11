# Mathematical Foundations of the Computational Representation of Biological Systems — A Meta-Model for Computational Precision Medicine

## 1. Introduction

Contemporary medicine produces an unprecedented amount of data. Electronic health records, laboratory tests, continuous physiological signals, medical imaging, wearable devices, and omics data now describe the patient at a resolution previously unattainable. In parallel, advances in machine learning and artificial intelligence have made it possible to exploit this growing volume of information for tasks such as diagnosis, risk stratification, phenotype discovery, and clinical decision support.

Despite this progress, there is a fundamental aspect that remains surprisingly underdiscussed: the object on which all these algorithms actually operate.

Although it is common to say that algorithms "learn from patients," this statement is, from a computational standpoint, incorrect. Algorithms never directly observe a patient. They observe a computational representation constructed from clinical, laboratory, functional, or imaging observations. In other words, the true input object of the algorithms is not the biological system itself, but a mathematical abstraction of that system.

This distinction, seemingly subtle, carries profound scientific implications.

In practice, the construction of this representation is rarely treated as an independent problem. In much of the literature, it emerges as a byproduct of preprocessing and feature-engineering steps, often defined by the specific needs of the algorithm employed or by data availability. The result is that different studies on the same disease may use completely distinct representations of the patient, hindering reproducibility, comparison between methods, model reuse, and integration across different data modalities.

Consider, for example, Obstructive Sleep Apnea Syndrome (OSAS). One study may represent each individual using only the Apnea-Hypopnea Index (AHI). Another may use age, body mass index (BMI), and minimum oxygen saturation. A third may incorporate hundreds of variables derived from polysomnography. Although all claim to study the same disease, each work mathematically describes a different patient. The difference does not necessarily lie in the algorithm used, but in the very computational definition of the object being analyzed.

This situation reveals a conceptual limitation of contemporary computational medicine. While fields such as computer vision, natural language processing, and representation learning have devoted decades to the formal study of representation structures, medical informatics has focused predominantly on algorithm development, implicitly assuming that patient representation was already a solved problem.

However, computational representation constitutes a scientific problem in its own right. Before asking which algorithm produces the best performance, it is necessary to answer more fundamental questions.

What is a patient from a computational point of view?

How should different clinical observations be organized to preserve their physiological semantics?

What mathematical properties must a representation satisfy to be considered clinically valid?

How can different data modalities be integrated without losing conceptual consistency?

These questions precede any machine-learning technique. They belong to the domain of the computational modeling of biological systems.

This work argues that the absence of a formal theory of the computational representation of the patient constitutes a structural gap in computational medicine. As a response, a formal meta-model is proposed, capable of explicitly separating four conceptual levels that are frequently treated as a single problem:

the real biological system; its observations; its computational representation; the inference algorithms responsible for the inference.

This separation profoundly changes the traditional architecture of artificial intelligence systems in healthcare. Instead of treating representation as an auxiliary preprocessing step, it comes to occupy a central position in computational modeling. Algorithms cease to operate directly on heterogeneous data and instead act on a formal representation built according to explicit principles that are independent of the learning technique used.

The proposal presented in this article is not a new machine-learning algorithm, nor a new phenotype-discovery technique. Nor is it restricted to a specific disease or a particular data modality. Its objective is to define a conceptual language capable of describing, in a uniform way, how biological systems can be observed, semantically organized, and computationally represented.

To this end, a meta-model is introduced composed of fundamental entities — Biological System, Observation, Semantic Domain, Representation, Representation Space, Geometry, Trajectory, Phenotype, and Cohort — together with their relations, properties, and formal constraints. The computational implementation thus becomes a consequence of a representation theory, rather than its starting point.

This perspective brings computational medicine closer to established paradigms in computer science. Just as the relational model established a formal theory for data organization and UML defined a common language for software system modeling, it is proposed here that the computational representation of the patient likewise be understood as an independent conceptual layer, capable of sustaining different mathematical implementations and different inference strategies.

Although the development of this theory was initially motivated by the modeling of patients with Obstructive Sleep Apnea Syndrome, its formulation is deliberately independent of the disease studied. The central hypothesis is that the principles presented here can be generalized to any biological system amenable to observation, constituting a unifying foundation for future applications in precision medicine, computational epidemiology, phenotype discovery, intelligent decision-support systems, and multimodal integration of clinical data.

Ultimately, this work proposes a shift in perspective. Rather than asking how to build increasingly sophisticated algorithms to analyze patients, it proposes asking how to mathematically represent a patient so that any algorithm can operate on that representation in a consistent, reproducible, and physiologically grounded way. It is this conceptual inversion that underlies the meta-model presented in the following sections.

## 2. Motivation: The Problem of Computational Representation

Most modern artificial-intelligence architectures in healthcare can be summarized by a relatively simple flow. Clinical data are collected, undergo cleaning and feature-engineering steps, are organized into a numerical matrix, and are subsequently used by machine-learning algorithms to produce diagnoses, predictions, or clusterings.

Formally, this process can be represented as:

$$Data → Preprocessing → Feature Engineering → Machine Learning → Inference.$$

Although this architecture has produced important advances, it hides a fundamental conceptual problem: the mathematical object on which the algorithm operates is never explicitly defined.

In practice, it is implicitly assumed that a patient can be represented by an attribute vector

$$x = (x₁, x₂, …, xₙ) ∈ ℝⁿ,$$

where each component corresponds to a clinical variable.

For example, for a patient with Obstructive Sleep Apnea Syndrome (OSAS), a conventional representation may be written as

$$x = (AHI, BMI, Age, minimum SpO₂, ODI, ESS).$$

From this representation, any learning algorithm can be defined as a function

$$A : ℝⁿ → Y,$$

where Y represents the set of possible outputs, such as diagnostic classes, probabilities, phenotypes, or prognostic values.

However, this formulation contains a rarely discussed assumption: it is assumed that the vector x adequately represents the patient.

In reality, it represents only an arbitrary selection of attributes made by the researcher.

In other words,

$$x ≠ P,$$

where P represents the patient as a biological system.

More precisely,

$$x = f(P),$$

where f is a representation function normally constructed during the feature-engineering step.

This distinction has important consequences.

Consider two independent studies on OSAS.

The first represents each patient using only the Apnea-Hypopnea Index (AHI):

$$R₁(P) = (AHI).$$

The second adds age and body mass index:

$$R₂(P) = (AHI, BMI, Age).$$

A third uses dozens of variables derived from polysomnography:

$$R₃(P) = (AHI, ODI, minimum SpO₂, mean SpO₂, T90, Hypoxic Burden, Microarousals, …).$$

Although all claim to study the same disease, each study uses a different computational representation of the patient.

Consequently,

$$R₁(P) ≠ R₂(P) ≠ R₃(P).$$

This difference does not stem from the algorithm used.

It stems from the mathematical definition of the object of study itself.

Therefore, two models trained on different representations are not, strictly speaking, learning about the same patient.

This observation leads to an even deeper consequence.

Consider two distinct algorithms, A₁ and A₂, applied to two distinct representations. One obtains

$$A₁(R₁(P)) and A₂(R₂(P)).$$

When their performances are compared, the observed difference is normally attributed to the algorithms' capacity.

However,

$$A₁(R₁(P)) ≠ A₂(R₂(P))$$

does not necessarily imply

$$A₁ ≠ A₂.$$

The difference may lie exclusively in

$$R₁ ≠ R₂.$$

This means that countless studies comparing different algorithms may, in reality, be comparing different representations.

The problem becomes even more evident when new data modalities are incorporated. Suppose a study begins to use full polysomnography signals. The representation ceases to be an attribute vector and comes to include time series,

$$s(t) ∈ S,$$

where S represents a space of physiological signals. Likewise, imaging exams belong to another space,

$$I ∈ 𝓘,$$

while clinical text belongs to a linguistic space,

$$L ∈ 𝓛.$$

The traditional representation based exclusively on vectors becomes incapable of naturally integrating these different modalities. The result is that each new type of data requires the complete reconstruction of the computational infrastructure.

This scenario reveals a conceptual limitation of the currently predominant architecture. The literature concentrates its efforts on developing inference functions,

$$A : R(P) → Y,$$

but rarely investigates the function itself

$$R : P → X,$$

responsible for transforming a biological system into a mathematical object.

In this work it is argued that this function should occupy a central position in computational medicine. More specifically, it is proposed that computational representation constitute an independent layer between the biological system and the inference algorithms.

Thus, the traditional flow

$$P → A → Y$$

is replaced by

$$P → R → X → A → Y,$$

where P represents the biological system; R represents the representation function; X represents the representation space; A represents any inference algorithm; Y represents the produced result.

This reformulation profoundly changes the conceptual organization of computational medicine. The algorithm ceases to be the central element of the architecture. Representation comes to occupy that position. As a consequence, it becomes possible to investigate properties of the representation independently of the algorithms that will later act on it, establishing a proper conceptual layer for the computational modeling of patients.

This shift in perspective constitutes the main motivation for the meta-model proposed in the following sections.

## 3. Central Hypothesis and Fundamental Principles

The main hypothesis of this work is that the computational representation of the patient constitutes a scientific problem independent of machine-learning algorithms. This statement may seem simple, but it profoundly changes how computational medicine is traditionally organized.

In the contemporary literature, practically all efforts concentrate on building inference functions. Algorithms are developed to predict clinical outcomes, identify phenotypes, estimate risks, or recommend treatments. However, these algorithms operate on a previously constructed representation of the patient, whose definition is rarely formalized.

In other words, the literature devotes great attention to the function

$$A : X → Y,$$

where X represents the space used by the algorithm and Y represents the produced output.

Yet the prior problem remains practically unexplored:

$$R : P → X,$$

that is, how to transform a biological system into a mathematical object while preserving its physiological structure.

In this work it is proposed to invert this perspective. Instead of treating representation as a consequence of the algorithm, it is proposed to treat it as an independent layer on which any algorithm may operate.

Thus, the logical chain ceases to be

$$P → A → Y,$$

and becomes

$$P → R → X → A → Y.$$

This shift establishes an explicit separation between four conceptual levels frequently conflated in the literature. The first level corresponds to the biological system, that is, to the patient as a real organism. The second corresponds to the observations, produced by exams, sensors, instruments, and clinical assessments. The third corresponds to the computational representation, responsible for mathematically organizing these observations. The fourth corresponds to computational inference, carried out by statistical or machine-learning algorithms.

Formally,

$$Biological System → Observations → Representation → Inference.$$

The central hypothesis of this theory can then be expressed as follows.

The validity of a computational model depends both on the quality of the representation and on the quality of the inference algorithm.

Mathematically, consider a biological system

$$P ∈ 𝒫,$$

where 𝒫 represents the set of all possible patients.

A representation function is defined as

$$R : P → X,$$

where X represents the computational representation space.

Subsequently, any algorithm

$$A : X → Y$$

operates exclusively on elements of X. Note that A never receives P directly. It receives only R(P). Therefore, the complete pipeline can be described by

$$Y = A(R(P)).$$

This equation synthesizes the entire proposed hypothesis. The result produced by the algorithm depends simultaneously on two independent functions: R, the representation function, and A, the inference function. Traditionally, the literature concentrates on optimizing A, keeping R practically fixed. However, it is argued in this work that changes in R can produce impacts as significant as changes in A. Consequently, representation becomes an independent object of study.

### 3.1 The Principle of Algorithmic Independence

The first consequence of this hypothesis is the Principle of Algorithmic Independence. This principle states that a computational representation must not depend on the algorithm subsequently used.

Formally, let A₁ and A₂ be any two algorithms. A valid representation must satisfy

$$R(P) = R(P),$$

regardless of the choice between A₁ or A₂. In other words, the representation is built before inference. This means that different algorithms can operate on exactly the same represented patient. Thus, it becomes possible to compare algorithms without altering the mathematical object analyzed.

### 3.2 The Principle of Compositionality

Biological systems are composed of different physiological processes. Consequently, a computational representation must also possess a compositional nature.

A set of clinical domains is defined,

$$D = {D₁, D₂, …, Dₙ},$$

where each domain represents a distinct physiological perspective. For example, Respiratory, Hypoxic, Cardiovascular, Metabolic, Symptoms, and so on.

The complete representation is no longer built directly. It is instead composed of the individual representations of each domain. Formally,

$$R(P) = (R₁(D₁), R₂(D₂), …, Rₙ(Dₙ)).$$

This decomposition reduces coupling between components and allows new domains to be added without rebuilding the entire architecture.

### 3.3 The Principle of Semantic Preservation

Every computational representation must preserve the clinical meaning of the observations it represents. Let S₁ and S₂ be two distinct physiological states. A consistent representation must satisfy

$$S₁ ≠ S₂ ⟹ R(S₁) ≠ R(S₂).$$

This property prevents clinically distinct states from being represented by the same mathematical structure. In other words, the representation must be sufficiently expressive to distinguish relevant physiological conditions.

### 3.4 The Principle of Continuity

Another desirable property is continuity. Small physiological changes should not produce abrupt changes in the representation.

Formally, for any patients P₁ and P₂, if

$$d_P(P₁, P₂) → 0,$$

then it must follow that

$$d_X(R(P₁), R(P₂)) → 0,$$

where d_P represents a measure of physiological difference, and d_X a distance defined over the representation space. This property guarantees robustness to experimental noise and small clinical variations.

### 3.5 The Principle of Extensibility

Medicine is in constant evolution. New biomarkers, new imaging modalities, new wearable devices, and new laboratory techniques continually emerge. Therefore, a representation cannot depend on a fixed set of variables.

Formally, let D = {D₁, …, Dₙ}. Introducing a new domain Dₙ₊₁ must produce

$$D′ = D ∪ {Dₙ₊₁},$$

without invalidating the previously built representation. That is, the architecture must admit incremental expansion while preserving backward compatibility.

### 3.6 The Principle of Reuse

A direct consequence of the previous principles is that the representation ceases to belong to a specific disease. It becomes a computational infrastructure.

Let Θ be the set of modeled diseases. For each disease θᵢ ∈ Θ, only a specific set of domains is defined, keeping the entire representation infrastructure unchanged. Formally,

$$Rθᵢ : P → X$$

shares exactly the same meta-model, changing only the composition of the clinical domains. This property turns computational representation into a reusable layer across different medical specialties.

The principles presented in this section constitute the conceptual hypothesis of the proposed theory. They do not yet define a specific mathematical implementation, nor do they impose the use of vector spaces, graphs, tensors, or any other structure. Their purpose is to establish the fundamental properties that any computational representation of the patient must satisfy, independent of the mathematics employed or the inference algorithm subsequently used.

From these principles, it becomes possible to introduce, in the following section, the formal meta-model responsible for defining the fundamental entities of the computational representation of biological systems.

## 4. The Formal Meta-Model of the Computational Representation of the Patient

The previous sections argued that the computational representation of the patient constitutes a scientific problem independent of machine-learning algorithms. However, stating that representation should occupy its own layer is not sufficient. It is necessary to formally define which objects compose this layer, which relations are permitted among them, and which properties must be preserved by any implementation.

This is precisely the objective of a meta-model.

In Computer Science, a meta-model does not implement a computational system. It defines the language used to describe it. The relational model defines concepts such as relations, attributes, and keys before the implementation of databases. UML defines classes, associations, and inheritance before the implementation of software. Likewise, a meta-model is proposed here capable of defining the fundamental elements of the computational representation of biological systems.

The proposed meta-model does not describe a specific disease, a learning algorithm, or a particular mathematical structure. It describes only the minimal concepts necessary to transform a biological system into a computational object amenable to analysis.

Formally, the meta-model is defined as the tuple

$$M = (B, O, D, R, X, G, Γ, F, C),$$

where B represents the set of biological systems; O represents the set of observations; D represents the semantic domains; R represents the computational representations; X represents the representation spaces; G represents the admissible geometric structures; Γ represents the longitudinal trajectories; F represents the phenotypes; C represents the cohorts.

These entities constitute the primitive elements of the theory. All other computational structures derive from them.

### 4.1 Biological System

The first element of the meta-model is the biological system itself. A biological system is defined as

$$B = (E, T),$$

where E represents the physiological state of the organism and T represents its temporal evolution.

Note that this definition precedes any clinical observation. The biological system exists independently of exams, sensors, or computational models. Consequently,

$$B ∉ X,$$

that is, the biological system does not belong to the computational space. It belongs to the physical world.

### 4.2 Observations

Since the physiological state cannot be observed directly, the Observation entity is introduced. Each observation corresponds to an operator that extracts partial information from the biological system. Formally,

$$Oᵢ : B → Mᵢ,$$

where Mᵢ represents the measurement space produced by the corresponding instrument. Examples include polysomnography, electrocardiogram, laboratory tests, magnetic resonance imaging, wearable devices, and clinical questionnaires.

Note that observations are not data. They are measurement operators. Different observations may measure distinct aspects of the same biological system.

### 4.3 Semantic Domains

Individual observations do not possess clinical meaning in isolation. They need to be organized according to their physiological interpretation. A semantic domain is thus defined as

$$Dᵢ = (Σᵢ, Oᵢ),$$

where Σᵢ represents the clinical semantics of the domain and Oᵢ represents the set of associated observations.

For example, the respiratory domain can be defined as

$$D_Resp = (Σ_Resp, {AHI, RDI, Flow, Apneas, Hypopneas}).$$

Likewise, the hypoxia domain can be written as

$$D_Hyp = (Σ_Hyp, {ODI, SpO₂, T90, Hypoxic Burden}).$$

Note that domains are defined by the clinical meaning of the observations, and not by their mathematical structure.

### 4.4 Representations

Once domains are defined, it becomes possible to represent them mathematically. Each domain possesses a representation operator

$$φᵢ : Dᵢ → Xᵢ,$$

where Xᵢ is a mathematical space appropriate to the domain. This space can assume different natures. For example,

$$Xᵢ ∈ {ℝⁿ, H, G, T, E},$$

representing, respectively, vector spaces, functional spaces, graphs, tensors, and embeddings.

The theory does not impose which mathematical structure must be used. It requires only that a consistent representation operator exist.

### 4.5 Global Computational Representation

The complete representation of a biological system is defined by the composition of the individual representations. Formally,

$$R : B → X,$$

such that

$$R(B) = (φ₁(D₁), φ₂(D₂), …, φₙ(Dₙ)).$$

This equation represents the core of the meta-model. Note that the patient ceases to be described by a single attribute vector. The patient becomes defined as a composition of multiple physiological domains.

### 4.6 Representation Space

The set of all possible representations defines the computational space. Formally,

$$X = X₁ × X₂ × ⋯ × Xₙ.$$

At this point, each patient occupies a unique position in X. However, this space does not yet possess any notion of distance or proximity. It represents only a structural organization of the representations.

### 4.7 Geometry

To allow comparison between patients, a geometric structure is introduced. An operator is defined,

$$G : X → M,$$

where M represents the set of admissible geometries. This formulation is purposefully generic. An implementation may use Euclidean distance, Mahalanobis, Wasserstein, information geometry, Riemannian manifolds, or hyperbolic spaces. All remain compatible with the meta-model. Geometry ceases to be a fixed property of the theory. It becomes a characteristic of the implementation.

### 4.8 Trajectories

Since biological systems evolve continuously, the representation must also incorporate temporality. A trajectory is defined as

$$Γ : T → X.$$

Each instant t ∈ T is associated with a specific representation, such that

$$Γ(t) = R(B_t).$$

The fundamental object ceases to be only the patient. It becomes the patient's trajectory across the representation space.

### 4.9 Phenotypes

Traditionally, computational phenotypes are treated as outputs of clustering algorithms. In this theory, a different interpretation is adopted. Phenotypes are defined as structured regions of the representation space. Formally,

$$F ⊆ X.$$

Learning algorithms cease to create phenotypes. They merely estimate subsets belonging to X. This distinction shifts the scientific contribution from the algorithm to the structure of the representation.

### 4.10 Cohorts

Finally, a cohort is defined as a set of trajectories. Formally,

$$C = {Γ₁, Γ₂, …, Γₙ}.$$

Note that, in this formulation, a cohort ceases to be merely a collection of patients. It comes to represent the longitudinal evolution of multiple biological systems within the same representation space. Each new exam adds only a new point to the existing trajectory, without the need to reconstruct the database.

### 4.11 Fundamental Relations of the Meta-Model

The entities defined above relate through a single causal chain,

$$B → O → D → R → X → G → Γ → F → C.$$

This chain represents the backbone of the theory. No implementation should violate this organization. In particular, learning algorithms must never operate directly on observations or biological systems. All computational inference must occur exclusively on representations belonging to the space X.

This separation establishes a clear distinction between observation, representation, and inference, allowing different mathematical implementations to coexist within the same conceptual structure.

The meta-model thus becomes a formal language for the computational representation of biological systems, independent of the disease studied, the data modality used, or the learning algorithm subsequently employed.

## 5. Formal Contracts of the Meta-Model

The entities presented in the previous section define the structure of the meta-model. However, defining entities is not sufficient to characterize a formal theory. It is necessary to specify which properties every implementation must satisfy.

In Software Engineering, these properties are often described by contracts, that is, conditions that must remain true regardless of the implementation used.

In this work, a contract represents a mathematical or semantic property that every computational representation of the patient must preserve. While the meta-model defines what exists, the contracts define how these entities must behave.

Formally, the set of contracts is defined as

$$K = {K₁, K₂, …, Kₙ},$$

where each Kᵢ represents a mandatory property of the computational representation. These contracts do not depend on the disease studied nor on the mathematical structure used to represent each domain. They represent universal constraints of the theory.

### 5.1 Traceability Contract

No representation can exist without its origin being identifiable. All information used by the representation must have known provenance. Formally, for every representation R(B), there must exist a set of observations

$$O = {O₁, O₂, …, Oₙ},$$

such that

$$R(B) ⇐ O.$$

The relation ⇐ denotes provenance dependence. This means that any component of the representation can be traced back to the observation responsible for its construction. Consequently, no computational attribute may arise without an explicit chain of derivation.

### 5.2 Semantic Preservation Contract

A computational representation must preserve the clinical meaning of the biological system. Consider two distinct physiological states, S₁ and S₂. If these states represent different clinical conditions, then their representations must also be different. Formally,

$$S₁ ≠ S₂ ⟹ R(S₁) ≠ R(S₂).$$

This contract prevents semantically distinct physiological states from being collapsed into the same representation. Note that this principle does not require a unique representation. It requires only that clinically relevant differences remain distinguishable after the representation process.

### 5.3 Compositionality Contract

No representation should be built directly from the biological system. It must result from the composition of independent domain representations. Let D = {D₁, D₂, …, Dₙ}. There must then exist a composition function Ψ such that

$$R(B) = Ψ(R₁(D₁), R₂(D₂), …, Rₙ(Dₙ)).$$

Note that R does not depend directly on B, but on the composition of the partial representations. This contract reduces coupling between domains and allows independent evolution of each physiological component.

### 5.4 Continuity Contract

Computational representations must vary continuously in response to small physiological changes. Let B₁ and B₂ be two biologically similar states. There must then exist a constant L > 0 such that

$$d_X(R(B₁), R(B₂)) ≤ L · d_B(B₁, B₂).$$

This property characterizes a Lipschitz-type continuity condition. It guarantees that small experimental variations, measurement errors, or instrumental noise do not produce arbitrarily large changes in the representation. This contract makes the system robust to the inevitable uncertainties present in clinical data.

### 5.5 Extensibility Contract

A computational representation must admit new domains without invalidating previously built representations. Consider D = {D₁, …, Dₙ}. If a new domain Dₙ₊₁ is incorporated, the new representation must satisfy

$$R′ = R ⊕ Rₙ₊₁,$$

where ⊕ represents an extension operator. This operator must preserve all previously existing components. Formally,

$$R ⊆ R′.$$

Thus, the architecture's evolution does not require complete reconstruction of the representations.

### 5.6 Algorithmic Independence Contract

This may be the most important contract of the theory. The computational representation cannot depend on the algorithm that will subsequently be used. Consider two algorithms, A₁ and A₂. It must hold that

$$R(B) = R(B),$$

regardless of the choice between A₁ or A₂. Consequently, the algorithm acts exclusively on R(B), never on the biological system. Formally,

$$Yᵢ = Aᵢ(R(B)).$$

Note that R remains constant, while Aᵢ may vary. This allows algorithms to be compared while preserving exactly the same mathematical object.

### 5.7 Temporality Contract

Biological systems evolve continuously. Therefore, every representation must preserve its temporal organization. A trajectory is defined as

$$Γ : T → X,$$

where T represents the temporal domain. Each observation produces an update of the trajectory, not a new independent representation. Formally, for a sequence of observations O(t₁), O(t₂), …, O(tₙ), one obtains

$$Γ = {R(B, t₁), R(B, t₂), …, R(B, tₙ)}.$$

This contract turns longitudinality into an intrinsic property of the representation.

### 5.8 Reproducibility Contract

Given the same observations, a representation must be deterministic. If

$$O₁ = O₂,$$

then

$$R(O₁) = R(O₂).$$

This contract guarantees that different researchers obtain exactly the same representation when using the same set of observations. The representation thus becomes reproducible and auditable.

### 5.9 Versionability Contract

Since clinical systems evolve continuously, the representation must possess a version history. R⁽ᵏ⁾ is defined as version k of the representation. Every update produces R⁽ᵏ⁺¹⁾, preserving prior versions. Formally,

$$R⁽ᵏ⁾ ≺ R⁽ᵏ⁺¹⁾,$$

where ≺ represents the temporal precedence relation. This contract allows historical analyses to be reproduced even after the architecture has evolved.

### 5.10 Interoperability Contract

Computational representations must be independent of the origin of the data. Let O_A and O_B be observations from different manufacturers, but semantically equivalent. There must exist a harmonization operator H such that

$$H(O_A) = H(O_B).$$

Consequently, the computational representation becomes independent of the equipment used for data acquisition. This property is particularly important in multicenter applications and longitudinal studies involving different diagnostic platforms.

### 5.11 Synthesis of the Contracts

The contracts presented in this section establish the minimal properties that any implementation compatible with the meta-model must respect. While the meta-model defines the fundamental entities of the computational representation, the contracts define the invariant properties that guarantee consistency, auditability, continuity, reuse, and algorithmic independence.

These contracts play a role similar to axioms in a mathematical theory. They do not specify how a representation must be implemented. Instead, they define the formal criteria that allow one to decide whether a given implementation is compatible with the proposed theory.

In the following sections, these contracts will serve as the basis for introducing the organization of representation spaces and their geometric structure, allowing learning algorithms to operate on physiologically consistent and mathematically well-defined representations.

## 6. Representation Spaces

The previous sections established the fundamental elements of the meta-model and the contracts that every implementation must satisfy. However, an essential question remains: where do computational representations exist?

Every representation produced by the function

$$R : B → X$$

must belong to some mathematical space. In the traditional literature, it is implicitly assumed that this space corresponds to a Euclidean vector space,

$$X = ℝⁿ,$$

in which each patient is described by an attribute vector. This hypothesis is sufficiently convenient for applying most classical machine-learning algorithms. However, it presents important limitations when one wishes to represent complex biological organisms.

First, different clinical domains possess distinct mathematical natures. Second, different data modalities can hardly be described by a single vector structure. Third, physiology itself is not naturally organized as a numerical vector.

For these reasons, it is proposed to understand the representation space as a composite structure, formed by the integration of different specialized spaces.

### 6.1 Individual Representation Spaces

Each semantic domain has its own representation. Formally,

$$φᵢ : Dᵢ → Xᵢ,$$

where Xᵢ represents the mathematical space associated with domain Dᵢ. Note that the meta-model does not impose any restriction on the nature of Xᵢ. Each domain may use the mathematical structure most appropriate to preserve its semantics.

For example, the respiratory domain may be represented by a functional space

$$X_Resp = F,$$

where each element corresponds to a respiratory time series. Likewise, the cardiovascular domain may be described by heart-rate time series,

$$X_Cardio = T.$$

A comorbidity domain may take the form of a discrete set,

$$X_Comorb = 2^C,$$

where C represents the set of possible diseases. Medical images may be represented by tensors,

$$X_Image = ℝ^(H×W×C),$$

while clinical text may be represented by embeddings,

$$X_Text = ℝ^d.$$

This freedom is a direct consequence of the Principle of Semantic Preservation. The theory does not seek to reduce all domains to the same type of mathematical object. It seeks to preserve the nature of each domain.

### 6.2 Product of Spaces

Once the individual spaces are defined, they must be integrated. Formally, the global space is defined as

$$X = X₁ × X₂ × ⋯ × Xₙ.$$

Each patient is then described by

$$R(B) = (x₁, x₂, …, xₙ),$$

where xᵢ ∈ Xᵢ. Note that each component belongs to its own space. There is no requirement that all share the same mathematical structure. The Cartesian product only organizes these representations into a single structure.

### 6.3 The Representation Space as a Computational Object

Traditionally, algorithms receive numerical matrices directly. In this theory, the computational object ceases to be a matrix. The object becomes the representation space itself. Formally,

$$X = {R(B₁), R(B₂), …, R(Bₙ)}.$$

Each element of X corresponds to a complete representation of a biological system. Thus, algorithms cease to operate on tables. They come to operate on elements belonging to X. This distinction may seem subtle, but it completely changes the computational architecture.

### 6.4 Independence Between Spaces

Another important consequence of the meta-model is that the individual spaces remain decoupled. Formally, for two domains Dᵢ and Dⱼ, their representations are independent,

$$Xᵢ ≠ Xⱼ, and φᵢ ≠ φⱼ.$$

This means that changes in the representation of one domain do not require reconstructing the representations of the others. For example, a new respiratory-signal processing technique may modify X_Resp without producing any change in X_Cardio or X_Metab. This property drastically reduces coupling between architecture components.

### 6.5 Hierarchical Representations

Not all domains possess the same level of abstraction. Often a representation can be built upon another. A sequence of operators is defined,

$$φᵢ⁽¹⁾, φᵢ⁽²⁾, …, φᵢ⁽ᵐ⁾,$$

such that

$$Dᵢ →φᵢ⁽¹⁾ Xᵢ⁽¹⁾ →φᵢ⁽²⁾ Xᵢ⁽²⁾ → ⋯ →φᵢ⁽ᵐ⁾ Xᵢ⁽ᵐ⁾.$$

This structure allows different levels of abstraction to be represented. For example, the raw respiratory flow may originate respiratory events, which subsequently originate respiratory indices, which finally originate a representation vector. Formally,

$$Signal → Events → Indicators → Representation.$$

Each transformation remains explicitly defined, guaranteeing complete traceability.

### 6.6 Latent Domains

Not all physiological domains are directly observable. Many biological states influence multiple observations without ever being directly measured. Consider, for example, systemic inflammatory state. There is no single observation capable of completely describing it. Instead, it manifests through changes distributed across different laboratory tests, physiological parameters, and clinical signals.

A latent domain, D_L, is thus defined, for which there is no direct observation operator. Its representation depends on an inference problem. Formally,

$$φ_L : O → X_L.$$

Note that, unlike the other domains, φ_L does not represent a projection, but a computational reconstruction of a non-observable physiological state. This property significantly expands the capacity of the meta-model, allowing it to incorporate physiological concepts that have no direct measures.

### 6.7 Adaptive Spaces

Another important consequence of the theory is that the representation space need not remain static. As new domains are incorporated, new spaces can be added. Formally, if

$$X = X₁ × ⋯ × Xₙ,$$

the introduction of a new domain produces

$$X′ = X × Xₙ₊₁.$$

Note that X ⊂ X′. Thus, the space evolves continuously, preserving compatibility with prior representations. This property follows directly from the Extensibility Contract.

### 6.8 Shared Spaces

Although each domain has its own representation, some problems require integration across domains. A composition operator is defined,

$$Ψ : X₁ × X₂ × ⋯ × Xₙ → X_c,$$

where X_c represents a shared space. This space can be used for phenotype discovery, supervised learning, causal inference, or decision-support systems. Note that X_c does not replace the individual spaces. It represents only an integrated projection, built from them.

### 6.9 Architectural Consequences

The explicit introduction of representation spaces profoundly changes the organization of computational medicine systems. Instead of storing patients as rows of a table, the system organizes representations within a structured mathematical space. Consequently, algorithms cease to operate directly on clinical variables. They come to operate on elements belonging to X. This change completely decouples representation and inference. New domains, new data modalities, and new mathematical structures can be incorporated without altering the general architecture of the system.

More importantly, the theory ceases to depend on a specific vector representation. Any mathematical structure capable of satisfying the contracts defined in the previous section becomes a valid implementation of the meta-model.

In the following sections, this representation space will be enriched by a geometric structure, allowing formal definition of concepts such as clinical proximity, physiological trajectories, and computational phenotype discovery.

## 7. Geometry of the Representation Space

The previous sections introduced the concept of the representation space as the mathematical environment where patients come to exist computationally. However, a space composed only of elements is not sufficient to sustain clinical inferences.

A representation only acquires operational meaning when it becomes possible to answer fundamental questions. How similar are two patients? Which individuals belong to the same phenotype? How can clinical progression be measured? How can therapeutic trajectories be compared?

These questions belong to the domain of geometry. In this theory, geometry is not considered a fixed characteristic of the representation, but an additional structure capable of organizing relations of proximity, continuity, and evolution among patients.

In other words, the representation describes what each patient is; geometry describes how different patients relate to one another.

### 7.1 The Problem of Clinical Distance

Consider two patients B₁ and B₂. After the representation process, one obtains

$$R(B₁) = x₁ ∈ X and R(B₂) = x₂ ∈ X.$$

An immediate question arises: what is the distance between x₁ and x₂? In most learning algorithms, this distance is simply assumed to be Euclidean,

$$d_E(x₁, x₂) = √(Σᵢ (xᵢ − yᵢ)²).$$

However, this hypothesis is rarely valid for biological systems. Distinct domains carry distinct meanings. Comparing age, respiratory flow, medical imaging, and clinical text using the same metric hardly preserves physiology. Therefore, clinical distance cannot be reduced to a distance between numbers. It must reflect physiological relations.

### 7.2 Geometry as an Independent Structure

Instead of assuming a specific geometry, a geometric operator is defined,

$$G : X → M,$$

where M represents the set of all admissible geometric structures. Note that G does not represent a metric. It represents a function capable of assigning a geometry to the representation space. This distinction is important. The theory remains independent of the mathematics chosen to implement the geometry.

### 7.3 Local Metrics

Once a geometry is defined, each point of the space acquires a notion of proximity. Formally, a distance function is defined,

$$d : X × X → ℝ.$$

This function must satisfy, at minimum, the classical properties of a metric. For any x, y, z ∈ X, the following must hold: non-negativity, d(x,y) ≥ 0; identity, d(x,y) = 0 ⟺ x = y; symmetry, d(x,y) = d(y,x); and the triangle inequality, d(x,z) ≤ d(x,y) + d(y,z). These properties guarantee mathematical consistency, regardless of the implementation adopted.

### 7.4 State-Dependent Geometry

However, a central hypothesis of this theory is that geometry may depend on the physiological state itself. In other words, the way two patients are compared may vary according to their clinical condition. Formally,

$$d_R : X × X → ℝ,$$

where d_R depends on the representation. Written explicitly,

$$d_R(x₁, x₂) = d(x₁, x₂ | R).$$

This formulation allows the representation itself to determine which physiological aspects are most relevant during comparison.

### 7.5 Clinical Similarity

In addition to distance, many applications depend on measures of similarity. A function is defined,

$$s : X × X → [0, 1],$$

such that s(x,y) = 1 indicates maximum similarity, while s(x,y) = 0 indicates absence of similarity. Note that similarity need not be derived directly from distance. It may incorporate physiological, causal, or probabilistic knowledge.

### 7.6 Neighborhood

Once geometry is defined, it becomes possible to introduce the concept of neighborhood. For a patient x ∈ X, its neighborhood of radius ε is defined as

$$N_ε(x) = {y ∈ X : d(x, y) ≤ ε}.$$

Note that this definition is independent of the algorithm subsequently used. It belongs exclusively to the representation space.

### 7.7 Continuity of Trajectories

Geometry also allows defining clinical continuity. Consider a trajectory Γ : T → X. It is desirable that small physiological evolutions produce small displacements in the space. Formally, as Δt → 0, it must follow that

$$d(Γ(t), Γ(t + Δt)) → 0.$$

This property guarantees the longitudinal stability of the representation.

### 7.8 Clinical Regions

An immediate consequence of geometry is the possibility of defining clinical regions. Let Ω ⊆ X. This region may represent, for example, high-risk patients, good therapeutic responders, elevated hypoxic burden, or rapid progression. Note that these regions exist independently of any clustering algorithm. They are geometric properties of the space.

### 7.9 Phenotypes as Geometric Regions

Traditionally, computational phenotypes are treated as outputs of algorithms. In this theory, a different interpretation is proposed. A phenotype is defined as a region

$$F ⊆ X.$$

More specifically, a set of patients sharing physiological proximity according to the defined geometry. Formally,

$$F = {x ∈ X : Φ(x) = 1},$$

where Φ represents a membership operator. Note that K-Means, Gaussian Mixture, HDBSCAN, DBSCAN, or any other algorithm do not define the phenotype. They merely estimate F. This distinction completely shifts the role of the algorithms.

### 7.10 The Representation Space as a Clinical Manifold

Under this perspective, the representation space ceases to be a collection of vectors. It comes to represent a continuous structure where patients occupy positions determined by their physiological characteristics. Geometry becomes responsible for organizing this space, allowing the definition of concepts such as clinical proximity, neighborhood, physiological regions, trajectories, and phenotypic stability. Note that none of these properties depends on the choice of a specific algorithm. All belong to the representation space itself.

### 7.11 Independence from Implementation

An important characteristic of the meta-model is that it does not impose a single geometry. Any implementation capable of satisfying the contracts established in Section 5 remains compatible with the theory. Thus, an implementation may use Euclidean geometry, Mahalanobis geometry, information geometry, kernel-based metrics, hyperbolic spaces, differentiable manifolds, or any other mathematical structure. All represent implementation choices. The theory remains unchanged.

This separation preserves the independence between the conceptual definition of the representation and the mathematical tools used to explore it.

In the next section, this geometric structure will be used to introduce a new interpretation of computational phenotyping, in which phenotypes cease to be products of clustering algorithms and come to be understood as emergent properties of the representation space itself.

## 8. Phenotyping as an Emergent Property of the Representation Space

One of the most recurrent applications of Artificial Intelligence in healthcare consists of automatically identifying clinical phenotypes. In recent decades, clustering algorithms have been used to identify subpopulations of patients with similar physiological characteristics, allowing a better understanding of the heterogeneity of various diseases.

However, a common characteristic of most of this literature is observed: the algorithm occupies a central position in the definition of phenotypes. Expressions such as "phenotypes obtained by K-Means," "phenotypes discovered by HDBSCAN," or "subgroups identified by Gaussian Mixture Models" implicitly suggest that the phenotype is a direct consequence of the algorithm employed.

In this theory, a different interpretation is proposed. It is argued that phenotypes are not produced by algorithms. Phenotypes constitute structural properties of the representation space. Algorithms merely estimate these structures. This shift in perspective moves the scientific contribution from the learning stage to the representation stage.

### 8.1 The Problem of Algorithmic Dependence

Consider two distinct algorithms, A₁ and A₂, applied to exactly the same set of patients. Traditionally, one obtains two sets of clusters,

$$F₁ = A₁(X) and F₂ = A₂(X).$$

In conventional literature, these two solutions are usually interpreted as different phenotypes. However, this interpretation presents an important conceptual limitation. It attributes to the algorithm the responsibility for the existence of the phenotypes.

In this theory, the opposite hypothesis is adopted. Phenotypes exist in the physiological space independently of the algorithm used. Formally, a set of phenotypes is defined,

$$F = {F₁, F₂, …, Fₘ},$$

where Fᵢ ⊆ X. Algorithms then play only the role of estimators of these regions.

### 8.2 Formal Definition of Phenotype

Let X be the representation space. A phenotype is defined as a physiologically consistent region of this space. Formally,

$$F ⊆ X.$$

This definition is purposefully simple. It does not depend on any specific algorithm. It states only that there exists a subset of the space whose elements share similar physiological characteristics. Consequently, the scientific problem ceases to be "which algorithm produces the best phenotypes?" and becomes "does the constructed representation allow the phenotypes to naturally emerge?"

### 8.3 Estimation Operators

Algorithms are then defined as operators

$$A : X → F̂,$$

where F̂ represents a computational estimate of the phenotypes. Thus, for any algorithm, one obtains

$$F̂ = A(X).$$

Note that F̂ does not necessarily correspond to the true phenotype. It represents only an approximation produced by the algorithm. This distinction is fundamental. It brings computational phenotyping closer to other classical problems in science, in which observational models estimate underlying structures that exist independently of the estimation method.

### 8.4 Phenotypic Consistency

An important consequence of this definition is the concept of consistency. If the representation adequately preserves physiology, different algorithms should converge to similar structures. Formally, let A₁, A₂, …, A_k be distinct algorithms. It is desired that

$$A₁(X) ≈ A₂(X) ≈ ⋯ ≈ A_k(X).$$

Note that the goal ceases to be maximizing agreement between algorithms. The goal becomes building a representation sufficiently rich that different algorithms produce compatible results. In this context, disagreement between algorithms comes to be interpreted as evidence of limitations of the representation, not necessarily of the clustering method.

### 8.5 Phenotypic Stability

Another important property concerns stability. Consider two independent samples, X₁ and X₂, obtained from the same population. A consistent representation must produce similar phenotypes. Formally,

$$F(X₁) ≈ F(X₂).$$

This property makes it possible to quantitatively evaluate the robustness of the representation. Highly unstable phenotypes suggest that the physiological structure is not yet being adequately preserved.

### 8.6 Clinical Interpretability

The theory also establishes that phenotypes must possess clinical meaning. An interpretation operator is defined,

$$I : F → Σ,$$

where Σ represents the set of possible clinical descriptions. Thus, each phenotype Fᵢ must possess a physiological interpretation, for example, respiratory predominance, hypoxic predominance, cardiovascular involvement, elevated sleep fragmentation, or high metabolic burden. Note that interpretation occurs after the regions are discovered, not during their construction.

### 8.7 Longitudinal Phenotypes

The main advantage of the proposed representation appears when temporality is incorporated. Consider a trajectory Γ : T → X. Each point of the trajectory belongs to some phenotype,

$$Γ(t) ∈ Fᵢ.$$

Over time, the patient may migrate between different regions of the space. Formally, a sequence is defined,

$$F_t1 → F_t2 → ⋯ → F_tn.$$

This sequence characterizes the individual's phenotypic trajectory. Note that, in this formulation, phenotypes cease to be fixed states. They become dynamic states of physiological evolution.

### 8.8 Phenotypic Transition Matrix

The dynamics between phenotypes can be described by a transition matrix. Let Fᵢ and Fⱼ. Define

$$p_ij = P(Fⱼ | Fᵢ),$$

representing the probability that a patient initially belonging to phenotype Fᵢ evolves to Fⱼ. One thus obtains a matrix

$$P = [p_ij],$$

which describes the entire population dynamics of the phenotypes. This structure allows investigation of disease progression, therapeutic effects, and the natural history of patients.

### 8.9 Phenotyping as a Space Operator

A direct consequence of the theory is that phenotyping ceases to be considered a clustering problem. It becomes defined as an operation performed on the representation space. Formally,

$$Φ : X → F,$$

where Φ represents a phenotype-identification operator. Algorithms become mere computational implementations of this operator. Thus, K-Means, Gaussian Mixture Models, HDBSCAN, DBSCAN, Spectral Clustering, or any other method cease to occupy a central position in the theory. They become merely different strategies for approximating Φ.

### 8.10 Experimental Validation

This shift in perspective has an important consequence for empirical validation. Traditionally, algorithms are compared with one another. In this theory, the representation becomes the primary object of validation. Consider two representations, R₁ and R₂. The comparison is then carried out according to criteria such as phenotypic stability, clinical interpretability, reproducibility, prognostic capacity, therapeutic response, and agreement between algorithms. Formally, it is desired that

$$Q(R₂) > Q(R₁),$$

where Q represents a representation-quality function. Note that the algorithm remains constant. The object of comparison becomes the representation. This methodological inversion constitutes one of the main contributions of the proposed theory.

By shifting attention from clustering to representation, phenotyping ceases to be understood as a machine-learning problem and comes to be understood as an emergent property of the mathematical organization of the physiological space.

In the following sections, this same structure will be used to show how phenotypic trajectories can be automatically integrated into large longitudinal cohorts, transforming healthcare data into permanent infrastructure for clinical and epidemiological research.

## 9. Automatic Construction of Longitudinal Cohorts

One of the main consequences of the computational representation proposed in this work is the transformation of the very nature of clinical cohorts.

In traditional epidemiology, a cohort constitutes a set of individuals selected according to specific inclusion criteria and followed over time to investigate clinical outcomes. Although this paradigm has produced much of modern epidemiological knowledge, its construction remains highly dependent on manual processes of data integration, cleaning, and organization.

Each new study normally requires the construction of a new database. Each new hypothesis requires a new definition of inclusion criteria. Each new analysis frequently implies rebuilding the entire computational pipeline. This approach makes longitudinal studies expensive, slow, and poorly reusable.

In this work a paradigm shift is proposed. It is argued that a cohort should not be understood as a database built to answer a specific question. It should be understood as a natural consequence of the computational representation of the patient. In other words, once a consistent representation is built, the cohort itself naturally emerges from the longitudinal evolution of the representations.

### 9.1 From Observation to Cohort

Consider a biological system B. Over time, new observations are made,

$$O(t₁), O(t₂), …, O(tₙ).$$

Each observation updates the semantic domains, which in turn produce a new computational representation. Formally,

$$R(B, tᵢ) = Rᵢ.$$

One thus obtains an ordered sequence

$$R₁, R₂, …, Rₙ.$$

This sequence naturally defines a clinical trajectory.

### 9.2 Longitudinal Trajectory

Formally, a patient's trajectory is defined as

$$Γ : T → X,$$

where T represents the temporal domain and X the representation space. At each instant,

$$Γ(t) = R(B, t).$$

Note that the fundamental object ceases to be an exam. It becomes the continuous evolution of the patient's representation. This distinction has profound methodological consequences. While an exam represents only an isolated observation, a trajectory represents the individual's complete computational history.

### 9.3 Continuous Update

A direct consequence of this definition is that new exams no longer generate new patients. They update patients that already exist. Formally, if R_t represents the current representation, the arrival of a new observation produces

$$R_{t+1} = U(R_t, O_{t+1}),$$

where U represents an update operator. Note that the patient remains the same. What evolves is their representation. This characteristic eliminates the need to completely rebuild the database whenever new clinical information is incorporated.

### 9.4 Cohorts as Sets of Trajectories

A direct consequence of the previous definition is that a cohort ceases to be a collection of individuals. It becomes a collection of trajectories. Formally,

$$C = {Γ₁, Γ₂, …, Γₙ}.$$

Each element of the cohort represents the complete temporal evolution of a patient. Thus, the temporal dimension becomes an integral part of the epidemiological structure, rather than an additional characteristic of the data.

### 9.5 Dynamic Phenotypes

Since each representation occupies a position in the space X, each instant of the trajectory also belongs to a phenotype. Formally,

$$Γ(t) ∈ Fᵢ.$$

Over the course of clinical evolution, the patient may migrate between different phenotypes. One thus obtains

$$F_t1 → F_t2 → ⋯ → F_tn.$$

This sequence characterizes the patient's phenotypic trajectory. Note that, in this formulation, phenotypes cease to represent static categories. They become transient states of physiological evolution.

### 9.6 Transition Matrices

The evolution between phenotypes can be described by transition probabilities. Define

$$p_ij = P(Fⱼ | Fᵢ),$$

where p_ij represents the probability that a patient initially belonging to phenotype Fᵢ evolves to Fⱼ. One thus obtains the matrix

$$P = [p_ij],$$

which describes the entire population dynamics of the disease. This structure allows the study of clinical progression, phenotypic stability, therapeutic response, and the natural history of the disease.

### 9.7 Therapeutic Response

Consider a therapeutic intervention τ. Its application modifies the patient's trajectory. Formally,

$$Γ_τ = τ(Γ).$$

Therapeutic response ceases to be analyzed only through differences between exams. It comes to be interpreted as a transformation of the trajectory within the representation space. This perspective allows treatments to be compared using the patient's entire physiological evolution, rather than isolated indicators alone.

### 9.8 Self-Updating Cohorts

One of the most important consequences of the theory is that cohorts become living structures. Whenever a new exam is incorporated, there is no need to rebuild the database. It suffices to update the corresponding trajectory. Formally, for a new observation O_new, one executes

$$Γ′ = U(Γ, O_new).$$

The cohort automatically becomes

$$C′ = (C − Γ) ∪ Γ′.$$

Note that no patient is recreated. Only their trajectory evolves. This mechanism allows the construction of continuously updated cohorts throughout clinical practice.

### 9.9 Permanent Epidemiological Infrastructure

In traditional epidemiology, a cohort is usually built to answer a specific hypothesis. In this theory, the cohort becomes permanent infrastructure. New questions no longer require new databases. It suffices to query the trajectories that already exist. Formally, any epidemiological study can be described by a function

$$Q : C → Y,$$

where Q represents an epidemiological query and Y the produced result. Note that the cohort remains the same. Only the questions change. This separation brings epidemiology closer to modern data science, in which the infrastructure remains stable while different analytical models are run over it.

### 9.10 The Case of Obstructive Sleep Apnea Syndrome

The initial implementation of this theory will be carried out using patients with Obstructive Sleep Apnea Syndrome undergoing therapy with a mandibular advancement device. Each polysomnography ceases to represent only a single exam. It comes to automatically update the patient's position within the representation space. Over the course of clinical follow-up, each individual builds their physiological trajectory, allowing observation of phenotypic migration, clinical stability, treatment response, progression patterns, disease recurrence, and prognostic factors.

More importantly, with each new patient incorporated, the epidemiological infrastructure itself grows automatically. There is no need to build a new cohort for each study. The cohort already exists. It simply keeps evolving.

### 9.11 Scientific Consequences

This change profoundly alters the relationship between clinical care and research. Data produced during care ceases to serve only as an assistance record. It comes to continuously feed a computational infrastructure capable of sustaining large-scale longitudinal studies.

In this scenario, cohort construction ceases to be a preparatory step of research. It becomes an emergent property of the computational representation of the patient. The consequence is an unprecedented convergence between clinical practice, computational epidemiology, and precision medicine.

Each new exam does not represent merely a new observation. It automatically expands the collective knowledge stored in the longitudinal cohort, allowing future hypotheses to be investigated without needing to reconstruct the data infrastructure.

This may be one of the main practical implications of the proposed theory: transforming routine care data into a permanent basis for clinical research, phenotype discovery, and the continuous development of computational models.

## 10. Generalization to Chronic Diseases

Up to this point, the development of the theory has been presented independently of any specific disease. Although the initial implementation was motivated by Obstructive Sleep Apnea Syndrome (OSAS), the central hypothesis of this work is that the proposed architecture describes a much more general property of computational medicine.

This hypothesis can be summarized as follows: the computational representation of the patient does not depend on the disease. It depends only on the existence of physiological domains capable of describing a biological system.

This statement completely shifts the focus of the theory. OSAS ceases to represent the object of the theory. It comes to represent only its first validation case. What remains constant is the meta-model. What changes are only the clinical domains used to build the representation.

### 10.1 Disease Independence

Consider a set of diseases

$$Θ = {θ₁, θ₂, …, θₘ}.$$

Each disease possesses a distinct pathophysiology. However, all can be described by a set of clinical domains. Formally, for each disease θᵢ ∈ Θ, define

$$D_θi = {D₁, D₂, …, D_k}.$$

Note that the domains may vary between diseases, but the structure used to organize them remains exactly the same. Consequently, the representation function continues to be

$$R : B → X.$$

The only difference lies in the composition of the domains.

### 10.2 Specializations of the Representation

The computational representation ceases to be unique. It comes to admit specializations. Formally, a family of representations is defined,

$${R_θ1, R_θ2, …, R_θm},$$

where R_θi represents the specific implementation for a disease. All share exactly the same meta-model, changing only their semantic domains. Thus, for OSAS, one obtains

$$R_OSAS = (D_Resp, D_Hyp, D_Sleep, D_Cardio, D_Symp, D_Treat).$$

While, for Diabetes Mellitus, one could define

$$R_DM = (D_Metab, D_Cardio, D_Renal, D_Retina, D_Neurop, D_Treat).$$

Note that the theory remains identical. Only the domains change.

### 10.3 The Meta-Model as Infrastructure

An important consequence of this formulation is that the meta-model comes to represent a common infrastructure for different medical specialties. Formally,

$$M = (B, O, D, R, X, G, Γ, F, C),$$

where M is completely independent of the disease studied. Each specialty implements only a particular instance of this meta-model. In computational terms, this means that classes such as Patient, Observation, Domain, Representation, RepresentationSpace, Phenotype, and Cohort do not need to be rewritten for each disease. Only new domains are added.

### 10.4 Examples of Specialization

In OSAS, the respiratory domain may contain

$$D_Resp = {AHI, RDI, Flow, Apneas, Hypopneas}.$$

In heart failure, the cardiovascular domain could be

$$D_Cardio = {Ejection Fraction, BP, Stroke Volume, NYHA Class}.$$

In COPD, the pulmonary domain can be defined as

$$D_Pulm = {FEV1, FVC, FEV1/FVC, Blood Gas Analysis}.$$

In oncology, a molecular domain could contain

$$D_Gen = {Gene Expression, Mutations, Biomarkers, Proteomics}.$$

Note that, in all cases, the structure of the representation remains exactly the same.

### 10.5 Computational Reuse

The computational consequence of this architecture is extremely important. Instead of developing a different library for each disease, a single infrastructure is defined. Formally, for any disease θ, one obtains

$$System = (M, R_θ),$$

where M remains constant, and R_θ corresponds only to the clinical specialization. This organization drastically reduces code duplication, increases component reuse, and facilitates multicenter validations.

### 10.6 Knowledge Transfer

Another important consequence is the possibility of knowledge transfer between diseases. Consider two diseases, θᵢ and θⱼ. If both share certain domains, these components can be reused. Formally, if

$$D_k ∈ D_θi and D_k ∈ D_θj,$$

then φ_k can be shared between both representations. This property favors the incremental development of the architecture. As new domains are implemented, they come to simultaneously benefit different specialties.

### 10.7 Discovery of New Domains

Another important aspect of the theory is that it admits continuous expansion. New pathophysiological knowledge may give rise to new domains. Formally, if D = {D₁, …, Dₙ}, the discovery of a new physiological mechanism produces

$$D′ = D ∪ {Dₙ₊₁}.$$

Consequently, the representation becomes

$$R′ = R ⊕ Rₙ₊₁.$$

Note that the architecture evolves naturally, without needing to reconstruct prior representations.

### 10.8 Representation-Based Medicine

This generalization produces an important conceptual shift. Traditionally, each disease tends to develop its own computational infrastructure. In this theory, the infrastructure is shared. Diseases become particular implementations of a common representation language. This perspective brings different medical fields closer together, allowing apparently distinct problems to be treated according to unified computational principles.

Medicine ceases to be organized around algorithms specific to each disease. It becomes organized around a general theory of computational representation.

### 10.9 OSAS as a Validation Case

It is important to emphasize that the choice of Obstructive Sleep Apnea Syndrome as the first object of study has a methodological nature. OSAS offers characteristics particularly suited for validating the theory. The disease presents multiple physiological domains, great clinical heterogeneity, an abundance of physiological signals, frequent longitudinal follow-up, and different therapeutic responses. These characteristics make it possible to empirically evaluate whether the proposed representation produces phenotypes that are more stable, more interpretable, and more prognostically relevant than traditional representations.

However, the success of the theory does not depend on OSAS. Its validity will depend on the ability to reproduce the same principles in other chronic diseases.

### 10.10 Scientific Consequences

The main consequence of this generalization is that the scientific object ceases to be a specific disease. The true object of the theory becomes the computational representation of biological systems. Sleep medicine represents only the first concrete implementation.

Just as libraries such as scikit-learn do not know specific problems but offer a general infrastructure for machine learning, the proposed architecture seeks to provide a general infrastructure for the computational representation of patients.

If this hypothesis is correct, new medical specialties could be incorporated not through the creation of new architectures, but through the definition of new semantic domains on the same meta-model.

This property brings computational medicine closer to a true science of representation, in which different diseases share a common formal language, allowing knowledge reuse, system interoperability, and the incremental development of physiologically grounded models.

## 11. Multimodal Integration

The previous sections established a theory for computational representation based on physiological domains. However, a fundamental characteristic of biological systems still needs to be considered: no single observation modality is sufficient to fully describe an organism.

Modern clinical practice simultaneously uses laboratory tests, continuous physiological signals, medical imaging, wearable devices, textual records, genetic data, and countless other sources of information. Each of these modalities captures only a partial perspective of the biological system.

Consequently, a general theory of the computational representation of the patient must be capable of integrating different modalities without losing conceptual coherence.

In this work, it is argued that this integration should not occur directly between the data, but between semantic representations built upon different physiological domains. This distinction constitutes one of the main differences between the proposed architecture and traditional data-fusion approaches.

### 11.1 The Problem of Heterogeneity

Consider a patient undergoing the following diagnostic procedures: polysomnography, electrocardiogram, laboratory tests, computed tomography, clinical questionnaires, and wearable-device monitoring. Each procedure produces information belonging to distinct mathematical spaces. Formally, modality Mᵢ generates observations belonging to

$$Oᵢ ∈ 𝒪ᵢ.$$

In general,

$$Oᵢ ≠ Oⱼ, for i ≠ j.$$

There is, therefore, no single observational space capable of simultaneously representing all these modalities. This heterogeneity constitutes one of the main challenges of contemporary computational medicine.

### 11.2 The Limitation of Direct Integration

Most current approaches attempt to solve this problem by concatenating attributes from different sources. Formally, one obtains

$$X = X_Lab ⊕ X_Image ⊕ X_Text ⊕ X_Sensor,$$

where ⊕ represents a simple concatenation of attributes. Although this strategy is simple, it presents important limitations. First, data from different modalities have completely different scales. Second, each modality has its own mathematical structure. Third, concatenation completely ignores the physiological semantics of the data. As a consequence, the resulting vector frequently represents a collection of attributes, not a biological organism.

### 11.3 Domain-Based Integration

In this theory, integration occurs at a higher level of abstraction. Each modality feeds one or more semantic domains. Formally, an association operator is defined,

$$Ω : O → D,$$

where Ω(Oᵢ) = Dⱼ. Note that different observations may simultaneously feed the same domain. For example, the cardiovascular domain may incorporate information from electrocardiogram, echocardiogram, blood pressure, heart rate, and heart-rate variability. Likewise, a single laboratory test may contribute to multiple physiological domains. Integration ceases to occur between variables. It comes to occur between clinical concepts.

### 11.4 Multimodal Representations

Once domains are defined, each produces its own representation. Formally,

$$φᵢ : Dᵢ → Xᵢ.$$

The global representation remains

$$R(B) = (φ₁(D₁), φ₂(D₂), …, φₙ(Dₙ)).$$

Note that the origin of the data becomes irrelevant. What matters is the representation produced by each domain. This property completely decouples the architecture from the observational modalities.

### 11.5 Integration of New Modalities

An important consequence of this architecture is that new modalities can be incorporated without modifying the overall structure of the system. Consider a new modality M_new. It produces observations O_new. These observations feed a set of domains {Dᵢ, Dⱼ, D_k}. The global representation becomes

$$R′ = R ⊕ φ(O_new).$$

Note that no change occurs in the other components of the architecture. Expansion becomes incremental.

### 11.6 Examples of Modalities

The architecture naturally admits different types of data. Continuous physiological signals, such as respiratory flow, can be represented as

$$s(t) ∈ S.$$

Medical images belong to

$$I ∈ 𝓘.$$

Laboratory tests can be written as

$$L ∈ ℝᵐ.$$

Clinical text belongs to the space

$$T ∈ 𝓛.$$

Genomic data can be represented by

$$G ∈ 𝒢.$$

Wearable devices produce time series

$$W(t) ∈ 𝒲.$$

Note that each modality preserves its own mathematical structure. The theory does not require its conversion into a single vector.

### 11.7 Shared Domains

Another important consequence is that different modalities can simultaneously contribute to the same domain. Formally, define

$$Dᵢ = f(O₁, O₂, …, O_k).$$

This means that a physiological domain represents a computational synthesis built from multiple observations. For example, the hypoxia domain can simultaneously integrate peripheral saturation, hypoxic burden, arterial blood gas analysis, heart rate, and desaturation index. None of these variables alone defines the domain. The domain emerges from the semantic integration among them.

### 11.8 Latent Domains

Multimodal integration makes it possible to infer physiological states that cannot be directly observed. Consider a latent domain D_L. There is no direct observation for D_L. However, it can be inferred from multiple modalities. Formally,

$$D_L = f(O_Lab, O_Image, O_Sensor, O_Text).$$

Examples include systemic inflammation, physiological reserve, frailty, autonomic burden, and metabolic stability. These states cease to depend on a single exam. They come to represent emergent properties of multimodal integration.

### 11.9 Continuous Evolution of the Representation

An important characteristic of the proposed architecture is that new modalities may emerge over time. Consider a sequence M₁, M₂, …, Mₙ. Each modality adds new observations but does not alter the fundamental structure of the meta-model. Formally, one obtains

$$Rₙ₊₁ = Rₙ ⊕ φ(M_new).$$

The representation evolves continuously, preserving compatibility with prior versions. This property makes the architecture naturally prepared for future diagnostic technologies.

### 11.10 An Architecture for Multimodal Medicine

The main consequence of the multimodal integration proposed in this work is that the patient ceases to be described by a set of variables from different exams. The patient comes to be described by an organized collection of physiological domains, each fed by multiple observational modalities.

This change has important implications. First, new technologies can be incorporated without rebuilding the architecture. Second, the representation remains semantically consistent regardless of the origin of the data. Third, algorithms cease to depend on the original observational structure, coming instead to operate on physiologically grounded representations.

Finally, multimodal integration ceases to be a feature-engineering problem. It becomes a natural consequence of the meta-model of computational patient representation itself.

This characteristic brings the proposed architecture closer to a general infrastructure for computational medicine, capable of continuously integrating new diagnostic modalities while preserving the mathematical and semantic coherence of the representation.

## 12. A General Theory of Computational Precision Medicine

The previous sections presented the mathematical foundations of a theory of the computational representation of the patient. First, a meta-model was defined capable of explicitly separating biological system, observations, semantic domains, computational representation, geometry, trajectories, and phenotypes. Next, it was shown how this structure allows the automatic construction of longitudinal cohorts and the integration of different data modalities while preserving their semantic coherence.

However, all these contributions converge toward an even broader consequence. The proposed theory does not represent merely a new way of organizing clinical data. It proposes a new way of understanding computational medicine itself.

Traditionally, artificial intelligence applied to healthcare has been organized around algorithms. Neural networks, decision trees, probabilistic methods, and deep-learning techniques occupy a central position in the scientific literature. Patient representation remains, for the most part, implicit, built during feature-engineering steps and rarely treated as an independent object of investigation.

In this work, it is proposed to completely invert this perspective. The central hypothesis is that the true object of computational medicine is not the algorithm. It is the mathematical representation of the biological system.

Formally, every computational application in healthcare can be described as a composition of functions

$$A ∘ R : B → Y,$$

where R : B → X represents the representation function, and A : X → Y represents any inference algorithm. Traditionally, scientific development has concentrated almost exclusively on improving A. However, it is argued that R constitutes an equally fundamental layer, capable of determining the limits of any subsequently applied algorithm.

This change profoundly alters the architecture of computational medicine.

### 12.1 The Centrality of Representation

In this theory, computational representation comes to occupy a central position. Every clinical application ceases to operate directly on observations. It comes to operate on a representation space. Formally, the computational flow becomes

$$B → O → D → R → X → A → Y.$$

Note that the algorithm ceases to receive clinical data. It receives only representations belonging to the space X. This separation produces an important consequence. Representation and inference become independent scientific problems.

### 12.2 Algorithms as Operators

A direct consequence of the theory is that algorithms cease to occupy a privileged position. Any algorithm comes to be interpreted merely as an operator defined over the representation space. Formally,

$$A : X → Y.$$

Nothing in the theory requires that A correspond to a specific method. It may represent classification, regression, phenotype discovery, causal inference, expert systems, therapeutic optimization, probabilistic models, or deep neural networks. All become particular implementations of the same abstract concept. This algorithmic independence constitutes one of the main objectives of the meta-model.

### 12.3 Medicine as Navigation in Representation Spaces

An even deeper consequence emerges when one considers the geometry of the computational space. If each patient occupies a position in X, then clinical practice can be reinterpreted as a navigation problem within this space. Diagnosis, prognosis, risk stratification, and therapeutic response cease to be independent problems. All come to represent operations performed on trajectories. Formally, a clinical trajectory is defined by

$$Γ : T → X.$$

The goal of medicine becomes understanding, predicting, and modifying this trajectory. Under this perspective, treatment can be interpreted as an operator

$$τ : X → X,$$

capable of altering the patient's evolution. One thus obtains

$$Γ′ = τ(Γ),$$

where Γ′ represents the trajectory after the therapeutic intervention. This formulation naturally brings together computational medicine, optimal control, dynamical systems, and causal inference.

### 12.4 Representation-Based Precision Medicine

Personalized medicine then assumes a completely different interpretation. Traditionally, precision medicine seeks to identify the best treatment for a given patient. In this theory, the problem becomes identifying which transformation τ produces the most favorable trajectory within the space X. Formally, one seeks

$$τ* = arg min_τ L(Γ_τ),$$

where L represents a clinical loss function, such as risk of death, hospitalization, disease progression, or therapeutic failure. Note that treatment ceases to be merely a clinical intervention. It comes to represent a geometric transformation of the physiological state.

### 12.5 Integration Between Research and Care

Another important consequence of the theory is the convergence between scientific research and clinical practice. Since each new observation automatically updates the representation, every clinical encounter produces computational knowledge. Formally, each new observation O_new generates

$$R_new = U(R, O_new),$$

where U represents the update operator. At the same time, the patient's trajectory also evolves,

$$Γ′ = U(Γ, O_new).$$

Consequently, the epidemiological infrastructure grows continuously, without the need for manual construction of new cohorts. Care and research cease to represent separate activities. Both come to share the same representation infrastructure.

### 12.6 A General Infrastructure for Computational Medicine

Under this perspective, the meta-model ceases to be merely an architecture for patient representation. It comes to define a general infrastructure for computational medicine. Formally, any clinical application can be described by a composition of operators over X. For example, diagnosis

$$D : X → Y_D,$$

prognosis

$$P : X → Y_P,$$

phenotyping

$$Φ : X → F,$$

therapeutic prediction

$$T : X → Y_T,$$

and causal inference

$$C : X → Y_C.$$

All these operators share exactly the same representation. This reuse reduces model duplication, facilitates comparisons, and significantly increases interoperability between different applications.

### 12.7 A New Perspective for Artificial Intelligence in Healthcare

Perhaps the main conceptual consequence of this theory is the redefinition of the role of Artificial Intelligence. Instead of understanding AI as a set of algorithms capable of learning directly from clinical data, it is proposed to understand it as a set of operators capable of exploring a physiologically grounded representation space.

In this scenario, the quality of the inferences ceases to depend exclusively on the sophistication of the algorithms. It comes to depend also on the quality of the constructed representation. This shift brings computational medicine closer to other areas of computer science, in which representation and inference are treated as independent problems.

### 12.8 Concluding Remarks

The objective of this work was not to propose a new machine-learning algorithm, a new clustering technique, or a new clinical-prediction method. Its main contribution consists of formalizing a theory of the computational representation of the patient, establishing a meta-model capable of separating observation, representation, geometry, and inference.

This separation allows biological systems to be understood as mathematical objects organized within representation spaces, on which different computational operators can act in a consistent, reproducible, and disease-independent manner.

If this hypothesis is correct, the impact of the theory extends beyond sleep medicine, which represents only its first validation case. It offers a formal language for computationally organizing any biological system, establishing the foundations for a computational medicine truly oriented toward representations.

In this new paradigm, algorithms cease to occupy the center of the architecture. The fundamental element becomes the mathematical representation of the patient. Computational medicine ceases to be defined by the models it uses to infer knowledge and comes to be defined by the way it represents the object of study itself.

This may be the main change proposed by this work: before asking which algorithm learns best, it is necessary to answer how a biological system should be mathematically represented. It is this representation, not the algorithm, that constitutes the true foundation of computational precision medicine.

## 13. Discussion

The theory presented in this work proposes a shift in perspective on how computational systems in healthcare are conceived. Rather than focusing attention on developing new machine-learning algorithms, it is proposed that the main scientific object of computational medicine be the mathematical representation of the patient itself. This shift may seem subtle, but it carries profound conceptual implications.

Much of the contemporary literature on Artificial Intelligence applied to healthcare is devoted to developing increasingly sophisticated inference methods. Deep neural networks, probabilistic models, Bayesian methods, foundation models, and multimodal architectures represent important advances for the analysis of clinical data. However, virtually all share a common characteristic: they implicitly assume that the patient has already been adequately represented.

This hypothesis is rarely questioned. In practice, the representation tends to arise during preprocessing, normalization, and feature-engineering steps, and is frequently determined by the limitations of the algorithms used or by data availability. Consequently, two studies may use exactly the same algorithm and obtain completely different results simply because they started from different representations of the patient.

This observation shifts the scientific discussion to a more fundamental level. Before asking which algorithm performs best, one must ask whether the algorithms are learning about the same mathematical object.

The proposed theory seeks to answer precisely this question. By explicitly separating biological system, observations, semantic domains, representation, representation space, geometry, and inference, a layered architecture is established in which each component has clearly defined responsibilities. This separation reduces coupling between data acquisition, physiological modeling, and computational learning, allowing each of these layers to evolve independently.

An important consequence of this organization is algorithmic independence. In the traditional paradigm, different algorithms often require different data representations. Linear algorithms tend to favor normalized vector representations; time-series models use sequences; deep neural networks frequently operate on tensors or embeddings. As a result, representation becomes subordinate to the algorithm.

In the proposed architecture, exactly the opposite occurs. First, a physiologically consistent representation of the patient is built. Only afterward is the algorithm that best exploits this representation chosen. This inversion profoundly changes the role of machine learning. Algorithms cease to define the patient's computational structure. They come to merely explore a previously constructed space.

This distinction also changes the interpretation of computational phenotyping. In the current literature, it is common to find expressions such as "phenotypes obtained by K-Means," "phenotypes discovered by HDBSCAN," or "subgroups found by Gaussian Mixture Models." These expressions implicitly attribute the existence of phenotypes to the algorithm.

In this work, a different hypothesis is adopted. Phenotypes are understood as structural properties of the representation space. Algorithms merely estimate these structures. This shift brings computational phenotyping closer to classical problems in physics and statistics, in which observational models seek to reconstruct structures that exist independently of the estimation method used.

Another important consequence concerns longitudinality. In most current systems, successive exams are treated as independent records. This hinders the construction of longitudinal studies and requires repeated database integrations for each new research project.

In the proposed theory, each new observation only updates the patient's position within the representation space. The patient ceases to be a collection of exams. They come to be described by a trajectory. Consequently, longitudinal cohorts become a natural property of the architecture. This characteristic brings clinical care and epidemiological research closer together. Each new exam performed during clinical care automatically comes to expand the infrastructure available for future scientific investigations.

Another important contribution is the separation between observational modalities and physiological domains. Most multimodal architectures directly integrate different types of data, concatenating vectors from images, physiological signals, laboratory tests, and clinical text. Although this approach is operationally simple, it ignores the physiological semantics of the observations.

In this work, a different strategy is proposed. Modalities feed clinical domains. Domains produce representations. Only then does integration occur. This organization preserves the physiological coherence of the representation and allows new modalities to be incorporated without rebuilding the entire computational infrastructure.

From a Software Engineering standpoint, the proposed architecture also presents important advantages. The meta-model defines formal contracts independent of implementations. Just as the relational model made it possible to separate databases from specific applications, and UML separated modeling from implementation, the proposed meta-model separates computational representation from inference algorithms.

This separation favors code reuse, interoperability between applications, incremental evolution of the architecture, and independent validation of each component.

However, some limitations must be acknowledged. The first concerns the definition of semantic domains. Although the meta-model establishes how these domains relate to one another, identifying the most appropriate physiological domains for each disease remains an open scientific problem. Different medical specialties may propose distinct decompositions of the same biological system, requiring empirical studies to evaluate their representational capacity.

Another limitation concerns the geometry of the representation space. Throughout this work, it was deliberately decided not to fix a specific geometric structure. This decision preserves the generality of the theory but transfers to future implementations the responsibility of investigating which geometries produce greater phenotypic stability, better clinical interpretability, and greater prognostic capacity.

Similarly, the theory does not impose specific mathematical structures for representing each domain. Vector spaces, time series, graphs, tensors, differentiable manifolds, or neural embeddings remain equally compatible with the meta-model, provided they satisfy the contracts defined earlier. This flexibility significantly broadens the scope of the theory but requires future investigation into the limits and advantages of each approach.

Another aspect that deserves future investigation is the modeling of latent domains. Many relevant physiological processes, such as systemic inflammation, functional reserve, or autonomic stability, cannot be directly observed. Their representation will depend on inverse problems, probabilistic models, or causal-inference methods, constituting an important extension of this theory.

Experimental validation also represents an essential step. In this work, Obstructive Sleep Apnea Syndrome was chosen as the first application case due to its high physiological heterogeneity, availability of multimodal signals, and possibility of longitudinal follow-up during treatment with mandibular advancement devices.

However, the theory can only be considered truly general if it produces consistent results in other chronic diseases, such as diabetes mellitus, heart failure, COPD, neurodegenerative diseases, oncology, and psychiatry.

Finally, the main contribution of this work is perhaps not a new computational architecture, a software library, or a learning algorithm. Its main contribution consists of proposing that the computational representation of the patient be recognized as a scientific object in its own right.

This shift in perspective moves computational medicine from a science centered on algorithms to a science centered on representations. In this new paradigm, algorithms become interchangeable operators. Data become observations. Patients become biological systems. And mathematical representation comes to constitute the true central element upon which all computational intelligence in healthcare is built.

This conceptual inversion may represent the main implication of this theory. If correct, the future of computational medicine will depend less on discovering increasingly sophisticated algorithms and more on building computational representations that are increasingly faithful to the complexity of the biological systems they seek to describe.

## 14. Conclusion

Contemporary medicine faces an unprecedented transformation. The exponential growth in the availability of clinical data, physiological signals, medical imaging, wearable devices, electronic health records, and omics information has significantly expanded the capacity to observe biological systems. In parallel, advances in machine learning have produced increasingly sophisticated algorithms for exploiting this data.

However, throughout this work it has been argued that there is a conceptual gap that precedes machine learning itself. Although algorithms are continuously refined, little attention has been devoted to the mathematical object on which these algorithms actually operate.

Computers do not observe patients. They observe representations.

This observation motivated the central hypothesis of this thesis: the computational representation of the patient constitutes a scientific problem independent of inference algorithms and must be formalized as its own layer of computational medicine.

To investigate this hypothesis, a meta-model was developed, organized across different levels of abstraction, explicitly separating biological system, observations, semantic domains, computational representation, representation space, geometry, trajectories, phenotypes, and cohorts.

This organization made it possible, for the first time in an explicit manner, to distinguish three problems frequently treated as a single process: observing a biological system; representing it computationally; and inferring knowledge from that representation.

By separating these layers, it becomes possible to study properties of the representation independently of the algorithms subsequently used. This may be the main conceptual contribution of this work.

Throughout the text it was also argued that patients should not simply be described as attribute vectors. Vectors represent only one possible implementation. The proposed theory allows different physiological domains to be represented by distinct mathematical structures — vector spaces, time series, graphs, tensors, embeddings, or functional spaces — preserving the specific nature of each clinical modality. This characteristic makes the representation significantly more flexible and naturally extensible.

Another important contribution consists of the redefinition of the concept of computational phenotype. Traditionally, phenotypes are often treated as results produced by clustering algorithms. In this work, a different interpretation was proposed. Phenotypes come to be understood as emergent properties of the representation space. Algorithms cease to "create" phenotypes. They come merely to estimate regions that already exist structurally within that space. This change shifts the scientific focus from the choice of algorithm to the quality of the constructed representation.

Likewise, it was shown that clinical trajectories and longitudinal cohorts can naturally emerge from the proposed architecture. Each new exam ceases to represent merely a new care record. It comes to automatically update the patient's representation, changing their position in the representation space and their physiological trajectory. As a consequence, cohorts cease to be databases built specifically for a research project. They come to constitute permanent representation infrastructures capable of continuously growing during clinical practice.

The theory also demonstrated that multimodal integration can be reinterpreted as a semantic-representation problem. Rather than concatenating attributes from different diagnostic modalities, it is proposed that physiological signals, laboratory tests, medical imaging, wearable devices, and clinical text feed common physiological domains, preserving the biological coherence of the representation. This approach reduces coupling between observational modalities and facilitates the incorporation of new diagnostic technologies without requiring reconstruction of the architecture.

Although the initial motivation for this research arose from Obstructive Sleep Apnea Syndrome, the theory was deliberately constructed to not depend on any specific disease. OSAS represents only the first validation scenario. The true scientific object of this proposal is the computational representation of biological systems.

Under this perspective, distinct chronic diseases come to share the same conceptual infrastructure, differing only in the physiological domains used to build their representations. This characteristic opens the way for a unified infrastructure for computational medicine, reusable across different medical specialties.

Naturally, several questions remain open. The optimal definition of physiological domains, the choice of the most appropriate geometries for the representation space, the modeling of latent states, the incorporation of causal mechanisms, and the multicenter validation of the theory constitute important challenges for future research.

Likewise, the computational implementation of this meta-model as an independent software library represents a natural next step toward turning the concepts presented here into a reusable infrastructure for research and clinical practice.

These limitations, however, do not diminish the main contribution of this work. On the contrary. They indicate that the proposed theory establishes a new research agenda. Instead of concentrating efforts exclusively on building more sophisticated algorithms, it becomes possible to systematically investigate how biological systems should be computationally represented before any algorithm is applied.

This shift brings computational medicine closer to other established areas of Computer Science. Just as the relational model formalized the organization of data before the construction of modern databases, and UML established a common language for system modeling before implementation, the proposed meta-model seeks to provide a formal language for representing patients and other biological systems in a manner that is consistent, extensible, and independent of the inference techniques subsequently employed.

Ultimately, this work proposes a paradigm shift. Artificial Intelligence in healthcare ceases to be understood as a discipline centered on algorithms. It comes to be understood as a discipline centered on representations. Algorithms continue to play an essential role. However, they cease to occupy the core of the architecture. The central element becomes the mathematical representation of the biological system.

If this hypothesis is correct, the advancement of computational medicine over the coming decades will depend not only on increasingly powerful algorithms, but principally on the construction of computational representations capable of preserving, with growing fidelity, the physiological complexity of the organisms they seek to describe.

More than proposing a new machine-learning method, this thesis proposes a new scientific object: a formal theory of the computational representation of the patient.

And perhaps it is precisely in this shift in perspective — from inference to representation — that the next great step of computational precision medicine lies.

