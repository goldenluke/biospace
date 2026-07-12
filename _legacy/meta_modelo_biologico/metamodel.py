"""
metamodel.py
============

Implementação computacional do meta-modelo descrito em:

    "Fundamentos Matemáticos da Representação Computacional de Sistemas
    Biológicos - Um Meta-Modelo para Medicina Computacional de Precisão"

O artigo define a tupla

    M = (B, O, D, R, X, G, Γ, F, C)

com o fluxo conceitual

    Sistema Biológico → Observações → Domínios → Representação
        → Espaço de Representação → Geometria → Trajetória → Fenótipo → Coorte

Este módulo implementa cada uma dessas entidades como uma classe Python,
mais um conjunto de "contratos" (Seção 5 do artigo) que podem ser
verificados automaticamente sobre qualquer instância concreta do meta-modelo.

Nenhuma estrutura matemática é imposta a priori: cada domínio escolhe seu
próprio espaço (vetor, série temporal, tensor, embedding etc.), e a
composição ocorre apenas no nível do produto cartesiano dos espaços
individuais (Seção 6).
"""

from __future__ import annotations

import uuid
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# 4.1  Sistema Biológico  ->  B = (E, T)
# =============================================================================
class BiologicalSystem:
    """
    Representa o sistema biológico real (o paciente), independentemente de
    qualquer observação ou representação computacional.

    Formalmente B ∉ X: o sistema biológico não pertence ao espaço
    computacional, apenas suas *representações* pertencem.

    Esta classe funciona como um "handle" estável: o paciente nunca é
    recriado, apenas atualizado (Seção 9.3 — Atualização Contínua).
    """

    def __init__(self, patient_id: Optional[str] = None, metadata: Optional[dict] = None):
        self.id: str = patient_id or str(uuid.uuid4())
        self.metadata: dict = metadata or {}
        # Estado bruto acumulado ao longo do tempo (fonte de todas as observações)
        self._raw_states: List[Tuple[datetime, dict]] = []

    def ingest(self, timestamp: datetime, raw_data: dict) -> None:
        """Registra uma nova leitura bruta (ex.: resultado de um exame)."""
        self._raw_states.append((timestamp, dict(raw_data)))

    def latest_raw_state(self) -> dict:
        """Retorna o estado bruto mais recente, agregando todas as leituras."""
        merged: dict = {}
        for _, data in self._raw_states:
            merged.update(data)
        return merged

    def history(self) -> List[Tuple[datetime, dict]]:
        return list(self._raw_states)

    def __repr__(self) -> str:
        return f"BiologicalSystem(id={self.id!r}, n_readings={len(self._raw_states)})"


# =============================================================================
# 4.2  Observações  ->  O_i : B -> M_i
# =============================================================================
@dataclass
class ObservationResult:
    """Resultado de uma medição, com proveniência explícita (Contrato 5.1)."""
    name: str
    value: Any
    source: str
    timestamp: datetime


class Observation:
    """
    Um operador de medida. Observações não são dados — são *funções* que
    extraem informação parcial de um BiologicalSystem, sempre rastreáveis
    até uma fonte concreta (instrumento, questionário, sensor etc.).
    """

    def __init__(
        self,
        name: str,
        extractor: Callable[[BiologicalSystem], Any],
        source: str,
        unit: Optional[str] = None,
    ):
        self.name = name
        self.extractor = extractor
        self.source = source
        self.unit = unit

    def measure(self, B: BiologicalSystem, timestamp: Optional[datetime] = None) -> ObservationResult:
        value = self.extractor(B)
        return ObservationResult(self.name, value, self.source, timestamp or datetime.now())

    def __repr__(self) -> str:
        return f"Observation({self.name!r}, source={self.source!r})"


# =============================================================================
# 4.3  Domínio Semântico  ->  D_i = (Σ_i, O_i)
# =============================================================================
class SemanticDomain:
    """
    Agrupa observações segundo seu significado clínico (não sua estrutura
    matemática). Ex.: domínio respiratório, domínio de hipóxia etc.
    """

    def __init__(self, name: str, semantics: str, observations: Sequence[Observation]):
        self.name = name
        self.semantics = semantics  # Σ_i: descrição textual da semântica clínica
        self.observations: List[Observation] = list(observations)

    def collect(self, B: BiologicalSystem, timestamp: Optional[datetime] = None) -> Dict[str, ObservationResult]:
        """Executa todas as observações do domínio sobre um sistema biológico."""
        return {obs.name: obs.measure(B, timestamp) for obs in self.observations}

    def __repr__(self) -> str:
        names = [o.name for o in self.observations]
        return f"SemanticDomain({self.name!r}, observations={names})"


# =============================================================================
# 4.4 / 4.5  Operador de Representação  ->  φ_i : D_i -> X_i   e   R(B) = (φ_1(D_1), ..., φ_n(D_n))
# =============================================================================
class RepresentationOperator:
    """
    φ_i : D_i -> X_i

    Transforma as observações coletadas de um domínio em um elemento de um
    espaço matemático X_i (vetor, tensor, embedding, série temporal, ...).
    A teoria não impõe qual estrutura X_i deve ter — apenas que φ_i seja
    determinística (Contrato 5.8 — Reprodutibilidade).
    """

    def __init__(self, domain: SemanticDomain, mapping: Callable[[Dict[str, ObservationResult]], np.ndarray]):
        self.domain = domain
        self.mapping = mapping

    def apply(self, B: BiologicalSystem, timestamp: Optional[datetime] = None) -> np.ndarray:
        obs_results = self.domain.collect(B, timestamp)
        return np.asarray(self.mapping(obs_results), dtype=float)

    def __repr__(self) -> str:
        return f"RepresentationOperator(domain={self.domain.name!r})"


class Representation:
    """
    R : B -> X

    Representação global de um sistema biológico como a composição das
    representações de cada domínio (Princípio da Composicionalidade, 3.2):

        R(B) = (φ_1(D_1), φ_2(D_2), ..., φ_n(D_n))

    Internamente guarda um dicionário {nome_do_domínio: vetor}, o que
    permite extensibilidade (novos domínios podem ser adicionados sem
    invalidar os anteriores — Contrato 5.5) e rastreabilidade por domínio.
    """

    def __init__(self, operators: Sequence[RepresentationOperator]):
        self.operators: Dict[str, RepresentationOperator] = {op.domain.name: op for op in operators}

    def compute(self, B: BiologicalSystem, timestamp: Optional[datetime] = None) -> "PatientRepresentation":
        components = {name: op.apply(B, timestamp) for name, op in self.operators.items()}
        return PatientRepresentation(patient_id=B.id, timestamp=timestamp or datetime.now(), components=components)

    def extend(self, new_operator: RepresentationOperator) -> "Representation":
        """
        Contrato da Extensibilidade (5.5): R' = R ⊕ R_{n+1}, preservando
        todos os componentes anteriores.
        """
        new_ops = list(self.operators.values()) + [new_operator]
        return Representation(new_ops)

    def domain_names(self) -> List[str]:
        return list(self.operators.keys())


@dataclass
class PatientRepresentation:
    """
    Um ponto x = R(B) no espaço de representação X, em um instante t.
    Mantém os componentes por domínio separados (para rastreabilidade
    e para permitir geometrias específicas por domínio).
    """
    patient_id: str
    timestamp: datetime
    components: Dict[str, np.ndarray]

    def as_vector(self, domain_order: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatena os componentes em um único vetor (projeção conveniente para X = R^n)."""
        names = domain_order or sorted(self.components.keys())
        return np.concatenate([np.ravel(self.components[n]) for n in names])

    def __repr__(self) -> str:
        dims = {k: np.ravel(v).shape[0] for k, v in self.components.items()}
        return f"PatientRepresentation(patient={self.patient_id}, t={self.timestamp}, dims={dims})"


# =============================================================================
# 4.6  Espaço de Representação  ->  X = X_1 × X_2 × ... × X_n
# =============================================================================
class RepresentationSpace:
    """
    X = {R(B_1), R(B_2), ..., R(B_n)}

    O objeto computacional deixa de ser uma matriz de atributos e passa a
    ser o próprio espaço: uma coleção de PatientRepresentation, sobre a
    qual algoritmos e geometrias atuam.
    """

    def __init__(self, domain_order: Optional[Sequence[str]] = None):
        self._points: Dict[str, PatientRepresentation] = {}
        self.domain_order = domain_order

    def add(self, rep: PatientRepresentation) -> None:
        self._points[rep.patient_id] = rep

    def get(self, patient_id: str) -> PatientRepresentation:
        return self._points[patient_id]

    def patient_ids(self) -> List[str]:
        return list(self._points.keys())

    def matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Projeta X em uma matriz numérica (n_pacientes x dim), útil como
        *implementação* concreta para alimentar algoritmos clássicos.
        A ordem dos domínios é fixada para garantir consistência entre
        pacientes.
        """
        ids = self.patient_ids()
        order = self.domain_order or sorted(next(iter(self._points.values())).components.keys())
        M = np.stack([self._points[i].as_vector(order) for i in ids])
        return M, ids

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"RepresentationSpace(n_patients={len(self)})"


# =============================================================================
# 4.7  Geometria  ->  G : X -> M
# =============================================================================
class Geometry:
    """
    Atribui uma noção de distância/similaridade ao espaço de representação.
    A teoria não fixa uma geometria — Euclidiana, Mahalanobis etc. são todas
    implementações válidas, desde que satisfaçam as propriedades de métrica
    (Seção 7.3).
    """

    def __init__(self, distance_fn: Callable[[np.ndarray, np.ndarray], float], name: str = "custom"):
        self.distance_fn = distance_fn
        self.name = name

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.distance_fn(x, y))

    def similarity(self, x: np.ndarray, y: np.ndarray, scale: float = 1.0) -> float:
        """s(x,y) ∈ [0,1], derivada da distância via kernel gaussiano (uma escolha entre várias)."""
        d = self.distance(x, y)
        return float(np.exp(-(d ** 2) / (2 * scale ** 2)))

    def neighborhood(
        self, space: RepresentationSpace, patient_id: str, radius: float, order: Optional[Sequence[str]] = None
    ) -> List[str]:
        """B_ε(x) = {y ∈ X : d(x, y) ≤ ε}  (Seção 7.6)."""
        ids = space.patient_ids()
        ref = space.get(patient_id).as_vector(order)
        result = []
        for pid in ids:
            if pid == patient_id:
                continue
            other = space.get(pid).as_vector(order)
            if self.distance(ref, other) <= radius:
                result.append(pid)
        return result

    @staticmethod
    def euclidean() -> "Geometry":
        return Geometry(lambda x, y: np.linalg.norm(x - y), name="euclidean")

    @staticmethod
    def mahalanobis(cov: np.ndarray) -> "Geometry":
        inv_cov = np.linalg.pinv(cov)

        def d(x, y):
            diff = x - y
            return float(np.sqrt(diff @ inv_cov @ diff.T))

        return Geometry(d, name="mahalanobis")

    def __repr__(self) -> str:
        return f"Geometry({self.name})"


# =============================================================================
# 4.8  Trajetória  ->  Γ : T -> X
# =============================================================================
class Trajectory:
    """
    A evolução longitudinal da representação de um único sistema biológico.
    Cada novo exame *atualiza* a trajetória — não cria um novo paciente
    (Seção 9.2 / 9.3).
    """

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self._points: List[PatientRepresentation] = []

    def update(self, rep: PatientRepresentation) -> None:
        assert rep.patient_id == self.patient_id, "Representação pertence a outro paciente."
        self._points.append(rep)
        self._points.sort(key=lambda r: r.timestamp)

    def at(self, index: int) -> PatientRepresentation:
        return self._points[index]

    def timestamps(self) -> List[datetime]:
        return [p.timestamp for p in self._points]

    def as_matrix(self, order: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.stack([p.as_vector(order) for p in self._points])

    def phenotype_sequence(self, phenotypes: "List[Phenotype]", order: Optional[Sequence[str]] = None) -> List[Optional[str]]:
        """
        F_{t1} -> F_{t2} -> ... -> F_{tn}  (Seção 8.7 — Fenótipos Longitudinais)
        """
        seq = []
        for p in self._points:
            vec = p.as_vector(order)
            match = next((ph.name for ph in phenotypes if ph.contains(vec)), None)
            seq.append(match)
        return seq

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"Trajectory(patient={self.patient_id}, n_points={len(self)})"


# =============================================================================
# 4.9  Fenótipo  ->  F ⊆ X
# =============================================================================
class Phenotype:
    """
    Um fenótipo é uma região do espaço de representação, não o resultado de
    um algoritmo (Seção 7.9 / 8.2). O algoritmo apenas *estima* Φ; esta
    classe representa a região já estimada, junto de sua interpretação
    clínica opcional (Seção 8.6).
    """

    def __init__(self, name: str, membership_fn: Callable[[np.ndarray], bool], interpretation: str = ""):
        self.name = name
        self.membership_fn = membership_fn
        self.interpretation = interpretation

    def contains(self, x: np.ndarray) -> bool:
        return bool(self.membership_fn(x))

    def __repr__(self) -> str:
        return f"Phenotype({self.name!r})"


class PhenotypeEstimator:
    """
    Φ_hat : X -> F_hat

    Um algoritmo (KMeans, GMM, HDBSCAN, regra clínica, ...) é apenas uma
    *implementação* deste operador de estimação — nunca o definidor do
    fenótipo em si (Seção 8.3 / 8.9).
    """

    def __init__(self, name: str):
        self.name = name

    def fit_predict(self, space: RepresentationSpace, order: Optional[Sequence[str]] = None) -> Dict[str, int]:
        raise NotImplementedError

    def to_phenotypes(
        self, space: RepresentationSpace, labels: Dict[str, int], order: Optional[Sequence[str]] = None
    ) -> List[Phenotype]:
        """Converte rótulos de cluster em objetos Phenotype (regiões estimadas)."""
        phenotypes = []
        unique_labels = sorted(set(labels.values()))
        centroids = {}
        for lbl in unique_labels:
            members = [pid for pid, l in labels.items() if l == lbl]
            vecs = np.stack([space.get(pid).as_vector(order) for pid in members])
            centroids[lbl] = vecs.mean(axis=0)

        def make_membership(lbl, all_centroids):
            def fn(x: np.ndarray) -> bool:
                dists = {k: np.linalg.norm(x - c) for k, c in all_centroids.items()}
                return min(dists, key=dists.get) == lbl
            return fn

        for lbl in unique_labels:
            phenotypes.append(
                Phenotype(
                    name=f"{self.name}_cluster_{lbl}",
                    membership_fn=make_membership(lbl, centroids),
                )
            )
        return phenotypes


class KMeansPhenotypeEstimator(PhenotypeEstimator):
    """Implementação concreta de PhenotypeEstimator usando sklearn.KMeans."""

    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(name="kmeans")
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, space: RepresentationSpace, order: Optional[Sequence[str]] = None) -> Dict[str, int]:
        from sklearn.cluster import KMeans

        M, ids = space.matrix()
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(M)
        return dict(zip(ids, labels.tolist()))


# =============================================================================
# 4.10  Coorte  ->  C = {Γ_1, Γ_2, ..., Γ_n}
# =============================================================================
class Cohort:
    """
    Uma coleção de trajetórias — não uma tabela estática. Adicionar uma
    nova observação a um paciente existente apenas atualiza sua trajetória
    (Seção 9.4 / 9.8), sem reconstrução da base.
    """

    def __init__(self, name: str, representation: Representation):
        self.name = name
        self.representation = representation
        self.trajectories: Dict[str, Trajectory] = {}
        self.patients: Dict[str, BiologicalSystem] = {}

    def add_observation(self, B: BiologicalSystem, raw_data: dict, timestamp: Optional[datetime] = None) -> PatientRepresentation:
        """
        Ingesta uma nova observação bruta, recomputa a representação e
        atualiza a trajetória do paciente correspondente (operador U da
        Seção 9.3: Γ' = U(Γ, O_novo)).
        """
        ts = timestamp or datetime.now()
        B.ingest(ts, raw_data)
        self.patients[B.id] = B

        rep = self.representation.compute(B, ts)

        if B.id not in self.trajectories:
            self.trajectories[B.id] = Trajectory(B.id)
        self.trajectories[B.id].update(rep)
        return rep

    def snapshot_space(self, at_index: int = -1, order: Optional[Sequence[str]] = None) -> RepresentationSpace:
        """
        Constrói um RepresentationSpace com a representação mais recente
        (ou em um índice específico) de cada paciente da coorte.
        """
        space = RepresentationSpace(domain_order=order)
        for pid, traj in self.trajectories.items():
            if len(traj) == 0:
                continue
            idx = at_index if -len(traj) <= at_index < len(traj) else -1
            space.add(traj.at(idx))
        return space

    def transition_matrix(self, phenotypes: List[Phenotype], order: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        P_ij = P(F_j | F_i)  (Seção 8.8 / 9.6): estima a matriz de transição
        fenotípica a partir das sequências longitudinais de todos os
        pacientes da coorte.
        """
        names = [ph.name for ph in phenotypes]
        idx = {n: i for i, n in enumerate(names)}
        counts = np.zeros((len(names), len(names)))

        for traj in self.trajectories.values():
            seq = traj.phenotype_sequence(phenotypes, order)
            for a, b in zip(seq, seq[1:]):
                if a is not None and b is not None:
                    counts[idx[a], idx[b]] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = counts / row_sums
        return P, names

    def __len__(self) -> int:
        return len(self.trajectories)

    def __repr__(self) -> str:
        return f"Cohort({self.name!r}, n_patients={len(self)})"


# =============================================================================
# 5.  Contratos formais (verificação automática de propriedades da Seção 5)
# =============================================================================
class ContractViolation(Exception):
    pass


class Contracts:
    """
    Utilitários para verificar, empiricamente, alguns dos contratos formais
    definidos na Seção 5 do artigo. Não são provas matemáticas — são testes
    de sanidade sobre uma implementação concreta.
    """

    @staticmethod
    def check_reproducibility(operator: RepresentationOperator, B: BiologicalSystem) -> bool:
        """Contrato 5.8: R(O1) = R(O2) se O1 = O2 (determinismo)."""
        x1 = operator.apply(B)
        x2 = operator.apply(B)
        return np.allclose(x1, x2)

    @staticmethod
    def check_semantic_preservation(
        operator: RepresentationOperator, B1: BiologicalSystem, B2: BiologicalSystem, tol: float = 1e-9
    ) -> bool:
        """Contrato 5.2: estados fisiológicos diferentes -> representações diferentes."""
        s1 = B1.latest_raw_state()
        s2 = B2.latest_raw_state()
        if s1 == s2:
            return True  # estados iguais: não há o que preservar
        x1 = operator.apply(B1)
        x2 = operator.apply(B2)
        return not np.allclose(x1, x2, atol=tol)

    @staticmethod
    def check_lipschitz_continuity(
        operator: RepresentationOperator,
        pairs: Sequence[Tuple[BiologicalSystem, BiologicalSystem]],
        raw_distance_fn: Callable[[BiologicalSystem, BiologicalSystem], float],
    ) -> float:
        """
        Contrato 5.4: estima a constante de Lipschitz empírica
        L = max_i d_X(R(B1),R(B2)) / d_B(B1,B2) sobre um conjunto de pares.
        Valores finitos e estáveis sugerem continuidade adequada.
        """
        ratios = []
        for B1, B2 in pairs:
            dB = raw_distance_fn(B1, B2)
            if dB == 0:
                continue
            dX = float(np.linalg.norm(operator.apply(B1) - operator.apply(B2)))
            ratios.append(dX / dB)
        return max(ratios) if ratios else float("nan")

    @staticmethod
    def check_extensibility(rep: Representation, new_operator: RepresentationOperator, B: BiologicalSystem) -> bool:
        """Contrato 5.5: R' = R ⊕ R_{n+1} deve preservar componentes antigos."""
        old = rep.compute(B)
        extended = rep.extend(new_operator)
        new = extended.compute(B)
        for name, vec in old.components.items():
            if name not in new.components or not np.allclose(vec, new.components[name]):
                return False
        return True
