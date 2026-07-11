"""
tests.test_graph
====================

Fase 7 (parte 1 — o grafo, sem a GNN, deliberadamente): o paciente
deixa de ser vetor e passa a ser rede. Dois níveis testados:

  1. `build_patient_graph` — rede INTERNA de um paciente (domínios,
     Features, comorbidades presentes, correlações reais entre eixos).
  2. `build_cohort_similarity_graph` — rede de PACIENTES (a estrutura
     que uma GNN futura consumiria) — validada contra uma estrutura de
     cluster sintética CONHECIDA, não só "roda sem erro".
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.geometry import Euclidean
from biospace.graph import build_cohort_similarity_graph, build_patient_graph, compute_feature_correlations, to_pyg_arrays


class _XObservable(Observable):
    key = "x"


class _YObservable(Observable):
    key = "y"


class _ToyDomain(SemanticDomain):
    name = "toy"

    def __init__(self):
        super().__init__([_XObservable(), _YObservable()])

    def encode(self, measurements):
        return [
            Feature(name="x", value=float(measurements["x"].value)),
            Feature(name="y", value=float(measurements["y"].value)),
        ]


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


class _ComorbidityLikeDomain(SemanticDomain):
    name = "comorbidity"

    def __init__(self):
        super().__init__([_FlagObservable("hipertensao")])

    def encode(self, measurements):
        m = measurements.get("hipertensao")
        v = float(m.value) if m and not m.is_missing() else 0.0
        return [Feature(name="hipertensao", value=v, raw_value=v)]


def _toy_cohort(n: int = 60, seed: int = 0):
    """2 clusters bem separados em (x,y) -- estrutura CONHECIDA para validar o grafo de similaridade."""
    rng = np.random.default_rng(seed)
    representation = Representation([_ToyDomain(), _ComorbidityLikeDomain()])
    cohort = Cohort()
    labels = {}
    t0 = datetime(2024, 1, 1)

    for i in range(n // 2):
        system = BiologicalSystem(identifier=f"grupoA_{i}")
        system.observe(Observation(timestamp=t0, source="teste", values={"x": rng.normal(0, 0.3), "y": rng.normal(0, 0.3), "hipertensao": 0.0}))
        cohort.update(system, representation, timestamp=t0)
        labels[system.id] = "A"
    for i in range(n // 2):
        system = BiologicalSystem(identifier=f"grupoB_{i}")
        system.observe(Observation(timestamp=t0, source="teste", values={"x": rng.normal(10, 0.3), "y": rng.normal(10, 0.3), "hipertensao": 1.0}))
        cohort.update(system, representation, timestamp=t0)
        labels[system.id] = "B"

    return cohort, representation, labels


def test_build_patient_graph_has_expected_node_kinds():
    cohort, representation, _ = _toy_cohort(n=4)
    system = next(iter(cohort.systems.values()))
    G = build_patient_graph(system, representation)

    kinds = {attrs["kind"] for _, attrs in G.nodes(data=True)}
    assert kinds == {"patient", "domain", "feature"} or kinds == {"patient", "domain", "feature", "entity"}

    n_domains = sum(1 for _, a in G.nodes(data=True) if a["kind"] == "domain")
    assert n_domains == 2  # toy + comorbidity


def test_build_patient_graph_creates_entity_node_only_when_flag_present():
    cohort, representation, labels = _toy_cohort(n=4)
    sid_a = next(sid for sid in labels if labels[sid] == "A")  # hipertensao=0
    sid_b = next(sid for sid in labels if labels[sid] == "B")  # hipertensao=1

    G_a = build_patient_graph(cohort.systems[sid_a], representation)
    G_b = build_patient_graph(cohort.systems[sid_b], representation)

    assert ("entity", "hipertensao") not in G_a.nodes
    assert ("entity", "hipertensao") in G_b.nodes


def test_feature_correlations_are_in_valid_range_and_deduplicated():
    cohort, representation, _ = _toy_cohort(n=60)
    space = cohort.snapshot()
    correlations = compute_feature_correlations(space)

    assert len(correlations) > 0
    for (a, b), corr in correlations.items():
        assert -1.0 - 1e-6 <= corr <= 1.0 + 1e-6
        assert (b, a) not in correlations, "Cada par deveria aparecer só uma vez (a,b), não também (b,a)."


def test_patient_graph_uses_real_correlations_not_arbitrary_threshold_noise():
    """
    Achado ao escrever este teste: numa população com 2 grupos bem
    separados, x e y correlacionam >0.9 mesmo sendo independentes DENTRO
    de cada grupo — correlação "ecológica" (populacional, entre grupos),
    não um bug. Por isso este teste usa uma população de UM grupo só,
    com x e y genuinamente independentes, para checar a ausência de
    aresta espúria sem essa interferência.
    """
    rng = np.random.default_rng(2)
    representation = Representation([_ToyDomain(), _ComorbidityLikeDomain()])
    cohort = Cohort()
    t0 = datetime(2024, 1, 1)
    for i in range(60):
        system = BiologicalSystem(identifier=f"p{i}")
        system.observe(Observation(timestamp=t0, source="teste", values={"x": rng.normal(0, 1), "y": rng.normal(0, 1), "hipertensao": 0.0}))
        cohort.update(system, representation, timestamp=t0)

    space = cohort.snapshot()
    correlations = compute_feature_correlations(space)

    system = next(iter(cohort.systems.values()))
    G = build_patient_graph(system, representation, feature_correlations=correlations, correlation_threshold=0.5)

    correlate_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("relation") == "CORRELATES_WITH"]
    assert (("feature", "toy.x"), ("feature", "toy.y")) not in correlate_edges
    assert (("feature", "toy.y"), ("feature", "toy.x")) not in correlate_edges


def test_similarity_graph_recovers_known_cluster_structure():
    """
    O TESTE DECISIVO: com 2 clusters bem separados (grupoA perto de
    (0,0), grupoB perto de (10,10)), os vizinhos no grafo de
    similaridade devem compartilhar o MESMO grupo quase sempre --
    muito mais que um baseline aleatório.
    """
    cohort, representation, labels = _toy_cohort(n=60, seed=1)
    space = cohort.snapshot()
    order = space.order()

    G = build_cohort_similarity_graph(space, Euclidean(), k=5, order=order, node_labels=labels)

    acertos, total = 0, 0
    for sid in G.nodes():
        for vizinho in G.neighbors(sid):
            total += 1
            if labels[sid] == labels[vizinho]:
                acertos += 1

    fracao = acertos / total
    assert fracao > 0.95, f"Esperava quase todos os vizinhos no mesmo cluster (clusters bem separados), achou {fracao:.2f}"


def test_similarity_graph_node_labels_are_attached():
    cohort, representation, labels = _toy_cohort(n=10)
    space = cohort.snapshot()
    G = build_cohort_similarity_graph(space, Euclidean(), k=3, order=space.order(), node_labels=labels)
    for sid in G.nodes():
        assert G.nodes[sid]["label"] == labels[sid]


def test_to_pyg_arrays_shapes_are_consistent():
    cohort, representation, _ = _toy_cohort(n=10)
    system = next(iter(cohort.systems.values()))
    G = build_patient_graph(system, representation)

    arrays = to_pyg_arrays(G)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    assert arrays["node_features"].shape == (n_nodes, 2)
    assert arrays["edge_index"].shape == (2, n_edges * 2)  # nao dirigido -> 2x
    assert arrays["node_labels"].shape == (n_nodes,)
    assert arrays["edge_index"].max() < n_nodes
