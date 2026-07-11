"""
examples/07_knowledge_graph.py
=================================

Fase 7 (parte 1 -- o grafo; GNN deliberadamente NAO implementada aqui,
"depois" como o proprio pedido especificou):

    Hoje:   patient.vector() -> x em X (um ponto)
    Depois: patient.graph()  -> G = (V, E) (uma rede)

Dois niveis: a rede INTERNA de um paciente (dominios, Features,
comorbidades presentes, correlacoes reais entre eixos) e a rede de
PACIENTES (similaridade populacional -- a estrutura que uma GNN
consumiria depois).

Rode com: python3 examples/07_knowledge_graph.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import numpy as np

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
        return [Feature(name="x", value=float(measurements["x"].value)), Feature(name="y", value=float(measurements["y"].value))]


def build_two_cluster_cohort(n=60, seed=1):
    rng = np.random.default_rng(seed)
    representation = Representation([_ToyDomain()])
    cohort = Cohort()
    labels = {}
    t0 = datetime(2024, 1, 1)
    for i in range(n // 2):
        s = BiologicalSystem(identifier=f"grupoA_{i}")
        s.observe(Observation(timestamp=t0, source="teste", values={"x": rng.normal(0, 0.3), "y": rng.normal(0, 0.3)}))
        cohort.update(s, representation, timestamp=t0)
        labels[s.id] = "A"
    for i in range(n // 2):
        s = BiologicalSystem(identifier=f"grupoB_{i}")
        s.observe(Observation(timestamp=t0, source="teste", values={"x": rng.normal(10, 0.3), "y": rng.normal(10, 0.3)}))
        cohort.update(s, representation, timestamp=t0)
        labels[s.id] = "B"
    return cohort, representation, labels


def main():
    cohort, representation, labels = build_two_cluster_cohort()
    space = cohort.snapshot()
    order = space.order()

    print("--- Rede INTERNA de um paciente (patient.graph()) ---")
    correlations = compute_feature_correlations(space)
    system = next(iter(cohort.systems.values()))
    G_patient = build_patient_graph(system, representation, feature_correlations=correlations, correlation_threshold=0.5)
    print(f"{G_patient.number_of_nodes()} nos, {G_patient.number_of_edges()} arestas")
    for u, v, attrs in G_patient.edges(data=True):
        print(f"  {u} --[{attrs['relation']}]--> {v}")
    print()

    print("--- Exportacao para formato de GNN (edge_index, node_features) ---")
    arrays = to_pyg_arrays(G_patient)
    print(f"node_features: {arrays['node_features'].shape}, edge_index: {arrays['edge_index'].shape}")
    print("(Isto e o INSUMO que uma GNN de verdade consumiria -- Fase 7, GNN em si, deliberadamente nao implementada aqui)")
    print()

    print("--- Rede de PACIENTES (similaridade populacional) ---")
    G_cohort = build_cohort_similarity_graph(space, Euclidean(), k=5, order=order, node_labels=labels)
    acertos, total = 0, 0
    for sid in G_cohort.nodes():
        for vizinho in G_cohort.neighbors(sid):
            total += 1
            if labels[sid] == labels[vizinho]:
                acertos += 1
    print(f"{G_cohort.number_of_nodes()} pacientes, {G_cohort.number_of_edges()} arestas de similaridade")
    print(f"Vizinhos compartilham grupo: {acertos}/{total} ({100*acertos/total:.1f}%) -- clusters bem separados, deveria ser quase 100%")


if __name__ == "__main__":
    main()
