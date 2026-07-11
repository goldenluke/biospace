"""
biospace.graph.patient_graph
================================

build_patient_graph: o paciente deixa de ser um vetor (`RepresentationVector`)
e passa a ser uma REDE heterogênea — nós para o paciente, cada domínio,
cada Feature, e cada comorbidade/sintoma/tratamento PRESENTE; arestas
conectando-os com significado explícito (não um grafo genérico, um
grafo TIPADO: cada aresta tem uma relação nomeada).

    Hoje:   patient.vector()  ->  x ∈ X (um ponto)
    Depois: patient.graph()   ->  G = (V, E) (uma rede)

O que faz isso ser "conhecimento" e não só "estrutura": as arestas
Feature-Feature (correlação) vêm da CORRELAÇÃO POPULACIONAL REAL entre
os eixos — relação empírica, não uma ontologia inventada. Isso é
DELIBERADAMENTE mais modesto que um "grafo de conhecimento médico" no
sentido de UMLS/SNOMED (que exigiria uma base de conhecimento clínico
externa, fora do escopo deste projeto) — aqui, "conhecimento" significa
"relações que os próprios dados revelam", consistente com o resto do
projeto (nunca inventar o que não foi validado).

PRÓXIMO PASSO EXPLICITAMENTE FORA DE ESCOPO AQUI (Fase 7, "depois GNN"):
este módulo constrói o grafo; NÃO treina nenhuma rede neural sobre ele.
Ver `to_pyg_arrays()` para o formato de exportação que uma GNN de
verdade (PyTorch Geometric, DGL, ...) consumiria — construído para que
o próximo passo seja direto, sem pretender já ser esse passo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from biospace.core import BiologicalSystem, Representation

__all__ = ["build_patient_graph", "to_pyg_arrays"]

_COMORBIDITY_LIKE_DOMAINS = {"comorbidity", "treatment"}


def build_patient_graph(
    system: "BiologicalSystem",
    representation: "Representation",
    feature_correlations: Optional[dict[tuple[str, str], float]] = None,
    correlation_threshold: float = 0.5,
) -> nx.Graph:
    """
    Constrói a rede de UM paciente.

    Nós:
      - `("patient", system.id)` — o próprio paciente.
      - `("domain", nome)` — um por SemanticDomain da Representation.
      - `("feature", nome_qualificado)` — um por Feature computada.
      - Domínios binários (`comorbidity`/`treatment`): cada Feature com
        valor > 0 vira também um nó de entidade clínica presente
        (`("entity", nome)`), com aresta `patient -[HAS]-> entity`.

    Arestas:
      - `patient -[OBSERVED {value, is_missing}]-> feature`
      - `feature -[BELONGS_TO]-> domain`
      - `feature -[CORRELATES_WITH {weight=correlacao}]-> feature` — SÓ
        se `feature_correlations` for informado e |correlação| >=
        `correlation_threshold` — relação EMPÍRICA (da população), não
        inventada.

    `feature_correlations`: dict `{(nome_a, nome_b): correlacao}` — ver
    `biospace.graph.similarity_graph.compute_feature_correlations()`
    para construir isso a partir de um `RepresentationSpace`.
    """
    vector = representation.transform(system)
    order = representation.domain_names()

    G = nx.Graph()
    patient_node = ("patient", system.id)
    G.add_node(patient_node, kind="patient")

    for domain_name in order:
        domain_node = ("domain", domain_name)
        G.add_node(domain_node, kind="domain", name=domain_name)

        for feature in vector.components.get(domain_name, []):
            feature_node = ("feature", f"{domain_name}.{feature.name}")
            G.add_node(
                feature_node,
                kind="feature",
                domain=domain_name,
                name=feature.name,
                value=feature.value,
                raw_value=feature.raw_value,
                is_missing=feature.is_missing,
            )
            G.add_edge(feature_node, domain_node, relation="BELONGS_TO")
            G.add_edge(patient_node, feature_node, relation="OBSERVED", value=feature.value, is_missing=feature.is_missing)

            if domain_name in _COMORBIDITY_LIKE_DOMAINS and (feature.raw_value or 0) > 0:
                entity_node = ("entity", feature.name)
                G.add_node(entity_node, kind="entity", name=feature.name, source_domain=domain_name)
                G.add_edge(patient_node, entity_node, relation="HAS")

    if feature_correlations:
        for (name_a, name_b), corr in feature_correlations.items():
            if abs(corr) < correlation_threshold:
                continue
            node_a, node_b = ("feature", name_a), ("feature", name_b)
            if node_a in G and node_b in G:
                G.add_edge(node_a, node_b, relation="CORRELATES_WITH", weight=corr)

    return G


def to_pyg_arrays(graph: nx.Graph) -> dict[str, np.ndarray]:
    """
    Exporta `graph` no formato cru que bibliotecas de GNN (PyTorch
    Geometric, DGL) esperam: uma matriz de features por nó e um
    `edge_index` (formato COO, 2 x nº de arestas). NÃO constrói nenhum
    objeto de framework específico (evita depender de PyTorch/DGL aqui)
    — só os arrays numpy que alimentariam um `torch_geometric.data.Data`
    ou equivalente, deixando esse próximo passo (Fase 7, GNN) pronto para
    ser conectado, sem pretender já ser esse passo.
    """
    nodes = list(graph.nodes())
    index_of = {node: i for i, node in enumerate(nodes)}

    node_features = []
    for node in nodes:
        attrs = graph.nodes[node]
        value = attrs.get("value", 0.0) if attrs.get("kind") == "feature" else 0.0
        is_missing = 1.0 if attrs.get("is_missing") else 0.0
        node_features.append([float(value) if value is not None else 0.0, is_missing])

    edges = []
    for u, v in graph.edges():
        edges.append((index_of[u], index_of[v]))
        edges.append((index_of[v], index_of[u]))  # nao dirigido -> ambas as direcoes, convencao usual de GNN

    return {
        "node_features": np.array(node_features, dtype=float),
        "edge_index": np.array(edges, dtype=int).T if edges else np.zeros((2, 0), dtype=int),
        "node_labels": np.array([str(n) for n in nodes], dtype=object),
    }
