"""
biospace.graph.similarity_graph
==================================

build_cohort_similarity_graph: a rede de PACIENTES (não a rede interna
de um único paciente — ver `patient_graph.py`) — nós = pacientes,
arestas = similaridade (k vizinhos mais próximos, sob qualquer
`Geometry` já existente no projeto). Esta é a estrutura que uma GNN
(Fase 7, deliberadamente NÃO implementada aqui) consumiria para
aprendizado populacional — "message passing" entre pacientes
parecidos —, não a rede interna de um paciente.

compute_feature_correlations: correlação populacional real entre
Features — o insumo empírico para as arestas Feature-Feature de
`build_patient_graph()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace
    from biospace.geometry import Geometry

__all__ = ["build_cohort_similarity_graph", "compute_feature_correlations"]


def compute_feature_correlations(space: "RepresentationSpace", order: Optional[Sequence[str]] = None) -> dict[tuple[str, str], float]:
    """
    Correlação de Pearson entre cada par de Features, sobre toda a
    população de `space` — a relação EMPÍRICA usada como aresta
    Feature-Feature em `build_patient_graph()`.
    """
    matrix, ids = space.matrix()
    used_order = order or space.order()

    names: list[str] = []
    for domain_name in used_order:
        vec = space.get(ids[0])
        for f in vec.components.get(domain_name, []):
            names.append(f"{domain_name}.{f.name}")

    if matrix.shape[1] != len(names):
        raise ValueError(
            f"Nº de colunas da matriz ({matrix.shape[1]}) não bate com o nº de nomes de Feature "
            f"resolvidos ({len(names)}) — `order` deve corresponder aos domínios reais de `space`."
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(matrix, rowvar=False)
    correlations: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = corr_matrix[i, j]
            if np.isfinite(c):
                correlations[(names[i], names[j])] = float(c)
    return correlations


def build_cohort_similarity_graph(
    space: "RepresentationSpace",
    geometry: "Geometry",
    k: int = 5,
    order: Optional[Sequence[str]] = None,
    node_labels: Optional[dict[str, str]] = None,
) -> nx.Graph:
    """
    Rede de similaridade entre pacientes: nó por paciente, aresta para
    cada um dos `k` vizinhos mais próximos (sob `geometry`) — não
    necessariamente simétrico par a par (A pode estar entre os k
    vizinhos de B sem o inverso valer), mas o grafo resultante é
    não-dirigido (uma aresta é adicionada se QUALQUER um dos dois
    considerar o outro um vizinho próximo).

    `node_labels`: rótulo opcional por paciente (ex.: nome do fenótipo)
    — vira atributo do nó, útil para depois medir se o grafo respeita
    uma estrutura conhecida (ver validação no README/testes: fração de
    vizinhos que compartilham o mesmo fenótipo, comparada a um baseline
    aleatório).
    """
    ids = space.ids()
    used_order = order or space.order()
    vectors = {sid: space.get(sid).as_vector(used_order) for sid in ids}

    G = nx.Graph()
    for sid in ids:
        attrs = {"kind": "patient"}
        if node_labels and sid in node_labels:
            attrs["label"] = node_labels[sid]
        G.add_node(sid, **attrs)

    for sid in ids:
        distances = [(other, geometry.distance(vectors[sid], vectors[other])) for other in ids if other != sid]
        distances.sort(key=lambda pair: pair[1])
        for neighbor, dist in distances[:k]:
            if G.has_edge(sid, neighbor):
                G[sid][neighbor]["weight"] = min(G[sid][neighbor]["weight"], dist)
            else:
                G.add_edge(sid, neighbor, weight=dist, relation="SIMILAR_TO")

    return G
