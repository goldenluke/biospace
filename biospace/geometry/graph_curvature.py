"""
biospace.geometry.graph_curvature
====================================

Fase 8 — Geometria: a TERCEIRA forma de estimar curvatura neste projeto
(ver `curvature.py` para as outras duas — temporal via
`FeatureDynamics.curvature` e transversal via `estimate_density_curvature`),
desta vez sobre a VARIEDADE em si (o grafo k-NN que `RiemannianGeometry`
já constrói para aproximar geodésicas) — não sobre a dinâmica temporal
de uma Feature nem sobre a distribuição populacional de uma Feature
isolada, mas sobre a ESTRUTURA RELACIONAL do espaço inteiro.

CURVATURA DE OLLIVIER-RICCI (Ollivier, 2009): para uma aresta (u,v) do
grafo, compara as vizinhanças de u e v via TRANSPORTE ÓTIMO — a mesma
matemática de Wasserstein já usada em `wasserstein.py` e
`gromov_wasserstein.py`, aqui entre distribuições de vizinhança de nós,
não entre Features de pacientes:

    κ(u,v) = 1 - W1(μ_u, μ_v) / d(u,v)

κ > 0 — vizinhanças sobrepostas, região "costurada", tipicamente mais
ESTÁVEL (perturbação se dissipa, caminhos alternativos abundantes). κ < 0
— vizinhanças pouco sobrepostas, "gargalo" estrutural, tipicamente mais
FRÁGIL. Por isso Ollivier-Ricci foi proposta na literatura como sinal de
alerta precoce em redes biológicas/financeiras — curvatura decrescente
pode sinalizar perda de resiliência antes de uma transição crítica.

RELAÇÃO COM AS OUTRAS DUAS CURVATURAS: são três medidas INDEPENDENTES,
de fontes de dados diferentes (temporal-por-paciente,
densidade-populacional-de-1-Feature, estrutura-relacional-do-grafo).
Não se espera que coincidam numericamente — mas se concordarem em
DIREÇÃO (quais pacientes/regiões são mais/menos estáveis), é evidência
adicional de que capturam algo real. Ver `tests/test_graph_curvature.py`
para essa checagem cruzada nos dados reais.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

__all__ = ["ollivier_ricci_curvature", "graph_curvature_summary"]


def _lazy_neighborhood_distribution(graph: nx.Graph, node, alpha: float) -> dict:
    """Caminhada preguiçosa: massa alpha em `node`, (1-alpha) dividida uniformemente entre seus vizinhos."""
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return {node: 1.0}
    dist = {node: alpha}
    mass_per_neighbor = (1 - alpha) / len(neighbors)
    for n in neighbors:
        dist[n] = dist.get(n, 0.0) + mass_per_neighbor
    return dist


def ollivier_ricci_curvature(graph: nx.Graph, alpha: float = 0.5, weight: Optional[str] = "weight") -> dict[tuple, float]:
    """
    Curvatura de Ollivier-Ricci para CADA aresta de `graph`. `alpha`:
    massa que fica no próprio nó na caminhada preguiçosa (0.5 é o padrão
    usual na literatura). `weight`: atributo de aresta usado como
    distância de base para o custo do transporte ótimo (None = todas as
    arestas com distância 1).

    Retorna `{(u, v): curvatura}` — uma entrada por aresta de `graph`.
    """
    import ot

    if weight is not None and not nx.get_edge_attributes(graph, weight):
        weight = None

    all_distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight))

    curvatures: dict[tuple, float] = {}
    for u, v in graph.edges():
        mu_u = _lazy_neighborhood_distribution(graph, u, alpha)
        mu_v = _lazy_neighborhood_distribution(graph, v, alpha)

        support_u = list(mu_u.keys())
        support_v = list(mu_v.keys())
        mass_u = np.array([mu_u[n] for n in support_u])
        mass_v = np.array([mu_v[n] for n in support_v])

        cost_matrix = np.array([[all_distances[a].get(b, np.inf) for b in support_v] for a in support_u])
        if np.isinf(cost_matrix).any():
            finite_max = cost_matrix[np.isfinite(cost_matrix)].max() if np.isfinite(cost_matrix).any() else 1.0
            cost_matrix = np.where(np.isinf(cost_matrix), finite_max * 10, cost_matrix)

        w1 = ot.emd2(mass_u, mass_v, cost_matrix)
        d_uv = all_distances[u].get(v, 1.0)
        curvatures[(u, v)] = 1.0 - w1 / d_uv if d_uv > 0 else 0.0

    return curvatures


def graph_curvature_summary(graph: nx.Graph, alpha: float = 0.5, weight: Optional[str] = "weight") -> dict:
    """Resumo agregado (por nó: média da curvatura das arestas que o tocam; global: média/min/max) — para inspeção rápida."""
    edge_curvatures = ollivier_ricci_curvature(graph, alpha=alpha, weight=weight)
    per_node: dict = {n: [] for n in graph.nodes()}
    for (u, v), k in edge_curvatures.items():
        per_node[u].append(k)
        per_node[v].append(k)

    node_mean_curvature = {n: float(np.mean(ks)) if ks else float("nan") for n, ks in per_node.items()}
    values = list(edge_curvatures.values())
    return {
        "edge_curvatures": edge_curvatures,
        "node_mean_curvature": node_mean_curvature,
        "global_mean": float(np.mean(values)) if values else float("nan"),
        "global_min": float(np.min(values)) if values else float("nan"),
        "global_max": float(np.max(values)) if values else float("nan"),
    }
