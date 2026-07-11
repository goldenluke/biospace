"""
tests.test_graph_curvature
==============================

Fase 8 — Geometria (terceira forma de curvatura: sobre a variedade/grafo,
ver `biospace.geometry.graph_curvature` para como se relaciona com as
outras duas já existentes em `biospace.geometry.curvature`).

Validação em 3 casos sintéticos com comportamento CONHECIDO da
literatura de Ollivier-Ricci antes de confiar em qualquer resultado real:

  1. Grafo completo -> curvatura fortemente POSITIVA.
  2. Ciclo grande -> curvatura EXATAMENTE ZERO (resultado clássico,
     bem citado na literatura de curvatura discreta).
  3. Árvore binária -> achado ao investigar um resultado agregado
     inicialmente surpreendente (levemente positivo): arestas INTERNAS
     (backbone, grau>1 nos dois lados) são NEGATIVAS como esperado;
     arestas de FOLHA são positivas (artefato conhecido da caminhada
     preguiçosa em nós de grau 1) — a média global mistura as duas.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from biospace.geometry import graph_curvature_summary, ollivier_ricci_curvature


def test_complete_graph_has_strongly_positive_curvature():
    K5 = nx.complete_graph(5)
    resumo = graph_curvature_summary(K5, weight=None)
    assert resumo["global_mean"] > 0.4, "Grafo completo (vizinhanças totalmente sobrepostas) deveria ter curvatura fortemente positiva."


def test_cycle_graph_has_zero_curvature():
    """Resultado clássico e bem citado na literatura de Ollivier-Ricci: um ciclo grande tem curvatura exatamente 0."""
    ciclo = nx.cycle_graph(20)
    resumo = graph_curvature_summary(ciclo, weight=None)
    assert resumo["global_mean"] == pytest.approx(0.0, abs=1e-9)


def test_tree_internal_edges_are_negative_leaf_edges_are_positive():
    """
    Achado ao validar: a curvatura média de TODA a árvore (0,033) parecia
    positiva demais para uma estrutura de 'gargalo' — investigando por
    tipo de aresta, confirma-se que arestas internas (backbone) SÃO
    negativas como esperado; a média global só mistura com arestas de
    folha (positivas, artefato conhecido do método).
    """
    arvore = nx.balanced_tree(2, 4)
    curvaturas = ollivier_ricci_curvature(arvore, weight=None)
    graus = dict(arvore.degree())

    internas = [k for (u, v), k in curvaturas.items() if graus[u] > 1 and graus[v] > 1]
    folhas = [k for (u, v), k in curvaturas.items() if graus[u] == 1 or graus[v] == 1]

    assert len(internas) > 0 and len(folhas) > 0
    assert np.mean(internas) < 0, "Arestas internas (backbone) de uma árvore deveriam ter curvatura negativa (gargalo estrutural)."
    assert np.mean(folhas) > 0, "Arestas de folha deveriam ter curvatura positiva (artefato conhecido da caminhada preguiçosa em nós de grau 1)."


def test_graph_curvature_handles_graph_without_edges():
    grafo_vazio = nx.Graph()
    grafo_vazio.add_node("sozinho")
    resumo = graph_curvature_summary(grafo_vazio, weight=None)
    assert resumo["edge_curvatures"] == {}
    assert np.isnan(resumo["global_mean"])


def _two_cluster_graph_with_bridge(n_per_cluster=15, seed=0):
    """2 clusters densos internamente, poucas arestas de ponte entre eles -- as pontes deveriam ter curvatura mais baixa."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    cluster_a = [f"a{i}" for i in range(n_per_cluster)]
    cluster_b = [f"b{i}" for i in range(n_per_cluster)]
    G.add_nodes_from(cluster_a + cluster_b)

    for cluster in (cluster_a, cluster_b):
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if rng.random() < 0.4:
                    G.add_edge(cluster[i], cluster[j])
    for _ in range(3):
        u = rng.choice(cluster_a)
        v = rng.choice(cluster_b)
        G.add_edge(u, v)

    labels = {n: "A" for n in cluster_a} | {n: "B" for n in cluster_b}
    return G, labels


def test_bridge_edges_between_clusters_have_lower_curvature_than_within_cluster_edges():
    """
    O TESTE DECISIVO: reproduz em miniatura o achado real deste projeto
    nos dados de SAOS -- arestas que CRUZAM clusters (pontes estruturais)
    devem ter curvatura menor que arestas DENTRO do mesmo cluster.
    """
    G, labels = _two_cluster_graph_with_bridge()
    curvaturas = ollivier_ricci_curvature(G, weight=None)

    dentro = [k for (u, v), k in curvaturas.items() if labels[u] == labels[v]]
    entre = [k for (u, v), k in curvaturas.items() if labels[u] != labels[v]]

    assert len(entre) > 0, "O grafo de teste deveria ter pelo menos uma aresta de ponte entre clusters."
    assert np.mean(dentro) > np.mean(entre), (
        f"Arestas dentro do cluster deveriam ter curvatura maior que as de ponte "
        f"(dentro={np.mean(dentro):.3f}, entre={np.mean(entre):.3f})"
    )
