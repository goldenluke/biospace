"""
biospace.topology.mapper
============================

Algoritmo Mapper (Singh, Mémoli & Carlsson, 2007) sobre um
`RepresentationSpace` — envelope fino sobre `kmapper`. Produz um grafo
que resume a forma do espaço de representação: cada nó é um cluster de
pacientes numa faixa sobreposta da "lente" (uma projeção escalar, ou de
baixa dimensão, do espaço); arestas conectam nós que compartilham
pacientes nas faixas sobrepostas. Diferente de um grafo de k-vizinhos
(usado no resto do projeto para GNN/curvatura), a estrutura de Mapper
é sensível à direção da lente escolhida — a mesma população pode
produzir grafos com formas topologicamente diferentes sob lentes
diferentes, e essa escolha deve ser reportada, nunca implícita.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["MapperResult", "compute_mapper_graph"]


@dataclass
class MapperResult:
    nodes: dict[str, list[int]]  # nome do no -> indices (na ordem de `ids`) dos pacientes membros
    links: dict[str, list[str]]  # nome do no -> nomes dos nos vizinhos
    ids: list[str]
    lens_description: str

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return sum(len(v) for v in self.links.values()) // 2

    def members(self, node_name: str) -> list[str]:
        """system_id dos pacientes membros de um no."""
        return [self.ids[i] for i in self.nodes[node_name]]

    def summary(self) -> str:
        return f"MapperResult (lente={self.lens_description}): {self.n_nodes()} nós, {self.n_edges()} arestas, {len(self.ids)} pacientes."


def compute_mapper_graph(
    space: "RepresentationSpace",
    order: Optional[list[str]] = None,
    lens: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lens_description: str = "norma L2 (distância euclidiana à origem no espaço já em z-score — quão extremo o paciente é, agregando todas as Features)",
    n_cubes: int = 10,
    perc_overlap: float = 0.4,
) -> MapperResult:
    """
    `lens`: função que recebe a matriz (n_pacientes, n_features) e
    devolve um vetor 1-D (a projeção escalar usada para cobrir o espaço
    em faixas sobrepostas). Se não passada, usa a 1ª componente
    principal — uma escolha comum, mas nunca a única razoável;
    `lens_description` deveria sempre ser atualizada para refletir o
    que de fato foi usado, para não deixar implícita qual estrutura o
    grafo resultante está de fato resumindo.
    """
    import kmapper as km

    order = order or space.order()
    matrix, ids = space.matrix()

    mapper = km.KeplerMapper(verbose=0)
    if lens is None:
        projecao = mapper.fit_transform(matrix, projection="l2norm")
    else:
        projecao = lens(matrix).reshape(-1, 1)

    grafo_bruto = mapper.map(projecao, matrix, cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap))

    links = {no: list(vizinhos) for no, vizinhos in grafo_bruto["links"].items()}
    for no, vizinhos in list(links.items()):
        for v in vizinhos:
            if v not in links:
                links[v] = []
            if no not in links[v]:
                links[v].append(no)

    return MapperResult(nodes=grafo_bruto["nodes"], links=links, ids=list(ids), lens_description=lens_description)
