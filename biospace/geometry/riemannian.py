"""
biospace.geometry.riemannian
===============================

RiemannianGeometry: o espaço de representação DEIXA DE SER PLANO. Em vez
de assumir que a menor distância entre dois pacientes é sempre uma linha
reta (Euclidean) ou uma elipse fixa (Mahalanobis), a distância respeita a
FORMA da região realmente ocupada pelos dados — a distância entre dois
pontos pode ser MAIOR que a linha reta entre eles, se essa linha reta
atravessar uma região sem suporte populacional (ex.: dois grupos de
pacientes em forma de crescente, onde a linha reta entre pontas opostas
atravessaria o "vazio" entre os crescentes).

IMPLEMENTAÇÃO: aproximação por GEODÉSICA EM GRAFO (Isomap — Tenenbaum,
de Silva & Langford, 2000), NÃO uma variedade Riemanniana com tensor
métrico contínuo resolvido por EDO (isso exigiria integrar geodésicas,
computacionalmente muito mais caro e sem ganho prático claro aqui):

  1. `fit(space)`: constrói um grafo k-NN sobre a população (arestas
     pesadas pela geometria de base — Euclidean por padrão — apenas
     entre vizinhos próximos, assumindo que LOCALMENTE o espaço é bem
     aproximado por um plano — a premissa padrão de manifold learning).
  2. `distance(x, y)`: adiciona `x` e `y` como nós temporários,
     conectados aos seus `k` vizinhos mais próximos na população
     ajustada, e computa o caminho mais curto (Dijkstra) até `y` —
     aproximando a distância geodésica ao longo da variedade de dados.

Isso é mais caro que Euclidean/Mahalanobis (precisa da população inteira
ajustada previamente via `fit()`, não funciona ponto-a-ponto sem
contexto) — por isso `distance(x, y)` levanta erro claro se `fit()`
não foi chamado antes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

from .base import Geometry
from .euclidean import Euclidean

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["RiemannianGeometry"]


class RiemannianGeometry(Geometry):
    name = "riemannian"

    def __init__(self, base_geometry: Optional[Geometry] = None, k_neighbors: int = 8):
        """
        `base_geometry`: geometria usada para pesar arestas LOCAIS do
        grafo (Euclidean por padrão — localmente, a premissa de manifold
        learning é que o espaço é aproximadamente plano).
        `k_neighbors`: nº de vizinhos mais próximos conectados por nó —
        poucos vizinhos (grafo esparso) força os caminhos a seguir a
        forma real dos dados; muitos vizinhos aproxima de volta a
        distância direta (`base_geometry`), perdendo o efeito "não-plano".
        """
        self.base_geometry = base_geometry or Euclidean()
        self.k_neighbors = k_neighbors
        self._population: Optional[np.ndarray] = None
        self._ids: list[str] = []
        self.is_fitted = False

    def fit(self, space: "RepresentationSpace") -> "RiemannianGeometry":
        """Constrói o grafo k-NN sobre todos os pontos de `space`. Necessário antes de `distance()`."""
        matrix, ids = space.matrix()
        self._population = matrix
        self._ids = ids
        self.is_fitted = True
        return self

    def _knn_indices(self, point: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
        dists = np.array([self.base_geometry.distance(point, p) for p in pool])
        return np.argsort(dists)[:k]

    def _build_graph(self, extra_points: list[np.ndarray]) -> tuple[lil_matrix, int, int]:
        """Constrói a matriz esparsa de adjacência (população + pontos extras temporários)."""
        n_pop = len(self._population)
        n_extra = len(extra_points)
        n_total = n_pop + n_extra
        graph = lil_matrix((n_total, n_total))

        # Arestas dentro da população (k-NN mútuo, computado uma vez por par único)
        for i in range(n_pop):
            neighbors = self._knn_indices(self._population[i], self._population, self.k_neighbors + 1)
            for j in neighbors:
                if j == i:
                    continue
                d = self.base_geometry.distance(self._population[i], self._population[j])
                graph[i, j] = d
                graph[j, i] = d

        # Conecta cada ponto extra (x, y sendo comparados) aos seus k vizinhos na população
        for e, point in enumerate(extra_points):
            idx_extra = n_pop + e
            neighbors = self._knn_indices(point, self._population, self.k_neighbors)
            for j in neighbors:
                d = self.base_geometry.distance(point, self._population[j])
                graph[idx_extra, j] = d
                graph[j, idx_extra] = d

        return graph, n_pop, n_extra

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise RuntimeError(
                "RiemannianGeometry.fit(space) deve ser chamado antes de distance() — "
                "a geodésica é aproximada sobre um grafo construído a partir de uma população."
            )
        graph, n_pop, _ = self._build_graph([x, y])
        idx_x, idx_y = n_pop, n_pop + 1
        dist_matrix = dijkstra(graph.tocsr(), directed=False, indices=idx_x)
        d = float(dist_matrix[idx_y])
        if not np.isfinite(d):
            raise ValueError(
                "Grafo desconectado: x e y caíram em componentes isolados. "
                "Aumente `k_neighbors` ou verifique se a população cobre bem o espaço."
            )
        return d
