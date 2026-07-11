"""
biospace.geometry.dtw
========================

DTW (Dynamic Time Warping): implementação de `TrajectoryGeometry` —
compara duas Trajectories inteiras, não dois pontos. Alinha as sequências
encontrando a correspondência (possivelmente não 1-para-1) entre pontos
de cada trajetória que minimiza a distância local acumulada — útil
justamente porque pacientes reais têm números de exames diferentes e
datas diferentes (Seção 9 da teoria); comparar trajetórias exige uma
noção de distância que tolera isso, diferente de comparar dois pontos
isolados de X.

PENALIZAÇÃO POR TEMPO (parâmetro `time_penalty_weight`): o DTW clássico
ignora completamente QUANDO cada ponto ocorreu — alinharia sem custo
extra um exame do mês 1 de um paciente a um exame do mês 24 de outro,
desde que as REPRESENTAÇÕES sejam parecidas. Para trajetórias clínicas,
isso pode ser fisiologicamente estranho (dois estados parecidos, mas
muito distantes no tempo de acompanhamento, talvez não devessem ser
"gratuitamente" equiparados). `time_penalty_weight > 0` soma ao custo
local uma penalidade proporcional à diferença de tempo (em dias) entre
os pontos sendo alinhados — 0.0 (padrão) recupera o DTW clássico.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from biospace.core import TrajectoryGeometry

from .base import Geometry
from .euclidean import Euclidean

if TYPE_CHECKING:
    from biospace.core import Trajectory

__all__ = ["DTW"]


def _dtw_dp(cost: np.ndarray, radius: Optional[int] = None) -> tuple[float, list[tuple[int, int]]]:
    """
    Programação dinâmica clássica de DTW sobre uma matriz de custo local
    `cost[i, j]` já calculada. `radius`, se informado, limita a banda de
    Sakoe-Chiba (|i - j| <= radius) — não necessário para trajetórias
    curtas (poucos exames por paciente), mas disponível para o caso geral.
    """
    n, m = cost.shape
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = 1 if radius is None else max(1, i - radius)
        j_hi = m if radius is None else min(m, i + radius)
        for j in range(j_lo, j_hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # backtrack do caminho de alinhamento ótimo
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [(D[i - 1, j], (i - 1, j)), (D[i, j - 1], (i, j - 1)), (D[i - 1, j - 1], (i - 1, j - 1))]
        _, (i, j) = min(candidates, key=lambda c: c[0])
    path.reverse()

    return float(D[n, m]), path


class DTW(TrajectoryGeometry):
    """Ver docstring do módulo para a penalização opcional por tempo e a banda de Sakoe-Chiba."""

    name = "dtw"

    def __init__(
        self,
        point_geometry: Optional[Geometry] = None,
        order: Optional[Sequence[str]] = None,
        time_penalty_weight: float = 0.0,
        radius: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        `point_geometry`: Geometry usada para o custo local entre dois
        RepresentationVector (padrão: Euclidean).
        `time_penalty_weight`: peso da penalidade por diferença de tempo
        (em dias) entre os pontos alinhados — 0.0 desativa (DTW clássico).
        `normalize`: divide a distância total pelo comprimento do caminho
        de alinhamento, evitando que trajetórias mais longas pareçam
        sistematicamente "mais distantes" só por terem mais pontos.
        """
        self.point_geometry = point_geometry or Euclidean()
        self.order = order
        self.time_penalty_weight = time_penalty_weight
        self.radius = radius
        self.normalize = normalize

    def _series(self, trajectory: "Trajectory") -> tuple[list[np.ndarray], list[float]]:
        n = len(trajectory)
        t0 = trajectory.at(0).timestamp
        points = [trajectory.at(i).as_vector(self.order) for i in range(n)]
        times = [(trajectory.at(i).timestamp - t0).total_seconds() / 86400 for i in range(n)]
        return points, times

    def _cost_matrix(self, trajectory_a: "Trajectory", trajectory_b: "Trajectory") -> np.ndarray:
        points_a, times_a = self._series(trajectory_a)
        points_b, times_b = self._series(trajectory_b)
        n, m = len(points_a), len(points_b)
        cost = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                c = self.point_geometry.distance(points_a[i], points_b[j])
                if self.time_penalty_weight > 0:
                    c += self.time_penalty_weight * abs(times_a[i] - times_b[j])
                cost[i, j] = c
        return cost

    def align(self, trajectory_a: "Trajectory", trajectory_b: "Trajectory") -> tuple[float, list[tuple[int, int]]]:
        """Retorna (distância DTW, caminho de alinhamento) — útil para visualizar a correspondência encontrada."""
        cost = self._cost_matrix(trajectory_a, trajectory_b)
        distance, path = _dtw_dp(cost, radius=self.radius)
        if self.normalize and path:
            distance = distance / len(path)
        return distance, path

    def distance(self, trajectory_a: "Trajectory", trajectory_b: "Trajectory") -> float:
        distance, _ = self.align(trajectory_a, trajectory_b)
        return distance

    def distance_matrix(
        self, cohort, min_points: int = 2
    ) -> tuple[np.ndarray, list[str]]:
        """
        Matriz de distâncias DTW par-a-par entre todas as trajetórias de
        uma Cohort com pelo menos `min_points` observações — conveniência
        para clusterização ou análise exploratória de forma de trajetória.
        Custo O(n_pacientes² × comprimento_médio²); adequado para coortes
        de até algumas centenas de pacientes elegíveis.
        """
        ids = [sid for sid, traj in cohort.trajectories.items() if len(traj) >= min_points]
        n = len(ids)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(cohort.trajectories[ids[i]], cohort.trajectories[ids[j]])
                matrix[i, j] = matrix[j, i] = d
        return matrix, ids
