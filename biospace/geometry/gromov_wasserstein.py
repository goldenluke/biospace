"""
biospace.geometry.gromov_wasserstein
=======================================

GromovWasserstein: compara duas Trajectories pela estrutura RELACIONAL
interna de cada uma (a matriz de distâncias par-a-par entre os próprios
pontos da trajetória) — não por correspondência ponto-a-ponto como DTW.

Por que isso importa: DTW (e as geometrias pontuais — Euclidean,
Wasserstein, ...) exigem que as duas trajetórias vivam no MESMO espaço de
representação X (a mesma Representation, os mesmos domínios).
Gromov-Wasserstein NÃO exige isso: compara apenas como os pontos de uma
trajetória se relacionam entre si, então pode em princípio comparar a
FORMA da evolução de um paciente de SAOS com a de um paciente de outra
doença, mesmo usando Representations completamente diferentes — desde
que cada uma tenha sua própria Geometry para medir distâncias internas.
É a primeira geometria deste projeto que de fato atravessa a fronteira
entre plugins de doença (Seção 10 da teoria — "diferentes doenças
compartilham o meta-modelo, mas usam domínios diferentes").

IMPORTANTE — honestidade de escopo: este projeto tem apenas UM plugin de
doença (sleep) até agora, então a comparação verdadeiramente
"cross-disease" não pôde ser testada empiricamente aqui — apenas a
comparação de forma DENTRO do mesmo espaço de representação (o que já é
uma validação legítima da matemática, mas não da promessa cross-disease
completa).

Implementação via POT (Python Optimal Transport, Flamary et al. 2021),
usando a variante entrópica (Sinkhorn) por padrão — mais estável e
rápida que a solução exata, que é NP-difícil em geral.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from biospace.core import TrajectoryGeometry

from .base import Geometry
from .euclidean import Euclidean

if TYPE_CHECKING:
    from biospace.core import Cohort, Trajectory

__all__ = ["GromovWasserstein"]


class GromovWasserstein(TrajectoryGeometry):
    """
    `internal_geometry` mede as distâncias INTERNAS de cada trajetória
    (não entre as duas trajetórias sendo comparadas — essa é a diferença
    central em relação a DTW). `epsilon` é o peso da regularização
    entrópica (Sinkhorn): valores menores aproximam da solução exata mas
    são mais instáveis numericamente; valores maiores são mais estáveis
    mas menos precisos.
    """

    name = "gromov_wasserstein"

    def __init__(
        self,
        internal_geometry: Optional[Geometry] = None,
        order: Optional[Sequence[str]] = None,
        epsilon: float = 0.05,
        max_iter: int = 500,
    ):
        self.internal_geometry = internal_geometry or Euclidean()
        self.order = order
        self.epsilon = epsilon
        self.max_iter = max_iter

    def _internal_distance_matrix(self, trajectory: "Trajectory") -> np.ndarray:
        n = len(trajectory)
        points = [trajectory.at(i).as_vector(self.order) for i in range(n)]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.internal_geometry.distance(points[i], points[j])
                C[i, j] = C[j, i] = d
        return C

    def distance(self, trajectory_a: "Trajectory", trajectory_b: "Trajectory") -> float:
        import ot

        n, m = len(trajectory_a), len(trajectory_b)
        if n < 2 or m < 2:
            raise ValueError("Gromov-Wasserstein exige trajetórias com pelo menos 2 pontos cada.")

        C1 = self._internal_distance_matrix(trajectory_a)
        C2 = self._internal_distance_matrix(trajectory_b)

        # Normaliza cada matriz de custo pelo seu próprio máximo — sem
        # isso, trajetórias com escalas de distância muito diferentes
        # (ex.: uma muito mais "espalhada" que a outra) dominariam a
        # comparação por magnitude, não por forma relacional.
        max1, max2 = C1.max(), C2.max()
        if max1 > 1e-12:
            C1 = C1 / max1
        if max2 > 1e-12:
            C2 = C2 / max2

        p1 = np.ones(n) / n
        p2 = np.ones(m) / m

        gw_squared = ot.gromov.entropic_gromov_wasserstein2(
            C1, C2, p1, p2, loss_fun="square_loss", epsilon=self.epsilon, max_iter=self.max_iter
        )
        return float(np.sqrt(max(float(gw_squared), 0.0)))

    def distance_matrix(self, cohort: "Cohort", min_points: int = 2) -> tuple[np.ndarray, list[str]]:
        """Matriz de distâncias GW par-a-par entre todas as trajetórias elegíveis de uma Cohort."""
        ids = [sid for sid, traj in cohort.trajectories.items() if len(traj) >= min_points]
        n = len(ids)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(cohort.trajectories[ids[i]], cohort.trajectories[ids[j]])
                matrix[i, j] = matrix[j, i] = d
        return matrix, ids
