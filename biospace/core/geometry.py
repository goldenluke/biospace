"""
biospace.core.geometry
========================

Geometry (G): G : X -> M. Atribui uma noção de distância/similaridade ao
espaço de representação (Seção 7 da teoria). Apenas a INTERFACE vive aqui
— implementações concretas (Euclidean, Mahalanobis, ...) vivem no pacote
irmão `biospace.geometry`, que reexporta esta classe para não duplicar a
definição.

TrajectoryGeometry: diferente de Geometry (que compara PONTOS de X — um
instante por sistema), TrajectoryGeometry compara TRAJETÓRIAS INTEIRAS —
sequências de RepresentationVector com possivelmente número diferente de
pontos e amostragem irregular (Seção 9 da teoria: o objeto fundamental
passa a ser a trajetória, não o ponto isolado). DTW é o exemplo canônico:
alinha duas trajetórias de comprimentos distintos encontrando a
correspondência que minimiza a distância cumulativa. Forçar isso na
mesma interface de Geometry seria estruturalmente falso — o mesmo
raciocínio que separou Operator de LongitudinalOperator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .representation_space import RepresentationSpace

if TYPE_CHECKING:
    from .trajectory import Trajectory

__all__ = ["Geometry", "TrajectoryGeometry"]


class Geometry(ABC):
    """Interface para métricas sobre o espaço de representação (compara PONTOS)."""

    name: str = "geometry"

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Deve satisfazer, no mínimo, as propriedades clássicas de métrica
        (Seção 7.3): não-negatividade, identidade, simetria e desigualdade
        triangular.
        """
        raise NotImplementedError

    def similarity(self, x: np.ndarray, y: np.ndarray, scale: float = 1.0) -> float:
        """s(x, y) ∈ [0, 1], derivada da distância via kernel gaussiano."""
        d = self.distance(x, y)
        return float(np.exp(-(d**2) / (2 * scale**2)))

    def neighborhood(self, space: RepresentationSpace, system_id: str, radius: float) -> list[str]:
        """B_ε(x) = {y ∈ X : d(x, y) ≤ ε}  (Seção 7.6)."""
        order = space.order()
        ref = space.get(system_id).as_vector(order)
        return [
            sid
            for sid in space.ids()
            if sid != system_id and self.distance(ref, space.get(sid).as_vector(order)) <= radius
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TrajectoryGeometry(ABC):
    """Interface para métricas entre TRAJETÓRIAS inteiras (não pontos isolados de X)."""

    name: str = "trajectory_geometry"

    @abstractmethod
    def distance(self, trajectory_a: "Trajectory", trajectory_b: "Trajectory") -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
