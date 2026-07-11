"""biospace.geometry.mahalanobis — implementação de Mahalanobis de Geometry."""

from __future__ import annotations

import numpy as np

from biospace.core import RepresentationSpace

from .base import Geometry

__all__ = ["Mahalanobis"]


class Mahalanobis(Geometry):
    name = "mahalanobis"

    def __init__(self, covariance: np.ndarray):
        self.covariance = covariance
        self.inv_covariance = np.linalg.pinv(covariance)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return float(np.sqrt(diff @ self.inv_covariance @ diff.T))

    @classmethod
    def from_space(cls, space: RepresentationSpace) -> "Mahalanobis":
        """Estima a covariância diretamente da população presente no espaço."""
        matrix, _ = space.matrix()
        covariance = np.cov(matrix, rowvar=False)
        return cls(covariance)
