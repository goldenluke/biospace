"""biospace.geometry.euclidean — implementação Euclidiana de Geometry."""

from __future__ import annotations

import numpy as np

from .base import Geometry

__all__ = ["Euclidean"]


class Euclidean(Geometry):
    name = "euclidean"

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.norm(x - y))
