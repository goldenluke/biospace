"""
biospace.core.phenotype
==========================

Phenotype (F): F ⊆ X — uma região do espaço de representação. Não conhece
KMeans, não conhece HDBSCAN. É apenas uma entidade (região + função de
pertencimento). Algoritmos (em `biospace.phenotyping`) apenas estimam
regiões como esta — nunca as definem (Seção 8 da teoria).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["Phenotype"]


class Phenotype:
    def __init__(self, name: str, membership_fn: Callable[[np.ndarray], bool], interpretation: str = ""):
        self.name = name
        self.membership_fn = membership_fn
        self.interpretation = interpretation

    def contains(self, x: np.ndarray) -> bool:
        return bool(self.membership_fn(x))

    def __repr__(self) -> str:
        return f"Phenotype({self.name!r})"
