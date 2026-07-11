"""
biospace.core.representation_space
=====================================

RepresentationSpace (X): X = {R(B_1), R(B_2), ..., R(B_n)}. Armazena e
organiza RepresentationVectors. Nunca executa inferência ou clusterização
aqui — isso é responsabilidade de um Operator.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .representation import RepresentationVector

__all__ = ["RepresentationSpace"]


class RepresentationSpace:
    def __init__(self, domain_order: Optional[Sequence[str]] = None):
        self._points: dict[str, RepresentationVector] = {}
        self.domain_order = domain_order

    def add(self, vector: RepresentationVector) -> None:
        """Adiciona (ou SUBSTITUI, se `vector.system_id` já existir — última chamada vence) um ponto ao espaço."""
        self._points[vector.system_id] = vector

    def get(self, system_id: str) -> RepresentationVector:
        if system_id not in self._points:
            raise KeyError(f"system_id {system_id!r} não está neste RepresentationSpace ({len(self)} pontos: {self.ids()[:5]}{'...' if len(self) > 5 else ''}).")
        return self._points[system_id]

    def ids(self) -> list[str]:
        return list(self._points.keys())

    def order(self) -> list[str]:
        if self.domain_order:
            return list(self.domain_order)
        if not self._points:
            return []
        return sorted(next(iter(self._points.values())).components.keys())

    def matrix(self) -> tuple[np.ndarray, list[str]]:
        """Projeção conveniente de X em uma matriz numérica (implementação, não teoria)."""
        ids = self.ids()
        if not ids:
            raise ValueError(
                "RepresentationSpace está vazio — não há pontos para formar uma matriz. "
                "Verifique se `Cohort.snapshot()` (ou o que quer que tenha construído este espaço) "
                "realmente encontrou sistemas com pelo menos uma observação."
            )
        order = self.order()
        M = np.stack([self._points[i].as_vector(order) for i in ids])
        return M, ids

    def __len__(self) -> int:
        return len(self._points)

    def __repr__(self) -> str:
        return f"RepresentationSpace(n_points={len(self)})"
