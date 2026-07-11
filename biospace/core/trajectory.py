"""
biospace.core.trajectory
===========================

Trajectory (Γ): Γ : T -> X. Evolução longitudinal da representação de um
único sistema biológico. Cada novo exame ATUALIZA a trajetória — não cria
um novo paciente (Seção 9.2 / 9.3 da teoria).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .representation import RepresentationVector

if TYPE_CHECKING:
    from .phenotype import Phenotype

__all__ = ["Trajectory"]


class Trajectory:
    def __init__(self, system_id: str):
        self.system_id = system_id
        self._points: list[RepresentationVector] = []

    def update(self, vector: RepresentationVector) -> None:
        if vector.system_id != self.system_id:
            raise ValueError("RepresentationVector pertence a outro sistema biológico.")
        self._points.append(vector)
        self._points.sort(key=lambda v: v.timestamp)

    def __len__(self) -> int:
        return len(self._points)

    def at(self, index: int) -> RepresentationVector:
        return self._points[index]

    def latest(self) -> RepresentationVector:
        if not self._points:
            raise IndexError(
                f"Trajectory do sistema {self.system_id!r} está vazia — nenhum ponto foi adicionado ainda "
                "(nenhuma chamada bem-sucedida a Cohort.update() para este sistema)."
            )
        return self._points[-1]

    def as_matrix(self, order: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.stack([p.as_vector(order) for p in self._points])

    def raw_series(self, domain_name: str, feature_name: str) -> list[tuple["datetime", float]]:
        """
        Extrai (timestamp, raw_value) de uma Feature específica ao longo
        de toda a trajetória — a base sobre a qual QUALQUER
        DerivedVariable é computada (ver `core/derived_variable.py`).
        Pontos onde a Feature está ausente (`is_missing=True`) ou o
        domínio não está presente são pulados, não convertidos em None
        — quem consome a série decide o que fazer com poucos pontos.
        """
        series: list[tuple["datetime", float]] = []
        for vec in self._points:
            for f in vec.components.get(domain_name, []):
                if f.name == feature_name and not f.is_missing and f.raw_value is not None:
                    series.append((vec.timestamp, f.raw_value))
                    break
        return series

    def phenotype_sequence(
        self, phenotypes: Sequence["Phenotype"], order: Optional[Sequence[str]] = None
    ) -> list[Optional[str]]:
        """F_{t1} -> F_{t2} -> ... -> F_{tn}  (fenótipos longitudinais)."""
        seq = []
        for p in self._points:
            vec = p.as_vector(order)
            match = next((ph.name for ph in phenotypes if ph.contains(vec)), None)
            seq.append(match)
        return seq

    def __repr__(self) -> str:
        return f"Trajectory(system={self.system_id}, n_points={len(self)})"
