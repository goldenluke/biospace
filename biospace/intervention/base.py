"""
biospace.intervention.base
=============================

InterventionOperator: τ : X -> X (Seção 12.3 da teoria). Diferente das
demais famílias, não produz uma predição/score sobre a população — ele
TRANSFORMA um único RepresentationVector, simulando o efeito de uma
intervenção terapêutica sobre o estado fisiológico representado.

`fit()` é opcional (tem um default que não faz nada): implementações
simples (deslocamento fixo) não precisam calibrar nada; implementações
mais sofisticadas podem sobrescrever `fit()` para aprender o efeito a
partir de pares antes/depois observados em uma Cohort real.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from biospace.core import Operator, RepresentationSpace

if TYPE_CHECKING:
    from biospace.core import RepresentationVector

__all__ = ["InterventionOperator"]


class InterventionOperator(Operator["InterventionOperator"]):
    """Interface base para τ: X -> X — transforma um RepresentationVector (simula uma intervenção)."""

    @abstractmethod
    def apply(self, vector: "RepresentationVector") -> "RepresentationVector":
        """τ(x): retorna o RepresentationVector transformado (nunca modifica `vector` in-place)."""
        raise NotImplementedError

    def fit(self, space: RepresentationSpace) -> "InterventionOperator":
        """Default: nada a calibrar. Sobrescreva para aprender o efeito a partir de dados históricos."""
        return self

    def describe(self) -> str:
        return f"{self.__class__.__name__}: τ(x) — transforma um RepresentationVector."
