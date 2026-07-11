"""
biospace.risk.base
=====================

RiskOperator: A : X -> dict[system_id, float]. Produz um score de risco
contínuo por sistema — diferente de Predictor, tipicamente NÃO exige
rótulos históricos (pode ser uma combinação transparente e auditável de
Features, não um modelo treinado).
"""

from __future__ import annotations

from abc import abstractmethod

from biospace.core import Operator, RepresentationSpace

__all__ = ["RiskOperator"]


class RiskOperator(Operator[dict[str, float]]):
    """Interface base para operadores que produzem um score de risco contínuo por sistema."""

    @abstractmethod
    def score(self, space: RepresentationSpace) -> dict[str, float]:
        """Retorna um score de risco (maior = mais grave, por convenção) por system_id."""
        raise NotImplementedError

    def fit(self, space: RepresentationSpace) -> dict[str, float]:
        """Para RiskOperators sem parâmetros a ajustar, `fit` é apenas um alias de `score`."""
        return self.score(space)

    def describe(self) -> str:
        return f"{self.__class__.__name__}: score de risco contínuo por sistema."
