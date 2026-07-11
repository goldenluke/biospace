"""
biospace.prediction.base
===========================

Predictor: A : X × Y_treino -> (X -> Y). Diferente de PhenotypingOperator
(não-supervisionado), um Predictor precisa de rótulos para ser ajustado —
por isso `fit()` recebe `labels` além do `RepresentationSpace`. Depois de
ajustado, `predict()` opera sobre qualquer RepresentationSpace (incluindo
um diferente do usado no fit, ex.: novos pacientes).

Nenhuma implementação concreta aqui conhece qual algoritmo de ML foi
usado — isso é responsabilidade de subclasses como `SklearnPredictor`
(trocar de RandomForest para XGBoost não deveria exigir mudar nada além
da instanciação do Predictor).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from biospace.core import Operator, RepresentationSpace

__all__ = ["Predictor"]


class Predictor(Operator[dict[str, Any]]):
    """Interface base para operadores supervisionados (classificação/regressão) sobre RepresentationSpace."""

    @abstractmethod
    def fit(self, space: RepresentationSpace, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Ajusta o preditor usando `labels` (dict system_id -> rótulo) sobre
        os sistemas presentes em `space`. Retorna as predições in-sample
        (dict system_id -> predição), por conveniência.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, space: RepresentationSpace) -> dict[str, Any]:
        """Aplica o preditor já ajustado a um RepresentationSpace (pode ser diferente do usado em fit)."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.__class__.__name__}: preditor supervisionado sobre RepresentationSpace."
