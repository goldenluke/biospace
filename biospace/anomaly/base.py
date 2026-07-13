"""
biospace.anomaly.base
=========================

OutlierDetector: X -> {system_id: score de anomalia}. Não-supervisionado,
como PhenotypingOperator — nenhum rótulo é necessário para ajustar.
Diferente de Predictor (supervisionado, exige `labels`).

Convenção de sinal do score: quanto MAIOR o valor devolvido por
`fit()`, mais NORMAL o sistema (seguindo a convenção do sklearn para
`score_samples` em IsolationForest/LocalOutlierFactor — não invertida
aqui, para não introduzir uma segunda convenção que o chamador precise
lembrar). `is_outlier()` devolve o booleano direto, não exige que o
chamador escolha um limiar sobre o score.
"""

from __future__ import annotations

from abc import abstractmethod

from biospace.core import Operator, RepresentationSpace

__all__ = ["OutlierDetector"]


class OutlierDetector(Operator[dict[str, float]]):
    """Interface base para qualquer algoritmo de detecção de anomalia sobre RepresentationSpace."""

    @abstractmethod
    def fit(self, space: RepresentationSpace) -> dict[str, float]:
        """Ajusta o detector e devolve score de anomalia por system_id (maior = mais normal)."""
        raise NotImplementedError

    @abstractmethod
    def is_outlier(self, space: RepresentationSpace) -> dict[str, bool]:
        """Aplica o detector já ajustado (pode ser a um RepresentationSpace diferente do usado em fit, se o estimador suportar)."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.__class__.__name__}: detecção de anomalia (não-supervisionada) sobre RepresentationSpace."
