"""
biospace.early_warning.base
==============================

EarlyWarningOperator: interface para operadores de alerta precoce —
variância/autocorrelação em janela deslizante sobre Trajectory, critical
slowing down (Scheffer et al.), e afins. Depende de ORDEM TEMPORAL, não
de um instantâneo — por isso herda de LongitudinalOperator, não de
Operator.

Esta é APENAS a interface (o "slot" arquitetural na hierarquia). A
implementação concreta (janela deslizante -> variância -> autocorrelação
-> critical slowing down -> sinal de alerta) é um módulo à parte, ainda
não construído — nenhuma biblioteca médica genérica faz isso hoje, e vale
mais rigor do que velocidade nessa parte.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from biospace.core import LongitudinalOperator

if TYPE_CHECKING:
    from biospace.core import Cohort

__all__ = ["EarlyWarningOperator"]


class EarlyWarningOperator(LongitudinalOperator[dict]):
    """Interface base para operadores de alerta precoce sobre trajetórias longitudinais."""

    @abstractmethod
    def fit(self, cohort: "Cohort") -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.__class__.__name__}: alerta precoce sobre trajetórias (não implementado ainda)."
