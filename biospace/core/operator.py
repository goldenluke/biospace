"""
biospace.core.operator
=========================

Operator: marcador para qualquer algoritmo que atue sobre primitivas do
biospace (RepresentationSpace, Cohort, Trajectory, RepresentationVector)
— nunca diretamente sobre Observation ou dados brutos (Seção 12.2 da
teoria: "o algoritmo nunca recebe P, apenas R(P)").

Deliberadamente NÃO prescreve a assinatura de um método `fit` único.
Famílias diferentes de operadores têm formatos de entrada/saída
genuinamente distintos:

  - Phenotyper       : fit(RepresentationSpace)      -> list[Phenotype]
  - Predictor        : fit(RepresentationSpace, y)    -> Predictor; predict(space) -> dict
  - RiskOperator     : score(RepresentationSpace)      -> dict[str, float]
  - InterventionOperator: apply(RepresentationVector)   -> RepresentationVector   (τ: X -> X, Seção 12.3)
  - SurvivalOperator / TransitionOperator / EarlyWarningOperator:
                        fit(Cohort ou Trajectory)        -> resultado longitudinal

Forçar todos em `fit(space) -> TOutput` seria estruturalmente falso para
metade dessas famílias (as longitudinais operam sobre trajetórias com
tempo, não sobre a nuvem de pontos estática de um RepresentationSpace; a
intervenção transforma um ponto, não "ajusta" um modelo). O único
contrato universal que sobra é `describe()` — cada subfamília declara seu
próprio método abstrato com a assinatura correta.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .cohort import Cohort

__all__ = ["Operator", "LongitudinalOperator"]

TOutput = TypeVar("TOutput")


class Operator(ABC, Generic[TOutput]):
    """Base para operadores TRANSVERSAIS — recebem um RepresentationSpace (um instantâneo da população)."""

    @abstractmethod
    def describe(self) -> str:
        """Descrição curta e legível do que este operador faz."""
        raise NotImplementedError


class LongitudinalOperator(ABC, Generic[TOutput]):
    """
    Base para operadores LONGITUDINAIS — recebem uma Cohort inteira
    (trajetórias com timestamps), não um instantâneo estático. Necessária
    para famílias como SurvivalOperator, TransitionOperator e
    EarlyWarningOperator, que dependem de ordem temporal.
    """

    @abstractmethod
    def describe(self) -> str:
        """Descrição curta e legível do que este operador faz."""
        raise NotImplementedError

