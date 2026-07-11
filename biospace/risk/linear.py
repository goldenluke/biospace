"""
biospace.risk.linear
=======================

LinearRiskOperator — score = soma ponderada de Features nomeadas,
extraídas de QUALQUER domínio do RepresentationVector (não precisam
pertencer todas ao mesmo domínio). Generalização transparente e auditável
da fórmula ad-hoc `score_risco = 0.40*ido + 0.25*(100-spo2_minima) + ...`
usada na pipeline legada (07_clusterizacao.py) — mas expressa como pesos
explícitos sobre Features já existentes na Representation, não como um
recálculo paralelo de valores brutos.
"""

from __future__ import annotations

from biospace.core import RepresentationSpace

from .base import RiskOperator

__all__ = ["LinearRiskOperator"]


class LinearRiskOperator(RiskOperator):
    def __init__(self, weights: dict[str, float]):
        """`weights`: dict Feature.name -> peso. Features não listadas não contribuem ao score."""
        self.weights = weights

    def score(self, space: RepresentationSpace) -> dict[str, float]:
        scores: dict[str, float] = {}
        for sid in space.ids():
            vector = space.get(sid)
            total = 0.0
            for features in vector.components.values():
                for feature in features:
                    if feature.name in self.weights:
                        total += self.weights[feature.name] * feature.value
            scores[sid] = total
        return scores

    def describe(self) -> str:
        pesos = ", ".join(f"{k}={v:+.2f}" for k, v in self.weights.items())
        return f"LinearRiskOperator({pesos})"
