"""
biospace.geometry.wasserstein
================================

Wasserstein-1 (Earth Mover's Distance) sobre o espaço de representação.

IMPORTANTE — leia antes de usar: a distância de Wasserstein é definida
entre DISTRIBUIÇÕES, não entre pontos arbitrários de R^n. Um
RepresentationVector, no entanto, é um vetor de coordenadas (algumas
z-scores que podem ser negativos), não uma distribuição de probabilidade.

Esta implementação trata cada vetor como uma distribuição empírica sobre
os PRÓPRIOS EIXOS do domínio: desloca os valores para não-negativos e
normaliza para somar 1, depois compara a "massa relativa" que cada
paciente coloca em cada eixo. Isso é literalmente correto quando o
domínio já é uma distribuição por natureza (ex.: `tempo_em_ronco_baixo /
_medio / _alto`, que somam ~100% do tempo de ronco). Para domínios
z-scored genéricos, a interpretação é mais fraca — "o quanto o perfil
deste paciente enfatiza cada eixo, relativamente aos outros" — não uma
distância física entre estados fisiológicos. Prefira `Euclidean` ou
`Mahalanobis` como padrão; use `Wasserstein` quando o domínio em questão
for explicitamente uma distribuição (histograma) sobre bins conhecidos.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from .base import Geometry

__all__ = ["Wasserstein"]


def _to_distribution(v: np.ndarray) -> np.ndarray:
    shifted = v - v.min()
    total = shifted.sum()
    if total <= 0:
        return np.ones_like(v) / len(v)
    return shifted / total


class Wasserstein(Geometry):
    name = "wasserstein"

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        positions = np.arange(len(x), dtype=float)
        p = _to_distribution(x)
        q = _to_distribution(y)
        return float(wasserstein_distance(positions, positions, u_weights=p, v_weights=q))
