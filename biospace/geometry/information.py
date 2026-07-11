"""
biospace.geometry.information
================================

Geometria da informação: distância de Fisher-Rao sobre o simplex de
probabilidade, aplicada ao espaço de representação.

    d_FR(p, q) = 2 · arccos( Σ_i sqrt(p_i · q_i) )

Esta é a distância geodésica EXATA (não uma aproximação) na variedade
estatística de distribuições multinomiais sob a métrica de Fisher — o
resultado clássico da geometria da informação de Rao (1945). Satisfaz as
propriedades de métrica e está sempre em [0, π].

Mesma ressalva de `Wasserstein`: exige tratar o vetor como uma
distribuição (valores deslocados para não-negativos e normalizados para
somar 1). É a escolha mais apropriada quando os eixos do domínio já são
proporções/contagens relativas (ex.: distribuição de tempo em diferentes
faixas de ronco); para domínios z-scored genéricos, interprete como
"divergência relativa de ênfase entre eixos", não uma distância
fisiológica direta.
"""

from __future__ import annotations

import numpy as np

from .base import Geometry
from .wasserstein import _to_distribution

__all__ = ["InformationGeometry"]


class InformationGeometry(Geometry):
    name = "information"

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        p = _to_distribution(x)
        q = _to_distribution(y)
        bhattacharyya_coefficient = np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0)
        return float(2.0 * np.arccos(bhattacharyya_coefficient))
