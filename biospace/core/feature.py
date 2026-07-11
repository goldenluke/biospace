"""
biospace.core.feature
========================

Feature: uma única coordenada computada de um domínio — o átomo da
representação.

Antes desta divisão, `SemanticDomain.encode()` retornava um `np.ndarray`
opaco: um float sem contexto. Feature guarda, para cada coordenada, não só
o valor final (o que efetivamente entra na geometria), mas também o valor
bruto medido, o z-score, o peso de completude aplicado, se foi imputado ou
excluído, e de qual(is) Measurement(s) ela deriva — tornando cada eixo do
espaço de representação auditável (Contrato 5.1 — Rastreabilidade).

`uncertainty` (opcional): quando a Measurement de origem carregava uma
Distribution (observação probabilística — ver `core.distribution`), o
domínio pode propagar essa incerteza algebricamente até aqui (desvio
padrão do `value` final, já na mesma escala/peso). `None` quando não há
incerteza de medição a propagar — o caso comum, retrocompatível.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["Feature", "features_to_array"]


@dataclass
class Feature:
    """
    name: nome estável da coordenada (ex.: "spo2_minima").
    value: valor final, já ponderado — o que `Geometry` de fato usa.
    raw_value: valor bruto medido, se houver (None para features categóricas puras).
    z_score: valor normalizado antes da ponderação por completude, se aplicável.
    weight: peso aplicado (ex.: completude populacional do campo).
    is_missing: True se o valor bruto estava ausente e foi imputado.
    is_excluded: True se o peso zerou a contribuição desta coordenada.
    provenance: chaves das Measurements usadas para derivar esta feature.
    uncertainty: desvio padrão de `value`, propagado de uma Distribution de origem (None = sem incerteza de medição).
    """

    name: str
    value: float
    raw_value: Optional[float] = None
    z_score: Optional[float] = None
    weight: float = 1.0
    is_missing: bool = False
    is_excluded: bool = False
    provenance: tuple[str, ...] = field(default_factory=tuple)
    uncertainty: Optional[float] = None

    def __repr__(self) -> str:
        flag = " [ausente]" if self.is_missing else (" [excluído]" if self.is_excluded else "")
        unc = f" ± {self.uncertainty:.3f}" if self.uncertainty else ""
        return f"Feature({self.name}={self.value:.3f}{unc}{flag})"


def features_to_array(features: Sequence[Feature]) -> np.ndarray:
    """Projeta uma sequência de Features em um vetor numérico (implementação, não teoria)."""
    return np.array([f.value for f in features], dtype=float)
