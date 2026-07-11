"""
biospace.core.measurement
============================

Measurement: o resultado de aplicar um Observable a um BiologicalSystem —
um valor com proveniência completa (fonte + timestamp).

Antes desta divisão, o valor extraído era um float/str cru dentro de um
dict; qualquer rastreabilidade (Contrato 5.1 da teoria) dependia de
convenção, não de estrutura. Measurement torna a proveniência um campo de
primeira classe.

`distribution` (opcional): quando a Observation de origem carregou uma
Distribution em vez de um valor pontual (ver `core.distribution`),
Measurement guarda tanto o ponto estimado (`value`, sempre um float — o
`mean` da distribuição, para retrocompatibilidade total com todo código
que já espera `.value` como float) quanto a distribuição completa, para
quem quiser propagar incerteza.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .distribution import Distribution

__all__ = ["Measurement"]


@dataclass(frozen=True)
class Measurement:
    """Um valor observado, com proveniência (fonte + timestamp) explícita."""

    key: str
    value: Any
    source: str
    timestamp: datetime
    unit: Optional[str] = None
    distribution: Optional["Distribution"] = None

    @classmethod
    def from_distribution(
        cls, key: str, distribution: "Distribution", source: str, timestamp: datetime, unit: Optional[str] = None
    ) -> "Measurement":
        """Constrói uma Measurement a partir de uma Distribution — `value` vira `distribution.mean`."""
        return cls(key=key, value=distribution.mean, source=source, timestamp=timestamp, unit=unit, distribution=distribution)

    @property
    def uncertainty(self) -> float:
        """Desvio padrão da medição — 0.0 se não houver distribuição (valor pontual determinístico, o caso comum)."""
        return self.distribution.std if self.distribution is not None else 0.0

    def is_missing(self) -> bool:
        """True se o valor é ausente (None ou NaN)."""
        if self.value is None:
            return True
        if isinstance(self.value, float) and math.isnan(self.value):
            return True
        return False

    def __repr__(self) -> str:
        if self.distribution is not None:
            return f"Measurement({self.key}={self.distribution!r}, source={self.source!r}, t={self.timestamp})"
        return f"Measurement({self.key}={self.value!r}, source={self.source!r}, t={self.timestamp})"
