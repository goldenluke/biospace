"""
biospace.core.biological_system
==================================

BiologicalSystem (B): representa um sistema biológico real. Acumula
Observations ao longo do tempo. Nunca é recriado — apenas atualizado
(Seção 9.3 da teoria). Nunca chamado de "Patient" no núcleo.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from .distribution import Distribution
from .measurement import Measurement
from .observation import Observation

if TYPE_CHECKING:
    from .domain import SemanticDomain
    from .trajectory import Trajectory

__all__ = ["BiologicalSystem"]


def _unwrap(value: Any) -> Any:
    """Se `value` é uma Distribution, retorna seu ponto estimado (`mean`); senão, retorna como está."""
    return value.mean if isinstance(value, Distribution) else value


class BiologicalSystem:
    """
    Plugins de doença especializam esta classe (ex.: SleepSystem), mas o
    núcleo em si não conhece nenhuma especialização.
    """

    def __init__(self, identifier: Optional[str] = None):
        self.id: str = identifier or str(uuid.uuid4())
        self.observations: list[Observation] = []
        self.domains: list["SemanticDomain"] = []
        self.trajectory: Optional["Trajectory"] = None

    def observe(self, observation: Observation) -> None:
        """Ingesta uma nova Observation. Nunca cria um novo sistema — apenas anexa."""
        self.observations.append(observation)
        self.observations.sort(key=lambda o: o.timestamp)

    def _observations_as_of(self, as_of: Optional[datetime]) -> list[Observation]:
        """Observations com timestamp <= as_of, ou todas se as_of for None (comportamento padrão, sem corte)."""
        if as_of is None:
            return self.observations
        return [o for o in self.observations if o.timestamp <= as_of]

    def latest_measurement(self, key: str, as_of: Optional[datetime] = None) -> Optional[Measurement]:
        """
        Retorna a Measurement mais recente para `key`, percorrendo as
        Observations em ordem cronológica reversa — a fonte de verdade
        usada por `Observable.extract()`.

        `as_of` (Contrato 5.7 — Temporalidade): se informado, ignora
        Observations com timestamp POSTERIOR a `as_of`. Sem isso,
        reconstruir um ponto histórico de uma Trajectory a partir de um
        sistema que já recebeu observações fora de ordem cronológica
        (ex.: preenchimento retroativo) poderia "espiar o futuro" —
        misturar valores de uma observação mais recente em um ponto que
        deveria refletir apenas o que era conhecido até `as_of`.

        Se o valor bruto armazenado for uma `Distribution` (observação
        probabilística — ver `core.distribution`), a Measurement
        resultante carrega a distribuição completa (`.distribution`),
        além do ponto estimado usual em `.value`.
        """
        for obs in reversed(self._observations_as_of(as_of)):
            if key in obs.values:
                raw = obs.values[key]
                if isinstance(raw, Distribution):
                    return Measurement.from_distribution(key, raw, source=obs.source, timestamp=obs.timestamp)
                return Measurement(key=key, value=raw, source=obs.source, timestamp=obs.timestamp)
        return None

    def latest_values(self, as_of: Optional[datetime] = None) -> dict[str, Any]:
        """
        Funde os valores observados (respeitando `as_of`, se informado —
        ver `latest_measurement()`); observações mais recentes
        sobrescrevem as antigas. Valores que forem `Distribution` são
        "desembrulhados" para seu ponto estimado (`.mean`) — este método
        sempre devolve escalares planos, para retrocompatibilidade total
        com código que já espera isso (use `latest_measurement()` se
        precisar da distribuição completa).
        """
        merged: dict[str, Any] = {}
        for obs in self._observations_as_of(as_of):
            merged.update(obs.values)
        return {k: _unwrap(v) for k, v in merged.items()}

    def values_at(self, index: int) -> dict[str, Any]:
        """Funde os valores observados até (e incluindo) observations[index]; mesma regra de desembrulho de `latest_values()`."""
        merged: dict[str, Any] = {}
        for obs in self.observations[: index + 1]:
            merged.update(obs.values)
        return {k: _unwrap(v) for k, v in merged.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r}, n_observations={len(self.observations)})"
