"""
biospace.longitudinal.updater
================================

TrajectoryUpdater formaliza, como uma classe própria, o fluxo que já
existe implicitamente em `Cohort.update()` + `BiologicalSystem.observe()`:
ingerir um novo exame SEM recriar o paciente (Seção 9.3 da teoria).

Por que isso não está apenas em `Cohort`? Porque `Cohort.update()` é
deliberadamente mínimo (não valida nada — apenas transforma e anexa).
`TrajectoryUpdater` adiciona, como camada opcional por cima, validações
que fazem sentido ter em um pipeline de produção (ex.: recusar uma nova
observação com timestamp anterior ao último exame já registrado) sem
forçar essas regras em todo mundo que usa `Cohort` diretamente.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from biospace.core import BiologicalSystem, Cohort, Observation, Representation, RepresentationVector

__all__ = ["TrajectoryUpdater", "NonMonotonicObservationError"]


class NonMonotonicObservationError(ValueError):
    """Levantado quando uma nova observação tem timestamp anterior ao último exame já registrado."""


class TrajectoryUpdater:
    """
    Wrapper de `Cohort.update()` com validações opcionais.

    Parâmetros
    ----------
    representation:
        A Representation usada para transformar cada nova observação.
    enforce_monotonic_time:
        Se True (padrão), recusa uma observação cujo timestamp seja
        anterior ao último exame já registrado para aquele sistema —
        um erro comum de pipeline (ex.: reprocessar exames fora de ordem).
    on_update:
        Callback opcional `(system, vector) -> None`, chamado após cada
        atualização bem-sucedida — útil para logging/auditoria externa
        sem acoplar isso ao núcleo.
    """

    def __init__(
        self,
        representation: "Representation",
        enforce_monotonic_time: bool = True,
        on_update: Optional[Callable[["BiologicalSystem", "RepresentationVector"], None]] = None,
    ):
        self.representation = representation
        self.enforce_monotonic_time = enforce_monotonic_time
        self.on_update = on_update

    def apply(
        self,
        cohort: "Cohort",
        system: "BiologicalSystem",
        observation: "Observation",
        timestamp: Optional[datetime] = None,
    ) -> "RepresentationVector":
        """Ingesta `observation` em `system` e atualiza `cohort` — nunca cria um novo paciente."""
        ts = timestamp or observation.timestamp

        if self.enforce_monotonic_time and system.observations:
            last_ts = system.observations[-1].timestamp
            if ts < last_ts:
                raise NonMonotonicObservationError(
                    f"Nova observação em {ts} é anterior ao último exame registrado "
                    f"({last_ts}) para o sistema {system.id!r}. Isso costuma indicar "
                    "dados fora de ordem — passe enforce_monotonic_time=False se for "
                    "intencional (ex.: reprocessamento retroativo de histórico)."
                )

        system.observe(observation)
        vector = cohort.update(system, self.representation, timestamp=ts)

        if self.on_update is not None:
            self.on_update(system, vector)

        return vector
