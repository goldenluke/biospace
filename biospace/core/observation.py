"""
biospace.core.observation
============================

Observation  (O): um procedimento observacional aplicado a um
BiologicalSystem — um exame, uma leitura de sensor, um wearable, um
questionário, uma imagem, um painel laboratorial etc.

Observable: um operador que sabe extrair a Measurement mais recente de uma
grandeza clinicamente nomeada (ex.: "AHI", "SpO2min") a partir de um
BiologicalSystem. Subclasses concretas vivem em plugins — o núcleo nunca
define um Observable concreto.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from .measurement import Measurement

if TYPE_CHECKING:
    from .biological_system import BiologicalSystem

__all__ = ["Observation", "Observable"]


@dataclass
class Observation:
    """
    Container simples — o núcleo não sabe o que "polissonografia" significa.
    Ele só sabe que uma `source` produziu `values` em um `timestamp`.
    """

    timestamp: datetime
    source: str
    values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class Observable(ABC):
    """
    Descreve uma grandeza clinicamente nomeada que pode ser extraída de um
    BiologicalSystem (ex.: "AHI", "SpO2min").

    `process`: nome opcional (string) de um PhysiologicalProcess (ver
    `core/process.py`) que esta grandeza mede, direta ou indiretamente
    — ex.: HbA1c mede o processo "glucose_homeostasis". Default `None`:
    Observables existentes que nunca souberam desta camada continuam
    funcionando de forma idêntica; declarar `process` é estritamente
    opcional e aditivo, nunca requerido pelo núcleo.
    """

    key: str
    unit: Optional[str] = None
    description: str = ""
    process: Optional[str] = None

    def extract(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> Optional[Measurement]:
        """
        Retorna a Measurement mais recente deste Observable para o
        sistema — isto é, o valor de `self.key` na Observation mais
        recente que o contém —, com proveniência completa (fonte +
        timestamp), ou None se o sistema nunca foi observado quanto a
        esta grandeza.

        `as_of` (Contrato 5.7 — Temporalidade): repassado a
        `system.latest_measurement()` para ignorar Observations
        posteriores a esse instante — ver docstring de lá.
        """
        return system.latest_measurement(self.key, as_of=as_of)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.key!r})"
