"""
biospace.core.domain
=======================

SemanticDomain (D): agrupa Observables por significado clínico (não por
estrutura matemática). Cada domínio possui uma função `encode` (φ_i) que
mapeia as Measurements coletadas por seus observables em uma lista de
Features — o ponto do domínio no espaço X_i.

Ver `latent_domain.py` para LatentDomain (D_L) — um domínio sem
Observables próprios, que reconstrói um estado a partir de outros
domínios (Seção 6.6 da teoria).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

from .feature import Feature
from .measurement import Measurement
from .observation import Observable

if TYPE_CHECKING:
    from .biological_system import BiologicalSystem

__all__ = ["SemanticDomain"]


class SemanticDomain(ABC):
    """
    name/description: identidade e semântica clínica do domínio.
    observables: os Observables que este domínio consulta.
    """

    name: str
    description: str = ""

    def __init__(self, observables: Sequence[Observable]):
        name = getattr(self, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError(
                f"{self.__class__.__name__} deve definir `name` (str não vazia) como atributo de classe "
                "antes de ser instanciado — sem isso, o domínio falharia de forma confusa muito mais tarde "
                "(ex.: em Representation.transform(), ao indexar componentes por domain.name)."
            )
        self.observables: list[Observable] = list(observables)

    def collect(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> dict[str, Measurement]:
        """Executa cada observable deste domínio contra o sistema, coletando Measurements (respeitando `as_of` — Contrato 5.7)."""
        result: dict[str, Measurement] = {}
        for obs in self.observables:
            measurement = obs.extract(system, as_of=as_of)
            if measurement is not None:
                result[obs.key] = measurement
        return result

    def processes(self) -> set[str]:
        """
        Consulta COMPUTADA, nunca exige override: reúne os nomes de
        PhysiologicalProcess declarados pelos Observables deste domínio
        (`Observable.process`, ignorando os que não declaram nenhum).
        Funciona automaticamente para QUALQUER domínio já existente,
        mesmo que tenha sido escrito antes desta camada existir —
        devolve conjunto vazio se nenhum observable declara processo.
        """
        return {obs.process for obs in self.observables if obs.process is not None}

    @abstractmethod
    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        """
        φ_i : D_i -> X_i

        `measurements` mapeia observable.key -> Measurement (já com
        proveniência). Deve ser determinística (Contrato 5.8).
        """
        raise NotImplementedError

    def transform(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> list[Feature]:
        return self.encode(self.collect(system, as_of=as_of))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(observables={[o.key for o in self.observables]})"
