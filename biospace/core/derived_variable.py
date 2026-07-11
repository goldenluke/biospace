"""
biospace.core.derived_variable
=================================

DerivedVariable: uma grandeza computada a partir de uma Trajectory
inteira — não de uma única Observation, como Feature. "Uma glicemia é
uma observação; variabilidade glicêmica, carga hiperglicêmica
acumulada e slope de HbA1c são variáveis derivadas" — motivação
levantada em revisão externa.

Por que isto NÃO pode viver dentro de SemanticDomain: `encode()` só
enxerga `measurements: dict[str, Measurement]` de UM instante — não
tem acesso à trajetória inteira. DerivedVariable é deliberadamente uma
entidade PARALELA a SemanticDomain, não uma subclasse: opera sobre
`Trajectory`, produz `Feature`s que podem ser ANEXADAS a um
RepresentationVector já construído (ver `augment_with_derived_variables`
abaixo), nunca substituindo a representação pontual.

CAMADA OPCIONAL E ADITIVA, mesmo espírito de PhysiologicalProcess
(`core/process.py`): nenhum plugin existente precisa declarar nenhuma
DerivedVariable — sleep e a representação pontual de metabolic
continuam funcionando de forma idêntica sem esta camada.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .feature import Feature

if TYPE_CHECKING:
    from .representation import RepresentationVector
    from .trajectory import Trajectory

__all__ = ["DerivedVariable", "augment_with_derived_variables"]


class DerivedVariable(ABC):
    """
    name: nome estável da variável derivada (vira o nome da Feature resultante).
    domain_name / feature_name: de qual Feature, em qual domínio, esta
    variável deriva — a série bruta é obtida via `Trajectory.raw_series`.
    process: nome opcional de PhysiologicalProcess (mesma camada de
    `core/process.py`) — uma variável derivada pode declarar o mesmo
    processo da Feature que a origina, ou nenhum.
    min_points: nº mínimo de pontos na trajetória para computar — se a
    trajetória tiver menos, `compute()` deve devolver None, nunca um
    valor inventado a partir de dado insuficiente.
    """

    name: str
    description: str = ""
    process: Optional[str] = None
    domain_name: str
    feature_name: str
    min_points: int = 2

    def series(self, trajectory: "Trajectory") -> list[tuple[float, float]]:
        """(dias_desde_o_primeiro_ponto, valor_bruto) — conveniência comum a quase toda subclasse concreta."""
        raw = trajectory.raw_series(self.domain_name, self.feature_name)
        if not raw:
            return []
        t0 = raw[0][0]
        return [((t - t0).total_seconds() / 86400.0, v) for t, v in raw]

    @abstractmethod
    def compute(self, trajectory: "Trajectory") -> Optional[Feature]:
        """
        Deve devolver None (não uma Feature com valor arbitrário) se a
        trajetória não tiver pontos suficientes (`min_points`) ou a
        Feature de origem nunca aparecer nela — silenciosamente
        inventar um valor a partir de dado insuficiente seria pior que
        não produzir a variável.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, from={self.domain_name}.{self.feature_name})"


def augment_with_derived_variables(
    vector: "RepresentationVector", trajectory: "Trajectory", derived_variables: Sequence[DerivedVariable], component_name: str = "derived"
) -> "RepresentationVector":
    """
    Computa cada DerivedVariable sobre `trajectory` e devolve uma CÓPIA
    de `vector` com um componente adicional `component_name` (default
    "derived") contendo as Features resultantes — nunca modifica
    `vector` in-place, e nunca inclui uma variável cujo `compute()`
    devolveu None (trajetória insuficiente para aquela variável
    específica — comum quando diferentes variáveis têm `min_points`
    diferentes).
    """
    from dataclasses import replace

    novas_features: list[Feature] = []
    for dv in derived_variables:
        resultado = dv.compute(trajectory)
        if resultado is not None:
            novas_features.append(resultado)

    novos_components = dict(vector.components)
    novos_components[component_name] = novas_features
    return replace(vector, components=novos_components)
