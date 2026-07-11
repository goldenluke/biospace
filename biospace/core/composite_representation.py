"""
biospace.core.composite_representation
==========================================

CompositeRepresentation: agrupa vários SemanticDomains (ou outras
CompositeRepresentations, permitindo aninhamento recursivo) sob um nome
único — ex.: "respiratory" agrupando ApneaDomain, "cardiovascular"
agrupando CardiovascularDomain.

Por que isso não exige NENHUMA mudança no núcleo: do ponto de vista de
uma `Representation` pai, uma CompositeRepresentation se comporta
EXATAMENTE como um domínio comum — tem `.name` e
`.transform(system) -> list[Feature]`. `Representation.transform()` já
funciona por duck typing (chama `domain.transform(system)` para cada
item em `self.domains`, sem checar o tipo), então uma
CompositeRepresentation pode ocupar o lugar de um SemanticDomain em
qualquer `Representation` existente sem quebrar `RepresentationSpace`,
`Cohort`, `Ontology` ou qualquer outra coisa que já dependa da API atual.

Isso viabiliza a hierarquia

    patient.representation
    ├── RespiratoryRepresentation
    ├── HypoxiaRepresentation
    ├── CardiovascularRepresentation
    └── MetabolicRepresentation

como uma composição aditiva, não uma reestruturação retroativa.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .feature import Feature, features_to_array

if TYPE_CHECKING:
    from .biological_system import BiologicalSystem
    from .domain import SemanticDomain

__all__ = ["CompositeRepresentation"]

_Child = Union["SemanticDomain", "CompositeRepresentation"]


class CompositeRepresentation:
    """
    `children`: SemanticDomains e/ou outras CompositeRepresentations
    (aninhamento recursivo — um grupo pode conter subgrupos).

    As Features produzidas são as de TODOS os filhos, concatenadas, com
    o nome qualificado por prefixo (`"apnea.ido"`, não só `"ido"`) para
    evitar colisão quando dois filhos tiverem Features de mesmo nome —
    e para manter rastreável de qual filho cada Feature veio, mesmo após
    o achatamento.
    """

    def __init__(self, name: str, children: Sequence[_Child], description: str = ""):
        if not name or not isinstance(name, str):
            raise ValueError(
                "CompositeRepresentation requer `name` (str não vazia) — mesma exigência já aplicada a "
                "SemanticDomain, pelo mesmo motivo: evita um domínio sem nome falhar de forma confusa "
                "mais tarde, ao ser indexado por nome dentro de uma Representation."
            )
        if not children:
            raise ValueError(f"CompositeRepresentation({name!r}) precisa de pelo menos 1 filho (children vazio não agrupa nada).")
        self.name = name
        self.description = description
        self.children: list[_Child] = list(children)

    def transform(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> list[Feature]:
        result: list[Feature] = []
        for child in self.children:
            for f in child.transform(system, as_of=as_of):
                result.append(
                    Feature(
                        name=f"{child.name}.{f.name}",
                        value=f.value,
                        raw_value=f.raw_value,
                        z_score=f.z_score,
                        weight=f.weight,
                        is_missing=f.is_missing,
                        is_excluded=f.is_excluded,
                        provenance=f.provenance,
                        uncertainty=f.uncertainty,
                    )
                )
        return result

    def sub_vector(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> np.ndarray:
        """Conveniência: só o vetor numérico deste grupo (para geometria/distância por sistema fisiológico)."""
        return features_to_array(self.transform(system, as_of=as_of))

    def child_vectors(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> dict[str, np.ndarray]:
        """Vetor numérico de CADA filho direto, separadamente (não achatado) — útil para resumo por subgrupo."""
        return {child.name: features_to_array(child.transform(system, as_of=as_of)) for child in self.children}

    def leaf_domains(self) -> list["SemanticDomain"]:
        """Todos os SemanticDomains folha, recursivamente (ignora o agrupamento, achata tudo)."""
        leaves: list["SemanticDomain"] = []
        for child in self.children:
            if isinstance(child, CompositeRepresentation):
                leaves.extend(child.leaf_domains())
            else:
                leaves.append(child)
        return leaves

    def __repr__(self) -> str:
        return f"CompositeRepresentation({self.name!r}, children={[c.name for c in self.children]})"
