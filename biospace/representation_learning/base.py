"""
biospace.representation_learning.base
========================================

RepresentationLearner: aprende um EMBEDDING em cima do
RepresentationSpace já computado pelos domínios fisiológicos — nunca a
partir de Observation/Measurement brutos diretamente.

    Hoje:   Sistema -> Representação
    Depois: Sistema -> Representação -> Representation Learning

A diferença arquitetural central (e o motivo de existir uma interface
separada, não apenas "mais um Geometry ou Operator"): o INSUMO do
aprendizado aqui é sempre X — o espaço já estruturado semanticamente por
SemanticDomain (cada eixo tem nome, proveniência, unidade clínica) —,
nunca os dados brutos. Isso significa que a Representation Learning
NUNCA pode redescobrir uma estrutura que os domínios já não expuseram
como eixo — ela reorganiza/comprime o que os domínios definiram, não
substitui essa definição. Um autoencoder rodando direto sobre valores
brutos de sensor não teria essa garantia; um rodando sobre X tem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["RepresentationLearner"]


class RepresentationLearner(ABC):
    """
    `fit(space)`: espaço de representação JÁ COMPUTADO (matriz X real,
    com nomes de eixo rastreáveis) — nunca uma Cohort/lista de
    Observation brutas. Essa assinatura é o contrato central desta
    interface, não um detalhe de implementação.
    """

    @abstractmethod
    def fit(self, space: "RepresentationSpace", order: Optional[Sequence[str]] = None) -> "RepresentationLearner":
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Projeta UM ponto de X (já achatado, mesma ordem usada em fit) no espaço aprendido."""
        raise NotImplementedError

    def fit_transform(self, space: "RepresentationSpace", order: Optional[Sequence[str]] = None) -> dict[str, np.ndarray]:
        self.fit(space, order=order)
        used_order = order or space.order()
        return {sid: self.transform(space.get(sid).as_vector(used_order)) for sid in space.ids()}

    def describe(self) -> str:
        return self.__class__.__name__
