"""
biospace.topology.persistence
=================================

Homologia persistente (Betti numbers, diagramas de persistência) sobre
um `RepresentationSpace` — envelope fino sobre `ripser` (algoritmo de
Vietoris-Rips, não reimplementado aqui).

H_0 conta componentes conexas; H_1 conta "buracos" (ciclos que não
delimitam uma região preenchida); dimensões maiores generalizam para
cavidades de dimensão mais alta, raramente interpretáveis com dado
clínico de dimensão moderada — este módulo, por padrão, só computa até
H_1 (`max_dimension=1`).

"Número de Betti" costuma ser apresentado na literatura de TDA como um
número fixo — mas para dado com ruído (todo dado real), qualquer
característica topológica tem alguma persistência não-nula por acaso;
a pergunta real é quais persistências são grandes o bastante para
representar estrutura genuína, não artefato de amostragem finita. Esse
projeto não resolve essa questão em aberto da própria área — em vez
de fingir uma resposta universal, expõe o limiar como parâmetro
explícito (`min_persistence`), documentado, nunca escondido atrás de
um valor padrão não declarado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["PersistenceResult", "compute_persistence"]


@dataclass
class PersistenceResult:
    diagrams: list[np.ndarray]  # diagrams[k] = array (n_features, 2) de (nascimento, morte) em dimensao k
    ids_order: list[str]

    def persistences(self, dimension: int) -> np.ndarray:
        """Persistencia (morte - nascimento) de cada caracteristica topologica na dimensao dada, ordenada decrescente. Recursos infinitos (nunca morrem) sao excluidos -- comuns em H_0 (a componente que nunca se funde com outra ate o fim do filtro) e tratados separadamente por convencao da area, nao um bug."""
        if dimension >= len(self.diagrams):
            return np.array([])
        diag = self.diagrams[dimension]
        vidas = diag[:, 1] - diag[:, 0]
        vidas = vidas[np.isfinite(vidas)]
        return np.sort(vidas)[::-1]

    def betti_number(self, dimension: int, min_persistence: float) -> int:
        """Numero de caracteristicas topologicas na dimensao dada com persistencia >= min_persistence -- o "Betti number" efetivo, condicionado ao limiar escolhido, nunca um numero universal."""
        return int(np.sum(self.persistences(dimension) >= min_persistence))

    def summary(self, min_persistence: float) -> str:
        linhas = [f"PersistenceResult (min_persistence={min_persistence}):"]
        for k in range(len(self.diagrams)):
            linhas.append(f"  H_{k}: Betti={self.betti_number(k, min_persistence)}, top persistencias={self.persistences(k)[:3].tolist()}")
        return "\n".join(linhas)


def compute_persistence(space: "RepresentationSpace", order: Optional[list[str]] = None, max_dimension: int = 1) -> PersistenceResult:
    """Computa homologia persistente (Vietoris-Rips) sobre o vetor achatado do `RepresentationSpace`, via `ripser`."""
    from ripser import ripser

    order = order or space.order()
    matrix, ids = space.matrix()
    resultado = ripser(matrix, maxdim=max_dimension)
    return PersistenceResult(diagrams=resultado["dgms"], ids_order=list(ids))
