"""
biospace.phenotyping.contracts
=================================

Contrato de Estabilidade Fenotípica (Seção 8.5 da teoria):

    "Considere duas amostras independentes, X1 e X2, obtidas da mesma
    população. Uma representação consistente deve produzir fenótipos
    semelhantes. F(X1) ≈ F(X2)."

Diferente dos contratos em `biospace.core.contracts` (que operam sobre
SemanticDomain/Representation, sem conhecer algoritmos), este contrato
exige um PhenotypingOperator de verdade para produzir F(X1) e F(X2) — por
isso vive na camada de fenotipagem, não no núcleo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from biospace.core import RepresentationSpace

from .base import PhenotypingOperator

__all__ = ["StabilityReport", "check_phenotype_stability"]


@dataclass
class StabilityReport:
    adjusted_rand_index: float
    n_common_points: int
    n_phenotypes_1: int
    n_phenotypes_2: int

    @property
    def is_stable(self) -> bool:
        """
        Limiar convencional da literatura de estabilidade de clustering
        (Ben-Hur, Elisseeff & Guyon, 2002, "A Stability Based Method for
        Discovering Structure in Clustered Data"): ARI >= 0.7 sugere
        estrutura genuína e reprodutível; abaixo disso, os "fenótipos"
        encontrados podem ser um artefato da amostra específica, não um
        padrão real da população.
        """
        return self.adjusted_rand_index >= 0.7


def check_phenotype_stability(
    operator_factory: Callable[[], PhenotypingOperator],
    space: RepresentationSpace,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> StabilityReport:
    """
    Divide `space` em duas subamostras DISJUNTAS e independentes, ajusta
    uma NOVA instância de `operator_factory()` em cada uma separadamente
    (para não vazar estado entre os dois ajustes), e aplica AMBOS os
    conjuntos de fenótipos resultantes a TODOS os pontos do espaço
    original — não apenas às respectivas amostras de treino (Phenotype é
    uma região de X; `contains(x)` funciona para qualquer ponto).

    A concordância entre os dois rotulamentos (Adjusted Rand Index, que
    lida corretamente mesmo se as duas amostras encontrarem números
    diferentes de fenótipos) mede a Estabilidade Fenotípica: se o mesmo
    algoritmo, em amostras diferentes da mesma população, converge para
    "os mesmos" fenótipos ou para algo diferente a cada vez.
    """
    from sklearn.metrics import adjusted_rand_score

    ids = space.ids()
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))
    split = int(len(shuffled) * train_fraction)
    ids_1, ids_2 = shuffled[:split], shuffled[split:]
    order = space.order()

    def _subspace(subset_ids: list[str]) -> RepresentationSpace:
        sub = RepresentationSpace(domain_order=order)
        for sid in subset_ids:
            sub.add(space.get(sid))
        return sub

    phenotypes_1 = operator_factory().fit(_subspace(ids_1))
    phenotypes_2 = operator_factory().fit(_subspace(ids_2))

    labels_1: list[str] = []
    labels_2: list[str] = []
    for sid in ids:
        x = space.get(sid).as_vector(order)
        l1 = next((ph.name for ph in phenotypes_1 if ph.contains(x)), None)
        l2 = next((ph.name for ph in phenotypes_2 if ph.contains(x)), None)
        if l1 is not None and l2 is not None:
            labels_1.append(l1)
            labels_2.append(l2)

    ari = float(adjusted_rand_score(labels_1, labels_2)) if labels_1 else float("nan")

    return StabilityReport(
        adjusted_rand_index=ari,
        n_common_points=len(labels_1),
        n_phenotypes_1=len(phenotypes_1),
        n_phenotypes_2=len(phenotypes_2),
    )
