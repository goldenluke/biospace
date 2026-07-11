"""
biospace.phenotyping.gaussian
================================

GaussianMixturePhenotyper — estimador de fenótipos via Gaussian Mixture
Model, com escolha automática do número de componentes por BIC (Bayesian
Information Criterion) — o análogo, para GMM, da escolha automática de K
por silhouette em `ClinicalKMeansPhenotyper`.

Diferente do K-Means (fronteiras esféricas/Voronoi), o GMM permite
clusters elípticos com covariâncias distintas — mais adequado quando
diferentes fenótipos têm variabilidade fisiológica diferente em cada
eixo (ex.: um fenótipo "leve" pode ser muito mais homogêneo que um
"grave").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from biospace.core import Phenotype, RepresentationSpace

from .base import PhenotypingOperator

__all__ = ["GaussianMixturePhenotyper", "BicResult"]


@dataclass
class BicResult:
    n_components: int
    bic: float
    log_likelihood: float


class GaussianMixturePhenotyper(PhenotypingOperator):
    name = "gmm"

    def __init__(
        self,
        n_components_range: Sequence[int] = range(2, 11),
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_components_range = list(n_components_range)
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.bic_table: list[BicResult] = []
        self.best_n_components: Optional[int] = None

    def fit(self, space: RepresentationSpace) -> list[Phenotype]:
        from sklearn.mixture import GaussianMixture

        matrix, ids = space.matrix()

        self.bic_table = []
        best_n, best_bic = None, np.inf
        for n in self.n_components_range:
            if n >= len(matrix):
                continue
            model = GaussianMixture(
                n_components=n, covariance_type=self.covariance_type, random_state=self.random_state
            )
            model.fit(matrix)
            bic = float(model.bic(matrix))
            self.bic_table.append(BicResult(n_components=n, bic=bic, log_likelihood=float(model.score(matrix))))
            if bic < best_bic:
                best_n, best_bic = n, bic

        if best_n is None:
            raise ValueError("Nenhum n_components válido para o tamanho desta população.")
        self.best_n_components = best_n

        final_model = GaussianMixture(
            n_components=best_n, covariance_type=self.covariance_type, random_state=self.random_state
        )
        raw_labels = final_model.fit_predict(matrix)
        labels = dict(zip(ids, raw_labels.tolist()))
        return self._labels_to_phenotypes(space, labels)
