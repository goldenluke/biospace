"""
biospace.phenotyping.spectral
================================

SpectralPhenotyper — estimador de fenótipos via Spectral Clustering.
Diferente de K-Means/GMM, não assume clusters convexos: agrupa por
conectividade no grafo de afinidade (por padrão, kernel RBF sobre a
Geometry do espaço), útil quando os fenótipos formam variedades não
convexas no espaço de representação (ex.: uma trajetória de progressão
em forma de "meia-lua").

Exige um n_clusters fixo (não há um análogo direto de silhouette/BIC tão
bem estabelecido para spectral clustering); ainda assim, oferece uma
varredura opcional por silhouette para ajudar a escolher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from biospace.core import Phenotype, RepresentationSpace

from .base import PhenotypingOperator

__all__ = ["SpectralPhenotyper", "SilhouetteResult"]


@dataclass
class SilhouetteResult:
    n_clusters: int
    silhouette: float


class SpectralPhenotyper(PhenotypingOperator):
    name = "spectral"

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_clusters_range: Sequence[int] = range(2, 11),
        affinity: str = "rbf",
        random_state: int = 42,
    ):
        """
        Se `n_clusters` for informado, usa-o diretamente. Caso contrário,
        varre `n_clusters_range` escolhendo o de maior silhouette (mesma
        lógica de `ClinicalKMeansPhenotyper`, aplicada aqui como
        conveniência, já que spectral clustering não tem um critério de
        seleção de K tão canônico quanto BIC/silhouette).
        """
        self.n_clusters = n_clusters
        self.n_clusters_range = list(n_clusters_range)
        self.affinity = affinity
        self.random_state = random_state
        self.silhouette_table: list[SilhouetteResult] = []
        self.best_n_clusters: Optional[int] = None

    def fit(self, space: RepresentationSpace) -> list[Phenotype]:
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score

        matrix, ids = space.matrix()

        if self.n_clusters is not None:
            best_n = self.n_clusters
        else:
            self.silhouette_table = []
            best_n, best_score = None, -np.inf
            for n in self.n_clusters_range:
                if n >= len(matrix):
                    continue
                model = SpectralClustering(
                    n_clusters=n, affinity=self.affinity, random_state=self.random_state, assign_labels="kmeans"
                )
                labels = model.fit_predict(matrix)
                if len(set(labels)) < 2:
                    continue
                score = float(silhouette_score(matrix, labels))
                self.silhouette_table.append(SilhouetteResult(n_clusters=n, silhouette=score))
                if score > best_score:
                    best_n, best_score = n, score
            if best_n is None:
                raise ValueError("Nenhum n_clusters válido para o tamanho desta população.")

        self.best_n_clusters = best_n
        final_model = SpectralClustering(
            n_clusters=best_n, affinity=self.affinity, random_state=self.random_state, assign_labels="kmeans"
        )
        raw_labels = final_model.fit_predict(matrix)
        labels = dict(zip(ids, raw_labels.tolist()))
        return self._labels_to_phenotypes(space, labels)
