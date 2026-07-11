"""biospace.phenotyping.kmeans — estimador de fenótipos via K-Means."""

from __future__ import annotations

from biospace.core import Phenotype, RepresentationSpace

from .base import PhenotypingOperator

__all__ = ["KMeansPhenotyper"]


class KMeansPhenotyper(PhenotypingOperator):
    name = "kmeans"

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, space: RepresentationSpace) -> list[Phenotype]:
        from sklearn.cluster import KMeans  # import local: núcleo não depende de sklearn

        matrix, ids = space.matrix()
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = dict(zip(ids, model.fit_predict(matrix).tolist()))
        return self._labels_to_phenotypes(space, labels)
