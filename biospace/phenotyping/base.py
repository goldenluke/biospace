"""
biospace.phenotyping.base
===========================

Φ_hat : X -> F_hat

Um algoritmo de fenotipagem (KMeans, HDBSCAN, GMM, regra clínica, ...) é
apenas uma *implementação* deste operador de estimação — nunca o
definidor do fenótipo em si (Seção 8.3 / 8.9 da teoria). Todo operador
recebe exclusivamente um RepresentationSpace, nunca dados brutos.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from biospace.core import Operator, Phenotype, RepresentationSpace

__all__ = ["PhenotypingOperator"]


class PhenotypingOperator(Operator[list[Phenotype]]):
    """Interface base para qualquer algoritmo que estima fenótipos (regiões de X)."""

    name: str = "phenotyper"

    @abstractmethod
    def fit(self, space: RepresentationSpace) -> list[Phenotype]:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.__class__.__name__}: estima fenótipos (regiões de X) via '{self.name}'."

    def _labels_to_phenotypes(self, space: RepresentationSpace, labels: dict[str, int]) -> list[Phenotype]:
        """
        Converte rótulos de cluster em objetos Phenotype (regiões
        estimadas por proximidade ao centróide mais próximo).
        """
        order = space.order()
        unique_labels = sorted(set(labels.values()))
        centroids: dict[int, np.ndarray] = {}
        for label in unique_labels:
            members = [sid for sid, lbl in labels.items() if lbl == label]
            vectors = np.stack([space.get(sid).as_vector(order) for sid in members])
            centroids[label] = vectors.mean(axis=0)

        def make_membership(label: int):
            def fn(x: np.ndarray) -> bool:
                distances = {lbl: float(np.linalg.norm(x - c)) for lbl, c in centroids.items()}
                return min(distances, key=distances.get) == label

            return fn

        return [
            Phenotype(name=f"{self.name}_{label}", membership_fn=make_membership(label))
            for label in unique_labels
        ]
