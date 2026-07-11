"""
biospace.phenotyping.hdbscan
==============================

HDBSCANPhenotyper — estimador de fenótipos via HDBSCAN (clusterização
hierárquica baseada em densidade). Diferente do K-Means, não exige um K
fixo e pode deixar pontos sem cluster ("ruído").

Pontos rotulados como ruído (-1 pelo HDBSCAN) NÃO viram um fenótipo
"Ruído_0" — são tratados como pacientes sem fenótipo definido
(Phenotype.contains() nunca retorna True para eles), refletindo
honestamente que a densidade populacional não sustentou uma região
naquele ponto do espaço.

AVISO — maldição da dimensionalidade: métodos baseados em densidade
degradam rapidamente em espaços de alta dimensão (comum em
`RepresentationSpace`, que facilmente passa de 40-50 eixos somando todos
os domínios). Testes empíricos mostraram que, com `min_cluster_size`
acima de ~8 em um espaço de 52 dimensões, TODOS os pontos viram ruído
(n_noise_ == n). Se isso acontecer, tente reduzir `min_cluster_size`,
usar menos domínios, ou aplicar redução de dimensionalidade (ex.: PCA)
antes de clusterizar — este operador não faz isso por padrão.
"""

from __future__ import annotations

import numpy as np

from biospace.core import Phenotype, RepresentationSpace

from .base import PhenotypingOperator

__all__ = ["HDBSCANPhenotyper"]


class HDBSCANPhenotyper(PhenotypingOperator):
    name = "hdbscan"

    def __init__(self, min_cluster_size: int = 5, min_samples: int | None = None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_noise_: int = 0

    def fit(self, space: RepresentationSpace) -> list[Phenotype]:
        import warnings

        from sklearn.cluster import HDBSCAN

        matrix, ids = space.matrix()
        model = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        raw_labels = model.fit_predict(matrix)

        self.n_noise_ = int(np.sum(raw_labels == -1))
        if self.n_noise_ == len(ids):
            warnings.warn(
                f"HDBSCAN classificou TODOS os {len(ids)} pontos como ruído "
                f"(dimensão do espaço: {matrix.shape[1]}). Isso costuma indicar "
                "maldição da dimensionalidade — tente reduzir min_cluster_size, "
                "usar menos domínios, ou aplicar redução de dimensionalidade antes.",
                RuntimeWarning,
                stacklevel=2,
            )

        labels = {sid: int(lbl) for sid, lbl in zip(ids, raw_labels) if lbl != -1}
        if not labels:
            return []
        return self._labels_to_phenotypes(space, labels)
