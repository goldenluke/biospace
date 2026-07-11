"""
biospace.phenotyping.clinical_kmeans
=======================================

ClinicalKMeansPhenotyper migra 07_clusterizacao.py inteiro para dentro do
meta-modelo:

  1. Varredura automática de K (Seção "AVALIAÇÃO DE CLUSTERS" da pipeline
     legada), escolhendo o K de maior silhouette — sem precisar fixar um
     número de clusters a priori.
  2. Rotulagem clínica automática dos clusters, ordenando os centróides
     por um score de severidade (Seção "ROTULAGEM CLÍNICA AUTOMÁTICA").

Continua sendo apenas uma *implementação* de PhenotypingOperator — os
fenótipos em si (regiões de X) não dependem desta classe, apenas são
estimados por ela (Seção 8.3 / 8.9 da teoria).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from biospace.core import Phenotype, RepresentationSpace

from .base import PhenotypingOperator

__all__ = ["ClinicalKMeansPhenotyper", "ElbowResult"]


@dataclass
class ElbowResult:
    """Um ponto da varredura de K — equivalente a uma linha de avaliacao_clusters.csv."""

    k: int
    inertia: float
    silhouette: float


# Nomenclatura migrada literalmente de 07_clusterizacao.py, usada apenas
# quando o K escolhido automaticamente é 4 (caso mais comum na prática
# clínica de SAOS). Para outros K, cai no fallback genérico "Fenótipo N".
_DEFAULT_NAMES_K4: dict[int, str] = {
    0: "Fenótipo Bradicárdico",
    1: "Fenótipo Leve",
    2: "Fenótipo Hiperadrenérgico",
    3: "Fenótipo Hipoxêmico Grave",
}


class ClinicalKMeansPhenotyper(PhenotypingOperator):
    """
    Φ_hat : X -> F_hat via K-Means com escolha automática de K e rotulagem
    clínica dos clusters resultantes.

    Como os domínios de `biospace.plugins.sleep` já orientam seus eixos de
    modo que "maior valor = mais grave" (ver `HypoxiaDomain.encode`,
    `SleepArchitectureDomain.encode`), o score de severidade de um cluster
    é, por padrão, a soma das coordenadas do centróide — uma generalização
    direta da soma ponderada manual (`ido + (100-spo2_minima) + 0.5*imc +
    0.2*fc_media_bpm`) usada em 07_clusterizacao.py. Um `severity_fn`
    customizado pode ser fornecido para replicar pesos específicos.
    """

    name = "clinical_kmeans"

    def __init__(
        self,
        k_range: Sequence[int] = range(2, 11),
        random_state: int = 42,
        n_init: int = 20,
        severity_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        self.k_range = list(k_range)
        self.random_state = random_state
        self.n_init = n_init
        self.severity_fn = severity_fn or (lambda centroid: float(np.sum(centroid)))
        self.elbow_table: list[ElbowResult] = []
        self.best_k: Optional[int] = None

    def fit(self, space: RepresentationSpace) -> list[Phenotype]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        matrix, ids = space.matrix()

        # --- 1. Varredura automática de K -----------------------------------
        self.elbow_table = []
        best_k, best_score = None, -np.inf
        for k in self.k_range:
            if k < 2 or k >= len(matrix):
                continue
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
            labels = model.fit_predict(matrix)
            score = float(silhouette_score(matrix, labels))
            self.elbow_table.append(ElbowResult(k=k, inertia=float(model.inertia_), silhouette=score))
            if score > best_score:
                best_k, best_score = k, score

        if best_k is None:
            raise ValueError("Nenhum K válido na faixa fornecida para o tamanho desta população.")
        self.best_k = best_k

        # --- 2. Treinamento final -----------------------------------
        final_model = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=self.n_init)
        raw_labels = final_model.fit_predict(matrix)
        labels = dict(zip(ids, raw_labels.tolist()))

        centroids: dict[int, np.ndarray] = {}
        for label in sorted(set(labels.values())):
            mask = raw_labels == label
            centroids[label] = matrix[mask].mean(axis=0)

        # --- 3. Rotulagem clínica automática -----------------------------------
        severity = {label: self.severity_fn(centroid) for label, centroid in centroids.items()}
        ordered = sorted(severity, key=severity.get)

        if len(ordered) == 4:
            clinical_names = {label: _DEFAULT_NAMES_K4[i] for i, label in enumerate(ordered)}
        else:
            clinical_names = {label: f"Fenótipo {i + 1}" for i, label in enumerate(ordered)}

        def make_membership(target_label: int):
            def fn(x: np.ndarray) -> bool:
                distances = {lbl: float(np.linalg.norm(x - c)) for lbl, c in centroids.items()}
                return min(distances, key=distances.get) == target_label
            return fn

        return [
            Phenotype(
                name=clinical_names[label],
                membership_fn=make_membership(label),
                interpretation=f"severidade_relativa={severity[label]:.2f}, k_escolhido={best_k}",
            )
            for label in ordered
        ]
