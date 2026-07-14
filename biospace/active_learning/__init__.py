"""
biospace.active_learning
============================

Seleção do próximo sistema biológico a rotular, dado um conjunto já
rotulado e um conjunto ainda não rotulado — não uma nova forma de
predizer, uma forma de decidir ONDE gastar o orçamento de rotulagem
(tipicamente o recurso mais caro em medicina computacional, muito
mais que poder computacional).

Duas estratégias clássicas, ambas operando sobre `RepresentationSpace`:

`UncertaintySampler`: consulta o(s) ponto(s) onde o modelo já
treinado está mais incerto (entropia da distribuição de classe, para
classificadores). `QueryByCommittee`: treina um comitê de modelos
estruturalmente diferentes sobre o mesmo conjunto rotulado, consulta
onde o comitê mais DISCORDA entre si (entropia de voto) — evidência
mais forte de incerteza genuína do que um único modelo relatar baixa
confiança, pela mesma lógica de triangulação já usada no Artigo V
desta série (concordância entre métodos independentes é mais
informativa que qualquer um isolado; aqui, é a DISCORDÂNCIA entre
eles que sinaliza onde rotular).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["UncertaintySampler", "QueryByCommittee"]


@dataclass
class UncertaintySampler:
    """`estimator`: qualquer objeto com `.predict_proba(X)` (classificador sklearn) já ajustado."""
    estimator: object

    def query(self, space: "RepresentationSpace", n_query: int = 1) -> list[str]:
        matrix, ids = space.matrix()
        probs = self.estimator.predict_proba(matrix)
        probs_seguras = np.clip(probs, 1e-12, 1.0)
        entropia = -np.sum(probs_seguras * np.log(probs_seguras), axis=1)
        ordem = np.argsort(-entropia)[:n_query]
        return [ids[i] for i in ordem]

    def uncertainty_scores(self, space: "RepresentationSpace") -> dict[str, float]:
        matrix, ids = space.matrix()
        probs = self.estimator.predict_proba(matrix)
        probs_seguras = np.clip(probs, 1e-12, 1.0)
        entropia = -np.sum(probs_seguras * np.log(probs_seguras), axis=1)
        return {sid: float(e) for sid, e in zip(ids, entropia)}


@dataclass
class QueryByCommittee:
    """`estimators`: lista de estimadores JA' AJUSTADOS, estruturalmente diferentes entre si (ex.: RandomForest + LogisticRegression + SVM) -- quanto mais diversos os vieses, mais informativa a discordancia."""
    estimators: list

    def query(self, space: "RepresentationSpace", n_query: int = 1) -> list[str]:
        matrix, ids = space.matrix()
        votos = np.array([est.predict(matrix) for est in self.estimators])
        n_estimadores = votos.shape[0]

        discordancias = []
        for j in range(votos.shape[1]):
            _, contagens = np.unique(votos[:, j], return_counts=True)
            probs_voto = contagens / n_estimadores
            probs_seguras = np.clip(probs_voto, 1e-12, 1.0)
            entropia_voto = -np.sum(probs_seguras * np.log(probs_seguras))
            discordancias.append(entropia_voto)

        discordancias = np.array(discordancias)
        ordem = np.argsort(-discordancias)[:n_query]
        return [ids[i] for i in ordem]

    def disagreement_scores(self, space: "RepresentationSpace") -> dict[str, float]:
        matrix, ids = space.matrix()
        votos = np.array([est.predict(matrix) for est in self.estimators])
        n_estimadores = votos.shape[0]

        resultado = {}
        for j, sid in enumerate(ids):
            _, contagens = np.unique(votos[:, j], return_counts=True)
            probs_voto = contagens / n_estimadores
            probs_seguras = np.clip(probs_voto, 1e-12, 1.0)
            resultado[sid] = float(-np.sum(probs_seguras * np.log(probs_seguras)))
        return resultado
