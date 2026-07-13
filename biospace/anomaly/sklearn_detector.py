"""
biospace.anomaly.sklearn_detector
=====================================

SklearnOutlierDetector — envelopa QUALQUER estimador de detecção de
anomalia compatível com a API sklearn de duas formas (`IsolationForest`,
`LocalOutlierFactor(novelty=True)`, `OneClassSVM`, ...). Trocar de
algoritmo é trocar o `estimator` passado ao construtor — nada mais muda,
mesmo espírito de `SklearnPredictor`.

Ressalva real, verificada contra a API do sklearn antes de escrever este
wrapper, não assumida: `LocalOutlierFactor` com `novelty=False` (o
padrão da classe) só suporta `fit_predict()` sobre o PRÓPRIO dado de
ajuste — `.predict()`/`.score_samples()` sobre um RepresentationSpace
diferente do usado em `fit()` levantam erro no sklearn. Detectado e
recusado aqui com uma mensagem clara, no construtor, em vez de deixar
o erro estourar de forma confusa só quando `is_outlier()` for chamado
depois, possivelmente em outra parte do código.
"""

from __future__ import annotations

from typing import Any

from biospace.core import RepresentationSpace

from .base import OutlierDetector

__all__ = ["SklearnOutlierDetector"]


class SklearnOutlierDetector(OutlierDetector):
    def __init__(self, estimator: Any):
        """`estimator`: qualquer objeto com `.fit(X)`, `.score_samples(X)` e `.predict(X)` (API sklearn de detecção de anomalia)."""
        novelty = getattr(estimator, "novelty", None)
        if novelty is False:
            raise ValueError(
                f"{estimator.__class__.__name__} tem novelty=False (o padrão da classe) — só suporta "
                "fit_predict() sobre o próprio dado de ajuste, não predict()/score_samples() sobre um "
                "RepresentationSpace diferente depois. Passe novelty=True na construção do estimador "
                "se pretende usar fit() e is_outlier() com spaces diferentes; se o uso pretendido é "
                "sempre sobre o mesmo space, ainda assim passe novelty=True -- este wrapper sempre "
                "separa fit() de is_outlier(), nunca usa fit_predict() internamente."
            )
        self.estimator = estimator
        self._fitted_ids: set[str] | None = None

    def fit(self, space: RepresentationSpace) -> dict[str, float]:
        matrix, ids = space.matrix()
        self.estimator.fit(matrix)
        self._fitted_ids = set(ids)
        scores = self.estimator.score_samples(matrix)
        return {sid: float(s) for sid, s in zip(ids, scores)}

    def is_outlier(self, space: RepresentationSpace) -> dict[str, bool]:
        if self._fitted_ids is None:
            raise RuntimeError("Chame fit() antes de is_outlier().")
        matrix, ids = space.matrix()
        predictions = self.estimator.predict(matrix)
        return {sid: (p == -1) for sid, p in zip(ids, predictions)}

    def describe(self) -> str:
        return f"SklearnOutlierDetector({self.estimator.__class__.__name__}): detecção de anomalia sobre RepresentationSpace."
