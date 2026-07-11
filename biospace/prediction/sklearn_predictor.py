"""
biospace.prediction.sklearn_predictor
========================================

SklearnPredictor — envelopa QUALQUER estimador compatível com a API
sklearn (`.fit(X, y)` / `.predict(X)`): RandomForest, LogisticRegression,
XGBoost (via wrapper sklearn-compatível), etc. Trocar de algoritmo é
trocar o `estimator` passado ao construtor — nada mais muda.
"""

from __future__ import annotations

from typing import Any

from biospace.core import RepresentationSpace

from .base import Predictor

__all__ = ["SklearnPredictor"]


class SklearnPredictor(Predictor):
    def __init__(self, estimator: Any):
        """`estimator`: qualquer objeto com `.fit(X, y)` e `.predict(X)` (API sklearn)."""
        self.estimator = estimator

    def fit(self, space: RepresentationSpace, labels: dict[str, Any]) -> dict[str, Any]:
        matrix, ids = space.matrix()
        missing = [sid for sid in ids if sid not in labels]
        if missing:
            raise KeyError(f"Faltam rótulos para {len(missing)} sistema(s), ex.: {missing[:5]}")

        y = [labels[sid] for sid in ids]
        self.estimator.fit(matrix, y)

        predictions = self.estimator.predict(matrix)
        return dict(zip(ids, predictions.tolist() if hasattr(predictions, "tolist") else predictions))

    def predict(self, space: RepresentationSpace) -> dict[str, Any]:
        matrix, ids = space.matrix()
        predictions = self.estimator.predict(matrix)
        return dict(zip(ids, predictions.tolist() if hasattr(predictions, "tolist") else predictions))

    def describe(self) -> str:
        return f"SklearnPredictor({self.estimator.__class__.__name__})"
