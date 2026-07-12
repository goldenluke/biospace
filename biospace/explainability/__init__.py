"""
biospace.explainability
==========================

Explica modelos preditivos já ajustados (`biospace.prediction`) — não
um método de representação nem de inferência em si, uma camada que
opera SOBRE um `Predictor` já treinado.
"""

from __future__ import annotations

from .shap_explainer import ShapExplanationReport, explain_predictor

__all__ = ["ShapExplanationReport", "explain_predictor"]
