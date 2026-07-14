"""
biospace.bayesian
=====================

Aprendizado bayesiano sobre RepresentationSpace: Regressão por
Processo Gaussiano (com quantificação de incerteza, não só predição
pontual) e Redes Bayesianas (estrutura de dependência aprendida +
inferência por eliminação de variáveis, via pgmpy).
"""

from __future__ import annotations

from .gaussian_process import GaussianProcessOperator, GPFitResult
from .network import BayesianNetworkOperator, discretize

__all__ = ["GaussianProcessOperator", "GPFitResult", "BayesianNetworkOperator", "discretize"]
