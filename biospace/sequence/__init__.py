"""
biospace.sequence
=====================

Aprendizado sobre trajetórias reais R(B,t) via GRU (Gated Recurrent
Unit), implementado em NumPy puro — forward e backward escritos à
mão, validados por checagem de gradiente antes de qualquer uso.

`SequenceForecaster`: prevê o próximo vetor de Features dado o
prefixo da trajetória (forecasting, #13 da taxonomia).
`SequenceClassifier`: classifica um desfecho a partir da trajetória
inteira (longitudinal learning, #12).
"""

from __future__ import annotations

from .gru import GRUParams, gru_backward, gru_forward, init_gru_params
from .trainer import AdamOptimizer, SequenceClassifier, SequenceForecaster

__all__ = [
    "GRUParams", "gru_forward", "gru_backward", "init_gru_params",
    "AdamOptimizer", "SequenceForecaster", "SequenceClassifier",
]
