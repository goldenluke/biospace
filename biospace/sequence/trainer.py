"""
biospace.sequence.trainer
=============================

Otimizador Adam (Kingma & Ba, 2015) em NumPy puro, e a API de alto
nível (`SequenceForecaster`, `SequenceClassifier`) que consome
`Trajectory`/`RepresentationSpace` — o mesmo papel que `SklearnPredictor`
cumpre pra modelos tabulares, agora pra sequências.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .gru import GRUParams, gru_backward, gru_forward, init_gru_params

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace


class AdamOptimizer:
    def __init__(self, params: GRUParams, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.__dict__.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.__dict__.items()}
        self.t = 0

    def step(self, params: GRUParams, grads: GRUParams) -> None:
        self.t += 1
        for k in params.__dict__:
            g = getattr(grads, k)
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(params, k, getattr(params, k) - update)


@dataclass
class SequenceForecaster:
    """Preve o PROXIMO vetor de Features (regressao) dada a sequencia ate o instante t -- nunca usa informacao futura, por construcao (a saida do GRU depende so de X[0..t])."""
    n_hidden: int = 8
    lr: float = 0.02
    n_epochs: int = 200
    seed: int = 0
    params: Optional[GRUParams] = None

    def fit(self, sequences: list[np.ndarray]) -> list[float]:
        """`sequences`: lista de arrays (T_i, n_features) -- uma por paciente. Treina prevendo X[t+1] a partir do prefixo X[0..t], para todo t possivel."""
        n_input = sequences[0].shape[1]
        self.params = init_gru_params(n_input, self.n_hidden, n_input, seed=self.seed)
        opt = AdamOptimizer(self.params, lr=self.lr)
        historico_perda = []

        for epoca in range(self.n_epochs):
            perda_epoca = 0.0
            n_exemplos = 0
            for seq in sequences:
                T = seq.shape[0]
                if T < 2:
                    continue
                for t in range(1, T):
                    prefixo = seq[:t]
                    alvo = seq[t]
                    cache = gru_forward(self.params, prefixo)
                    pred = cache["y_final"]
                    erro = pred - alvo
                    perda_epoca += 0.5 * np.sum(erro ** 2)
                    n_exemplos += 1
                    grads = gru_backward(self.params, cache, erro)
                    opt.step(self.params, grads)
            historico_perda.append(perda_epoca / max(n_exemplos, 1))
        return historico_perda

    def predict_next(self, sequence_prefix: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Chame fit() antes de predict_next().")
        cache = gru_forward(self.params, sequence_prefix)
        return cache["y_final"]


@dataclass
class SequenceClassifier:
    """Classifica um DESFECHO (0/1) a partir da sequencia inteira de um paciente -- usa toda a trajetoria disponivel no momento da predicao, nunca informacao de apos o desfecho (a sequencia de entrada deve ja vir cortada corretamente pelo chamador)."""
    n_hidden: int = 8
    lr: float = 0.02
    n_epochs: int = 200
    seed: int = 0
    params: Optional[GRUParams] = None

    def fit(self, sequences: list[np.ndarray], labels: list[int]) -> list[float]:
        n_input = sequences[0].shape[1]
        self.params = init_gru_params(n_input, self.n_hidden, 1, seed=self.seed)
        opt = AdamOptimizer(self.params, lr=self.lr)
        historico_perda = []

        for epoca in range(self.n_epochs):
            perda_epoca = 0.0
            for seq, y in zip(sequences, labels):
                cache = gru_forward(self.params, seq)
                logit = cache["y_final"][0]
                prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
                perda_epoca += -(y * np.log(prob + 1e-9) + (1 - y) * np.log(1 - prob + 1e-9))
                dlogit = np.array([prob - y])  # gradiente de entropia cruzada + sigmoid
                grads = gru_backward(self.params, cache, dlogit)
                opt.step(self.params, grads)
            historico_perda.append(perda_epoca / len(sequences))
        return historico_perda

    def predict_proba(self, sequence: np.ndarray) -> float:
        if self.params is None:
            raise RuntimeError("Chame fit() antes de predict_proba().")
        cache = gru_forward(self.params, sequence)
        logit = cache["y_final"][0]
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))


# Nota: para converter uma Trajectory numa matriz (T, n_features), use
# `trajectory.as_matrix(order)`, já existente em `biospace.core` --
# não há necessidade de um helper próprio aqui.
