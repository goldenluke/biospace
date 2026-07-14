"""
biospace.sequence.gru
=========================

Gated Recurrent Unit (Cho et al., 2014) implementado em NumPy puro —
forward E backward (backpropagation through time) escritos à mão, sem
autodiff. Segue o mesmo espírito de `biospace.gnn.SimpleGCN`: quando
não há framework de deep learning disponível no ambiente, a
alternativa não é fingir com um MLP sobre janela deslizante — é
implementar o método de verdade, e provar que a implementação está
certa antes de confiar em qualquer treino real.

A prova de correção usada aqui é checagem de gradiente por diferenças
finitas (comparar o gradiente analítico do backward manual contra uma
aproximação numérica, célula por célula) — o padrão-ouro pra validar
uma implementação de backprop escrita à mão, feito ANTES de qualquer
uso em dado sintético ou real, não depois.

Equações (Cho et al., 2014):
  z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)      # gate de atualização
  r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)      # gate de reset
  h~_t = tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h) # candidato
  h_t = (1 - z_t) * h_{t-1} + z_t * h~_t           # estado oculto
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class GRUParams:
    """Todos os pesos do GRU + camada de saida linear, num unico objeto pra facilitar treino/gradiente."""
    Wz: np.ndarray; Uz: np.ndarray; bz: np.ndarray
    Wr: np.ndarray; Ur: np.ndarray; br: np.ndarray
    Wh: np.ndarray; Uh: np.ndarray; bh: np.ndarray
    Wout: np.ndarray; bout: np.ndarray

    def flatten(self) -> np.ndarray:
        return np.concatenate([v.ravel() for v in self.__dict__.values()])

    def unflatten_like(self, flat: np.ndarray) -> "GRUParams":
        out, i = {}, 0
        for k, v in self.__dict__.items():
            n = v.size
            out[k] = flat[i:i + n].reshape(v.shape)
            i += n
        return GRUParams(**out)

    def copy(self) -> "GRUParams":
        return GRUParams(**{k: v.copy() for k, v in self.__dict__.items()})


def init_gru_params(n_input: int, n_hidden: int, n_output: int, seed: int = 0) -> GRUParams:
    rng = np.random.default_rng(seed)
    scale_in = 1.0 / np.sqrt(n_input)
    scale_h = 1.0 / np.sqrt(n_hidden)
    return GRUParams(
        Wz=rng.normal(0, scale_in, (n_hidden, n_input)), Uz=rng.normal(0, scale_h, (n_hidden, n_hidden)), bz=np.zeros(n_hidden),
        Wr=rng.normal(0, scale_in, (n_hidden, n_input)), Ur=rng.normal(0, scale_h, (n_hidden, n_hidden)), br=np.zeros(n_hidden),
        Wh=rng.normal(0, scale_in, (n_hidden, n_input)), Uh=rng.normal(0, scale_h, (n_hidden, n_hidden)), bh=np.zeros(n_hidden),
        Wout=rng.normal(0, scale_h, (n_output, n_hidden)), bout=np.zeros(n_output),
    )


def gru_forward(params: GRUParams, X: np.ndarray) -> dict:
    """X: (T, n_input) -- uma unica sequencia. Devolve cache completo (necessario pro backward)."""
    T, n_input = X.shape
    n_hidden = params.Wz.shape[0]
    h = np.zeros(n_hidden)
    cache = {"X": X, "h": [h.copy()], "z": [], "r": [], "htilde": []}
    for t in range(T):
        x_t = X[t]
        z_t = _sigmoid(params.Wz @ x_t + params.Uz @ h + params.bz)
        r_t = _sigmoid(params.Wr @ x_t + params.Ur @ h + params.br)
        htilde_t = np.tanh(params.Wh @ x_t + params.Uh @ (r_t * h) + params.bh)
        h = (1 - z_t) * h + z_t * htilde_t
        cache["z"].append(z_t); cache["r"].append(r_t); cache["htilde"].append(htilde_t); cache["h"].append(h.copy())
    y_final = params.Wout @ h + params.bout
    cache["y_final"] = y_final
    return cache


def gru_backward(params: GRUParams, cache: dict, dy_final: np.ndarray) -> GRUParams:
    """Backpropagation through time -- gradiente de uma perda escalar (via dy_final = dL/dy_final) em relacao a todos os parametros."""
    X = cache["X"]; T = X.shape[0]
    hs = cache["h"]  # hs[0]=h inicial (zero), hs[t+1]=h apos passo t
    grads = {k: np.zeros_like(v) for k, v in params.__dict__.items()}

    h_final = hs[-1]
    grads["Wout"] += np.outer(dy_final, h_final)
    grads["bout"] += dy_final
    dh_next = params.Wout.T @ dy_final  # gradiente vindo da saida, propagando pro ultimo h

    for t in reversed(range(T)):
        x_t = X[t]; h_prev = hs[t]; h_t = hs[t + 1]
        z_t = cache["z"][t]; r_t = cache["r"][t]; htilde_t = cache["htilde"][t]

        dh_t = dh_next
        dz_t = dh_t * (htilde_t - h_prev)
        dhtilde_t = dh_t * z_t
        dh_prev_direct = dh_t * (1 - z_t)

        dhtilde_pre = dhtilde_t * (1 - htilde_t ** 2)  # derivada de tanh
        grads["Wh"] += np.outer(dhtilde_pre, x_t)
        grads["Uh"] += np.outer(dhtilde_pre, r_t * h_prev)
        grads["bh"] += dhtilde_pre
        dr_times_hprev = params.Uh.T @ dhtilde_pre
        dr_t = dr_times_hprev * h_prev
        dh_prev_via_htilde = dr_times_hprev * r_t

        dz_pre = dz_t * z_t * (1 - z_t)  # derivada de sigmoid
        grads["Wz"] += np.outer(dz_pre, x_t)
        grads["Uz"] += np.outer(dz_pre, h_prev)
        grads["bz"] += dz_pre
        dh_prev_via_z = params.Uz.T @ dz_pre

        dr_pre = dr_t * r_t * (1 - r_t)
        grads["Wr"] += np.outer(dr_pre, x_t)
        grads["Ur"] += np.outer(dr_pre, h_prev)
        grads["br"] += dr_pre
        dh_prev_via_r = params.Ur.T @ dr_pre

        dh_next = dh_prev_direct + dh_prev_via_htilde + dh_prev_via_z + dh_prev_via_r

    return GRUParams(**grads)
