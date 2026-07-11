"""
biospace.gnn.gcn
===================

SimpleGCN: uma Graph Convolutional Network de verdade (Kipf & Welling,
2017 — "Semi-Supervised Classification with Graph Convolutional
Networks"), implementada em NUMPY PURO — forward E backward (gradiente)
derivados e escritos à mão, sem PyTorch/TensorFlow/DGL.

POR QUE NUMPY PURO, NÃO TORCH: já documentado em
`biospace.representation_learning` — `pip install torch` baixa ~1GB de
dependências CUDA mesmo para uso só em CPU (e o índice CPU-only do
PyTorch não está na lista de domínios de rede permitidos aqui). A
matemática de uma GCN de 1-2 camadas é inteiramente tratável à mão:
propagação de mensagens = multiplicação por uma matriz de adjacência
normalizada, e o gradiente de cada camada é uma regra da cadeia direta.

TAREFA: classificação de nós SEMI-SUPERVISIONADA e TRANSDUTIVA — alguns
nós têm rótulo (treino), outros não (teste), mas TODOS participam da
propagação de mensagens (a estrutura do grafo dos nós não-rotulados
ainda ajuda a suavizar a predição dos rotulados, e vice-versa). Esse é
exatamente o experimento original de Kipf & Welling.

Arquitetura (2 camadas):

    Z0 = A_hat @ X
    H1 = ReLU(Z0 @ W0 + b0)
    Z1 = A_hat @ H1
    P  = softmax(Z1 @ W1 + b1)

onde `A_hat` é a matriz de adjacência normalizada simetricamente, com
self-loops: A_hat = D^(-1/2) (A + I) D^(-1/2).

VALIDAÇÃO OBRIGATÓRIA ANTES DE CONFIAR NO RESULTADO: o gradiente
analítico (backward manual) é conferido contra diferenças finitas
(`tests/test_gnn.py::test_gradient_matches_numerical_gradient`) — um
erro de sinal ou de transposição em backprop manual é fácil de cometer
e produziria um modelo que "treina" mas para algo errado, sem nenhum
erro em tempo de execução.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["SimpleGCN", "normalize_adjacency"]


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """A_hat = D^(-1/2) (A + I) D^(-1/2) — normalização simétrica com self-loops (Kipf & Welling, 2017)."""
    n = A.shape[0]
    A_tilde = A + np.eye(n)
    degrees = A_tilde.sum(axis=1)
    d_inv_sqrt = np.power(degrees, -0.5, out=np.zeros_like(degrees), where=degrees > 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class _AdamState:
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    t: int = 0


class SimpleGCN:
    """GCN de 2 camadas para classificação de nós semi-supervisionada e transdutiva. Ver docstring do módulo."""

    def __init__(
        self,
        hidden_dim: int = 16,
        n_classes: Optional[int] = None,
        learning_rate: float = 0.05,
        l2_reg: float = 5e-4,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.is_fitted = False
        self._params: dict[str, np.ndarray] = {}
        self.loss_history_: list[float] = []

    def _init_params(self, n_features: int, n_classes: int) -> None:
        rng = np.random.default_rng(self.random_state)
        scale0 = np.sqrt(2.0 / (n_features + self.hidden_dim))
        scale1 = np.sqrt(2.0 / (self.hidden_dim + n_classes))
        self._params = {
            "W0": rng.normal(0, scale0, (n_features, self.hidden_dim)),
            "b0": np.zeros(self.hidden_dim),
            "W1": rng.normal(0, scale1, (self.hidden_dim, n_classes)),
            "b1": np.zeros(n_classes),
        }

    def _forward(self, A_hat: np.ndarray, X: np.ndarray) -> dict[str, np.ndarray]:
        p = self._params
        Z0 = A_hat @ X
        H1_pre = Z0 @ p["W0"] + p["b0"]
        H1 = np.maximum(0, H1_pre)
        Z1 = A_hat @ H1
        H2_pre = Z1 @ p["W1"] + p["b1"]
        P = _softmax(H2_pre)
        return {"Z0": Z0, "H1_pre": H1_pre, "H1": H1, "Z1": Z1, "H2_pre": H2_pre, "P": P}

    def _loss_and_grads(
        self, A_hat: np.ndarray, X: np.ndarray, y_onehot: np.ndarray, labeled_mask: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        cache = self._forward(A_hat, X)
        P = cache["P"]
        n_labeled = int(labeled_mask.sum())

        eps = 1e-12
        loss_per_node = -np.sum(y_onehot * np.log(P + eps), axis=1)
        loss = float(np.sum(loss_per_node * labeled_mask) / n_labeled)
        for key in ("W0", "W1"):
            loss += self.l2_reg * float(np.sum(self._params[key] ** 2))

        mask_col = labeled_mask.reshape(-1, 1).astype(float)
        dH2_pre = (P - y_onehot) * mask_col / n_labeled  # (n, C)

        p = self._params
        dW1 = cache["Z1"].T @ dH2_pre + 2 * self.l2_reg * p["W1"]
        db1 = dH2_pre.sum(axis=0)
        dZ1 = dH2_pre @ p["W1"].T
        dH1 = A_hat.T @ dZ1  # A_hat é simétrica, mas escrito explicitamente por clareza
        dH1_pre = dH1 * (cache["H1_pre"] > 0)
        dW0 = cache["Z0"].T @ dH1_pre + 2 * self.l2_reg * p["W0"]
        db0 = dH1_pre.sum(axis=0)

        return loss, {"W0": dW0, "b0": db0, "W1": dW1, "b1": db1}

    def fit(self, A: np.ndarray, X: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray, epochs: int = 300) -> "SimpleGCN":
        """
        `A`: adjacência (n x n, sem self-loop — adicionado internamente).
        `X`: Features por nó (n x d).
        `y`: rótulo inteiro por nó (n,) — só as posições com `labeled_mask=True` entram na perda.
        `labeled_mask`: bool (n,).
        """
        n_classes = self.n_classes or int(y.max()) + 1
        self.n_classes = n_classes
        self._init_params(X.shape[1], n_classes)

        A_hat = normalize_adjacency(A)
        y_onehot = np.eye(n_classes)[y]

        adam = _AdamState()
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        for key in self._params:
            adam.m[key] = np.zeros_like(self._params[key])
            adam.v[key] = np.zeros_like(self._params[key])

        self.loss_history_ = []
        for _ in range(epochs):
            loss, grads = self._loss_and_grads(A_hat, X, y_onehot, labeled_mask)
            self.loss_history_.append(loss)
            adam.t += 1
            for key, grad in grads.items():
                adam.m[key] = beta1 * adam.m[key] + (1 - beta1) * grad
                adam.v[key] = beta2 * adam.v[key] + (1 - beta2) * (grad**2)
                m_hat = adam.m[key] / (1 - beta1**adam.t)
                v_hat = adam.v[key] / (1 - beta2**adam.t)
                self._params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)

        self.is_fitted = True
        return self

    def predict_proba(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(...) deve ser chamado antes de predict_proba().")
        A_hat = normalize_adjacency(A)
        return self._forward(A_hat, X)["P"]

    def predict(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(A, X), axis=1)

    def describe(self) -> str:
        status = f"ajustado, {len(self.loss_history_)} épocas, loss final={self.loss_history_[-1]:.4f}" if self.is_fitted else "não ajustado"
        return f"SimpleGCN(hidden_dim={self.hidden_dim}, n_classes={self.n_classes}, {status})"
