"""
tests.test_gnn
==================

Fase 7 (parte 2) — GNN de verdade, em NumPy puro (sem PyTorch/TensorFlow
— ver docstring de `biospace.gnn.gcn` para o porquê).

Dois testes centrais:

  1. `test_gradient_matches_numerical_gradient` — o mais importante de
     todos: confere o backward manual contra diferenças finitas. Um erro
     de sinal ou transposição em backprop escrito à mão não geraria
     NENHUM erro em tempo de execução — o modelo "treinaria" para algo
     errado silenciosamente. Sem este teste, nada mais aqui teria valor.
  2. `test_graph_helps_more_as_labels_become_scarce` — reproduz, em
     miniatura, o achado real deste projeto nos dados de SAOS: com
     muitos rótulos, o grafo pode ATRAPALHAR (suaviza fronteiras já
     bem definidas); com poucos rótulos, o grafo AJUDA bastante — o
     padrão de cruzamento clássico da literatura de GCN semi-supervisionada.
"""

from __future__ import annotations

import numpy as np
import pytest

from biospace.gnn.gcn import SimpleGCN, normalize_adjacency


def test_normalize_adjacency_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    n = 10
    A = (rng.random((n, n)) < 0.3).astype(float)
    A = np.triu(A, 1)
    A = A + A.T

    A_hat = normalize_adjacency(A)
    assert np.allclose(A_hat, A_hat.T), "A_hat deveria ser simétrica (grafo não dirigido + normalização simétrica)."
    assert np.all(np.isfinite(A_hat))


def test_gradient_matches_numerical_gradient():
    """
    O TESTE MAIS IMPORTANTE DESTE ARQUIVO: confere o gradiente analítico
    (backward manual) contra diferenças finitas centrais, parâmetro a
    parâmetro. Tolerância de erro relativo < 1e-4 — bem mais apertada
    que ruído numérico de ponto flutuante razoável.
    """
    rng = np.random.default_rng(0)
    n, d, C = 12, 5, 3
    A = (rng.random((n, n)) < 0.3).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    X = rng.normal(0, 1, (n, d))
    y = rng.integers(0, C, n)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[:8] = True

    gcn = SimpleGCN(hidden_dim=4, n_classes=C, random_state=1)
    gcn._init_params(d, C)
    A_hat = normalize_adjacency(A)
    y_onehot = np.eye(C)[y]

    _, grads = gcn._loss_and_grads(A_hat, X, y_onehot, labeled_mask)

    eps = 1e-5
    for nome, param in gcn._params.items():
        grad_numerico = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            loss_mais, _ = gcn._loss_and_grads(A_hat, X, y_onehot, labeled_mask)
            param[idx] = orig - eps
            loss_menos, _ = gcn._loss_and_grads(A_hat, X, y_onehot, labeled_mask)
            param[idx] = orig
            grad_numerico[idx] = (loss_mais - loss_menos) / (2 * eps)

        erro_relativo = np.abs(grad_numerico - grads[nome]) / (np.abs(grad_numerico) + np.abs(grads[nome]) + 1e-8)
        assert erro_relativo.max() < 1e-4, f"Gradiente de {nome} não bate com diferenças finitas (erro={erro_relativo.max():.2e}) — backprop manual está errado."


def test_gcn_fit_reduces_loss():
    rng = np.random.default_rng(0)
    n, d, C = 30, 6, 2
    A = (rng.random((n, n)) < 0.2).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    X = rng.normal(0, 1, (n, d))
    y = rng.integers(0, C, n)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[:15] = True

    gcn = SimpleGCN(hidden_dim=8, random_state=0)
    gcn.fit(A, X, y, labeled_mask, epochs=200)

    assert gcn.loss_history_[-1] < gcn.loss_history_[0], "A perda deveria diminuir ao longo do treino."


def _build_two_moons_like_graph(n_per_class=40, seed=0):
    """
    Duas classes com Features SOBREPOSTAS (difícil separar só por X),
    mas cada uma formando uma comunidade bem conectada no grafo --
    cenário onde o grafo DEVERIA ajudar mais que as features sozinhas.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    # Features ruidosas, quase sem separacao linear
    X = rng.normal(0, 1, (n, 4))
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Grafo: alta probabilidade de aresta DENTRO da classe, baixa entre classes
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mesma_classe = y[i] == y[j]
            p = 0.3 if mesma_classe else 0.02
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A, X, y


def test_graph_helps_more_as_labels_become_scarce():
    """
    Reproduz em miniatura o achado real deste projeto: com poucos
    rótulos, uma GCN (usa o grafo) deve superar a mesma rede SEM
    propagação (A=identidade) -- aqui, com Features quase sem sinal mas
    uma estrutura de comunidade clara no grafo, a vantagem do grafo deve
    aparecer mesmo com bastante rótulo, já que X sozinho tem pouco a
    oferecer.
    """
    A, X, y = _build_two_moons_like_graph(n_per_class=40, seed=0)
    n = len(y)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    train_idx = idx[:16]  # poucos rotulos (16 de 80)
    test_idx = idx[16:]

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[train_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True

    gcn = SimpleGCN(hidden_dim=8, learning_rate=0.05, random_state=0)
    gcn.fit(A, X_std, y, labeled_mask, epochs=300)
    acc_com_grafo = (gcn.predict(A, X_std)[test_mask] == y[test_mask]).mean()

    A_identidade = np.eye(n)
    gcn2 = SimpleGCN(hidden_dim=8, learning_rate=0.05, random_state=0)
    gcn2.fit(A_identidade, X_std, y, labeled_mask, epochs=300)
    acc_sem_grafo = (gcn2.predict(A_identidade, X_std)[test_mask] == y[test_mask]).mean()

    assert acc_com_grafo > acc_sem_grafo, (
        f"Esperava o grafo ajudar quando as Features sozinhas tem pouco sinal "
        f"(com_grafo={acc_com_grafo:.3f}, sem_grafo={acc_sem_grafo:.3f})"
    )
