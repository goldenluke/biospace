"""
examples/08_gnn.py
=====================

Fase 7 (parte 2) -- GNN de verdade: uma Graph Convolutional Network
(Kipf & Welling, 2017) em NUMPY PURO, sobre o grafo de similaridade de
pacientes ja construido (Fase 7, parte 1 -- ver examples/07_knowledge_graph.py).

    Hoje:   patient.graph()          -> G = (V, E)
    Depois: patient.graph() -> GNN   -> classificacao/embedding de nos

Reproduz o achado real deste projeto (ver README): com MUITOS rotulos,
o grafo pode ATRAPALHAR (suaviza fronteiras ja bem definidas); com
POUCOS rotulos, o grafo AJUDA bastante -- o padrao de cruzamento
classico da literatura de GCN semi-supervisionada.

Rode com: python3 examples/08_gnn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from biospace.gnn import SimpleGCN


def build_community_graph(n_per_class=40, seed=0):
    """2 classes com Features quase sem sinal, mas comunidades bem conectadas no grafo."""
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    X = rng.normal(0, 1, (n, 4))
    y = np.array([0] * n_per_class + [1] * n_per_class)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.3 if y[i] == y[j] else 0.02
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A, X, y


def main():
    A, X, y = build_community_graph()
    n = len(y)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    print("--- Padrao de cruzamento: grafo ajuda mais quanto menos rotulo houver ---\n")
    for frac in [0.5, 0.2, 0.1, 0.05]:
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        n_train = max(4, int(frac * n))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        labeled_mask = np.zeros(n, dtype=bool)
        labeled_mask[train_idx] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True

        gcn = SimpleGCN(hidden_dim=8, learning_rate=0.05, random_state=0)
        gcn.fit(A, X_std, y, labeled_mask, epochs=300)
        acc_com_grafo = (gcn.predict(A, X_std)[test_mask] == y[test_mask]).mean()

        A_id = np.eye(n)
        gcn2 = SimpleGCN(hidden_dim=8, learning_rate=0.05, random_state=0)
        gcn2.fit(A_id, X_std, y, labeled_mask, epochs=300)
        acc_sem_grafo = (gcn2.predict(A_id, X_std)[test_mask] == y[test_mask]).mean()

        print(f"treino={n_train:2d} ({100*frac:.0f}%): com_grafo={acc_com_grafo:.3f}  sem_grafo={acc_sem_grafo:.3f}  diferenca={acc_com_grafo-acc_sem_grafo:+.3f}")

    print()
    print("Achado real nos dados de SAOS (355 pacientes, ver README):")
    print("  50% rotulado: com_grafo=0.882  sem_grafo=0.927  diferenca=-0.045 (grafo atrapalha)")
    print("   5% rotulado: com_grafo=0.757  sem_grafo=0.580  diferenca=+0.178 (grafo ajuda MUITO)")


if __name__ == "__main__":
    main()
