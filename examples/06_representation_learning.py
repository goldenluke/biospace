"""
examples/06_representation_learning.py
=========================================

Fase 5 - Representation Learning: aprende em cima do RepresentationSpace
ja computado pelos dominios fisiologicos, nunca dos dados brutos.

    Hoje:   Sistema -> Representacao
    Depois: Sistema -> Representacao -> Representation Learning

Compara um autoencoder NAO LINEAR (via MLPRegressor, sem dependencia
pesada de deep learning) contra PCA (linear) em dois cenarios: um com
estrutura latente genuinamente nao linear (autoencoder deveria vencer),
outro mencionando o achado real deste projeto nos dados de SAOS (PCA
venceu -- redes neurais precisam de mais dados que metodos lineares).

Rode com: python3 examples/06_representation_learning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import numpy as np

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.representation_learning import AutoencoderRepresentationLearner, compare_reconstruction_error


def build_nonlinear_space(n: int = 300, seed: int = 0) -> RepresentationSpace:
    """8 Features observadas de um latente 2D (z1,z2), com combinacoes lineares E nao lineares."""
    rng = np.random.default_rng(seed)
    z1 = rng.uniform(-2, 2, n)
    z2 = rng.uniform(-2, 2, n)
    X = np.stack(
        [z1, z2, z1 + z2, np.sin(z1 * 1.5), z1 * z2, z1**2 - z2**2, np.cos(z2 * 1.2) * z1, z1 - 0.5 * z2],
        axis=1,
    ) + rng.normal(0, 0.05, (n, 8))

    space = RepresentationSpace()
    for i in range(n):
        vec = RepresentationVector(
            system_id=f"p{i}", timestamp=datetime(2024, 1, 1),
            components={"d": [Feature(name=f"f{j}", value=float(X[i, j])) for j in range(8)]},
        )
        space.add(vec)
    return space


def main():
    print("--- Cenario 1: estrutura latente NAO LINEAR conhecida (2D -> 8D) ---")
    space = build_nonlinear_space()
    resultado = compare_reconstruction_error(space, embedding_dim=2, hidden_dim=16, max_iter=3000, random_state=0)
    print(f"Erro PCA (linear):             {resultado['erro_pca_linear']:.4f}")
    print(f"Erro Autoencoder (nao linear): {resultado['erro_autoencoder_nao_linear']:.4f}")
    print(f"Recomendacao: {resultado['recomendacao']} (autoencoder venceu: {resultado['autoencoder_melhor']})")
    print()

    print("--- Extrair o embedding aprendido de um paciente especifico ---")
    ae = AutoencoderRepresentationLearner(embedding_dim=2, hidden_dim=16, max_iter=3000, random_state=0)
    ae.fit(space)
    order = space.order()
    x = space.get("p0").as_vector(order)
    embedding = ae.transform(x)
    print(f"Vetor original (8D): {x}")
    print(f"Embedding aprendido (2D): {embedding}")
    print()

    print("--- Achado real deste projeto (ver README): nos dados reais de SAOS, PCA venceu ---")
    print("(355 pacientes, 52 dimensoes -- poucos dados para o autoencoder aprender bem;")
    print(" testado com varios hiperparametros, PCA venceu em toda configuracao)")


if __name__ == "__main__":
    main()
