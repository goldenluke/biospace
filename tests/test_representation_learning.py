"""
tests.test_representation_learning
======================================

Fase 5 — Representation Learning: aprende em cima do RepresentationSpace
já computado pelos domínios fisiológicos, nunca dos dados brutos.

Teste decisivo: sobre uma estrutura latente 2D CONHECIDA e NÃO LINEAR,
o autoencoder deve reconstruir melhor que PCA (linear) — do contrário, a
não-linearidade não estaria agregando nada de verdade.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.representation_learning import AutoencoderRepresentationLearner, compare_reconstruction_error


def _nonlinear_latent_space(n: int = 300, seed: int = 0) -> RepresentationSpace:
    """8 Features observadas a partir de um latente 2D (z1, z2), com mistura de combinações lineares e NÃO lineares."""
    rng = np.random.default_rng(seed)
    z1 = rng.uniform(-2, 2, n)
    z2 = rng.uniform(-2, 2, n)

    X = np.stack(
        [
            z1,
            z2,
            z1 + z2,
            np.sin(z1 * 1.5),
            z1 * z2,
            z1**2 - z2**2,
            np.cos(z2 * 1.2) * z1,
            z1 - 0.5 * z2,
        ],
        axis=1,
    ) + rng.normal(0, 0.05, (n, 8))

    space = RepresentationSpace()
    for i in range(n):
        vec = RepresentationVector(
            system_id=f"p{i}",
            timestamp=datetime(2024, 1, 1),
            components={"d": [Feature(name=f"f{j}", value=float(X[i, j])) for j in range(8)]},
        )
        space.add(vec)
    return space


def test_representation_learner_requires_fit_before_transform():
    ae = AutoencoderRepresentationLearner(embedding_dim=2)
    with pytest.raises(RuntimeError):
        ae.transform(np.zeros(8))


def test_autoencoder_rejects_invalid_activation():
    with pytest.raises(ValueError):
        AutoencoderRepresentationLearner(activation="nao_existe")


def test_autoencoder_fits_and_transforms_to_correct_dimension():
    space = _nonlinear_latent_space(n=100)
    ae = AutoencoderRepresentationLearner(embedding_dim=3, hidden_dim=8, max_iter=500, random_state=0)
    ae.fit(space)

    order = space.order()
    x = space.get("p0").as_vector(order)
    embedding = ae.transform(x)
    assert embedding.shape == (3,)


def test_autoencoder_reconstructs_better_than_pca_on_known_nonlinear_structure():
    """
    O TESTE DECISIVO: com estrutura latente genuinamente não linear
    (produtos, senos, quadrados de z1/z2), o autoencoder deve reconstruir
    com MENOS erro que PCA (linear) na mesma dimensão de embedding.
    """
    space = _nonlinear_latent_space(n=300, seed=0)

    resultado = compare_reconstruction_error(space, embedding_dim=2, hidden_dim=16, max_iter=3000, random_state=0)

    assert resultado["autoencoder_melhor"] is True, (
        f"Esperava autoencoder superar PCA em estrutura não linear conhecida "
        f"(pca={resultado['erro_pca_linear']:.4f}, ae={resultado['erro_autoencoder_nao_linear']:.4f})"
    )
    assert resultado["erro_autoencoder_nao_linear"] < resultado["erro_pca_linear"] * 0.7, "Esperava uma margem clara, não uma vitória marginal."


def test_reconstruction_error_is_finite_and_nonnegative():
    space = _nonlinear_latent_space(n=100)
    ae = AutoencoderRepresentationLearner(embedding_dim=2, hidden_dim=8, max_iter=500, random_state=0)
    ae.fit(space)
    erro = ae.reconstruction_error(space)
    assert erro >= 0
    assert np.isfinite(erro)


def test_compare_reconstruction_error_on_linear_data_is_reasonable():
    """
    Contraprova: em dados PURAMENTE lineares (rank ~2 exato + ruído
    minúsculo), PCA acha a solução ANALITICAMENTE ótima (erro quase
    zero) — comparar por RAZÃO contra um denominador quase zero é
    instável (achado real ao rodar este teste: razão de 22x, não porque
    o autoencoder fosse ruim, mas porque PCA fica artificialmente
    perto de zero). A checagem certa é o erro ABSOLUTO do autoencoder
    permanecer pequeno, não a razão.
    """
    rng = np.random.default_rng(1)
    n = 200
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    X = np.stack([z1, z2, z1 + z2, 2 * z1 - z2, z1 - z2, 0.5 * z1 + 0.5 * z2], axis=1) + rng.normal(0, 0.05, (n, 6))

    space = RepresentationSpace()
    for i in range(n):
        vec = RepresentationVector(
            system_id=f"p{i}", timestamp=datetime(2024, 1, 1),
            components={"d": [Feature(name=f"f{j}", value=float(X[i, j])) for j in range(6)]},
        )
        space.add(vec)

    resultado = compare_reconstruction_error(space, embedding_dim=2, hidden_dim=8, max_iter=2000, random_state=0)
    assert resultado["erro_autoencoder_nao_linear"] < 0.1, (
        f"Erro absoluto do autoencoder deveria ser pequeno em dados lineares fáceis, "
        f"achou {resultado['erro_autoencoder_nao_linear']:.4f}"
    )
    assert resultado["recomendacao"] == "pca", "Em dados exatamente lineares, PCA (solução analítica) deveria vencer."
