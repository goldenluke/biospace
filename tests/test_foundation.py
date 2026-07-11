"""
tests.test_foundation
=========================

Fase 10 — Foundation Model (protótipo de arquitetura, não um foundation
model de verdade — ver aviso central em
`biospace.foundation.masked_prediction`). Masked feature prediction
(análogo ao MLM do BERT, sobre Features fisiológicas em vez de palavras).

TESTE DECISIVO: estrutura de correlação SINTÉTICA CONHECIDA — Features
genuinamente correlacionadas devem reconstruir muito melhor que o
baseline ingênuo (prever a média); Features de ruído puro NÃO devem
"fingir" reconstruir melhor que esse mesmo baseline.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.foundation import MaskedFeaturePredictor


def _correlated_space(n=300, seed=0):
    """f2=2*f1+ruído pequeno, f3=-f1+ruído pequeno (correlacionadas); f4,f5 ruído puro, sem relação com nada."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = 2 * f1 + rng.normal(0, 0.2, n)
    f3 = -f1 + rng.normal(0, 0.2, n)
    f4 = rng.normal(0, 1, n)
    f5 = rng.normal(0, 1, n)
    X = np.stack([f1, f2, f3, f4, f5], axis=1)

    space = RepresentationSpace()
    for i in range(n):
        vec = RepresentationVector(
            system_id=f"p{i}", timestamp=datetime(2024, 1, 1),
            components={"d": [Feature(name=f"f{j+1}", value=float(X[i, j])) for j in range(5)]},
        )
        space.add(vec)
    return space, X


def test_masked_predictor_requires_fit_before_use():
    modelo = MaskedFeaturePredictor()
    with pytest.raises(RuntimeError):
        modelo.predict_masked(np.zeros(5), masked_indices=[0])


def test_correlated_features_reconstruct_much_better_than_baseline():
    """
    O TESTE DECISIVO: com correlação CONHECIDA e forte, o erro de
    reconstrução mascarada deve ser uma FRAÇÃO PEQUENA da variância da
    própria Feature (o baseline ingênuo de "prever a média").
    """
    space, X = _correlated_space()
    modelo = MaskedFeaturePredictor(hidden_dim=16, mask_fraction=0.2, n_masks_per_patient=30, max_iter=3000, random_state=0)
    modelo.fit(space)

    resultado = modelo.masked_reconstruction_error(space, mask_fraction=0.2, seed=1)

    for nome, idx in [("d.f1", 0), ("d.f2", 1), ("d.f3", 2)]:
        mse = resultado["mse_por_feature"][nome]
        variancia = np.var(X[:, idx])
        razao = mse / variancia
        assert razao < 0.2, f"{nome} é fortemente correlacionada com outras Features -- esperava razão MSE/variância < 0.2, achou {razao:.3f}"


def test_pure_noise_features_do_not_beat_baseline_by_much():
    """Contraprova: Features de ruído puro (sem relação com nada) não deveriam reconstruir muito melhor que o baseline -- não há estrutura real ali para o modelo aprender."""
    space, X = _correlated_space()
    modelo = MaskedFeaturePredictor(hidden_dim=16, mask_fraction=0.2, n_masks_per_patient=30, max_iter=3000, random_state=0)
    modelo.fit(space)

    resultado = modelo.masked_reconstruction_error(space, mask_fraction=0.2, seed=1)

    for nome, idx in [("d.f4", 3), ("d.f5", 4)]:
        mse = resultado["mse_por_feature"][nome]
        variancia = np.var(X[:, idx])
        razao = mse / variancia
        assert razao > 0.7, f"{nome} é ruído puro -- não deveria reconstruir muito melhor que o baseline (razão esperada > 0.7, achou {razao:.3f})"


def test_masked_predictor_output_is_finite():
    space, _ = _correlated_space(n=50)
    modelo = MaskedFeaturePredictor(hidden_dim=8, n_masks_per_patient=10, max_iter=500, random_state=0)
    modelo.fit(space)

    x = space.get("p0").as_vector(space.order())
    reconstruido = modelo.predict_masked(x, masked_indices=[0, 2])
    assert np.all(np.isfinite(reconstruido))
    assert reconstruido.shape == x.shape


def test_masked_reconstruction_error_returns_entry_per_feature():
    space, _ = _correlated_space(n=80)
    modelo = MaskedFeaturePredictor(hidden_dim=8, n_masks_per_patient=10, max_iter=500, random_state=0)
    modelo.fit(space)

    resultado = modelo.masked_reconstruction_error(space, seed=2)
    assert set(resultado["mse_por_feature"].keys()) <= {"d.f1", "d.f2", "d.f3", "d.f4", "d.f5"}
    assert np.isfinite(resultado["mse_global"])
