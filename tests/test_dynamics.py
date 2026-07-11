"""
tests.test_dynamics
======================

Regressão do bug mais sutil do módulo `dynamics`: a primeira versão do
estimador de φ (taxa de contração diária) usava uma regressão log-linear
restrita a pares "do mesmo lado da média" — e isso produzia uma
estimativa SISTEMATICAMENTE enviesada para perto de 1 (nenhuma reversão),
mesmo em dados sintéticos com φ real conhecido, e o viés NÃO diminuía com
mais dados (descartando amostra pequena como causa).

Corrigido com mínimos quadrados não lineares conjuntos sobre (μ, log φ),
sem filtrar nenhum par. Este teste trava a correção como contrato
permanente: ajustado sobre uma série sintética com φ conhecido, o
estimador deve recuperar φ dentro de uma tolerância razoável.
"""

from __future__ import annotations

import random

from biospace.dynamics.evolution import _fit_mean_reversion


def _generate_mean_reverting_pairs(mu: float, phi: float, n: int, noise_std: float, seed: int) -> list[tuple[float, float, float]]:
    rng = random.Random(seed)
    pairs = []
    x = mu + 0.3  # começa perto da média, oscilando
    for _ in range(n):
        dt = rng.uniform(5, 20)
        noise = rng.gauss(0, noise_std)
        x_next = mu + phi**dt * (x - mu) + noise
        pairs.append((x, x_next, dt))
        x = x_next
    return pairs


def test_mean_reversion_estimator_recovers_known_phi_with_enough_data():
    """Com dados suficientes (3000 pares, ruído moderado), o φ estimado deve ficar próximo do real (0.9)."""
    pairs = _generate_mean_reverting_pairs(mu=20.0, phi=0.9, n=3000, noise_std=0.5, seed=0)
    mu_fit, phi_fit, n_used, resid, _ = _fit_mean_reversion(pairs)

    assert abs(mu_fit - 20.0) < 1.0
    assert abs(phi_fit - 0.9) < 0.05, (
        f"φ estimado ({phi_fit:.4f}) longe do real (0.9) mesmo com dados abundantes — "
        "o viés sistemático que já corrigimos uma vez pode ter voltado."
    )


def test_mean_reversion_estimator_exact_recovery_without_noise():
    """Sem ruído, o estimador deve recuperar φ e μ EXATAMENTE (o processo é determinístico)."""
    pairs = _generate_mean_reverting_pairs(mu=20.0, phi=0.9, n=30, noise_std=0.0, seed=0)
    mu_fit, phi_fit, n_used, resid, _ = _fit_mean_reversion(pairs)

    assert abs(mu_fit - 20.0) < 1e-3
    assert abs(phi_fit - 0.9) < 1e-3


def test_mean_reversion_estimator_bias_does_not_grow_with_more_data():
    """
    O teste mais direto da regressão: o viés do estimador ANTIGO NÃO
    diminuía com mais dados (permanecia perto de φ=1 mesmo com 3000
    pares). O estimador corrigido deve melhorar (ou pelo menos não
    piorar) a estimativa conforme mais dados são adicionados.
    """
    erro_pouco_dado = None
    erro_muito_dado = None
    for n, slot in [(100, "pouco"), (3000, "muito")]:
        pairs = _generate_mean_reverting_pairs(mu=20.0, phi=0.9, n=n, noise_std=0.5, seed=0)
        _, phi_fit, _, _, _ = _fit_mean_reversion(pairs)
        erro = abs(phi_fit - 0.9)
        if slot == "pouco":
            erro_pouco_dado = erro
        else:
            erro_muito_dado = erro

    assert erro_muito_dado < erro_pouco_dado + 0.05, (
        f"Erro com mais dados ({erro_muito_dado:.4f}) não melhorou em relação a menos dados "
        f"({erro_pouco_dado:.4f}) — sintoma do viés sistemático que já corrigimos uma vez."
    )


def test_diverging_series_correctly_flagged_unstable():
    """φ > 1 (processo divergente) deve continuar sendo recuperado corretamente, sem 'suavizar' para perto de 1."""
    pairs = _generate_mean_reverting_pairs(mu=20.0, phi=1.15, n=15, noise_std=0.0, seed=1)
    _, phi_fit, _, _, _ = _fit_mean_reversion(pairs)

    assert phi_fit > 1.0, f"φ={phi_fit:.4f} deveria ser > 1 para um processo divergente conhecido."


def test_feature_dynamics_stability_flag():
    from biospace.dynamics.evolution import FeatureDynamics

    stable = FeatureDynamics(name="x", mu=0.0, phi_per_day=0.95, n_pairs=100, residual_std=0.1)
    unstable = FeatureDynamics(name="y", mu=0.0, phi_per_day=1.05, n_pairs=100, residual_std=0.1)

    assert stable.is_stable is True
    assert unstable.is_stable is False
