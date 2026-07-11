"""
tests.test_simulation
=========================

Fase 9 — Simulação: `twin = patient.clone(); twin.simulate(...)`.
`DigitalTwin.simulate()` (determinística) já existia; o que faltava era
simulação em CONJUNTO — múltiplos futuros possíveis, com incerteza real,
não um único ponto.

TESTE DECISIVO: validado contra a variância ESTACIONÁRIA TEÓRICA
CONHECIDA de um processo de Ornstein-Uhlenbeck sintético. Achado real ao
validar: a primeira versão usava `residual_std` como se já fosse ruído
de escala "1 dia" — a variância simulada convergiu para ~7x o valor
teórico. Corrigido invertendo a relação de variância estacionária usando
`mean_dt_days` (ver `FeatureDynamics.sigma_eps_per_day`).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.causal import DigitalTwin
from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.dynamics import MeanRevertingEvolutionOperator


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_FlagObservable("x")])

    def encode(self, measurements):
        return [Feature(name="x", value=float(measurements["x"].value))]


def _build_ou_cohort(mu, k_true, target_var, dt_fixo=15, n_patients=150, n_steps=5, seed=0):
    """Processo de Ornstein-Uhlenbeck sintético, com variância estacionária CONHECIDA (target_var)."""
    phi = np.exp(-k_true * dt_fixo)
    sigma_eps = np.sqrt(target_var * (1 - phi**2))

    representation = Representation([_Dom()])
    cohort = Cohort()
    rng = np.random.default_rng(seed)
    t0 = datetime(2020, 1, 1)
    for i in range(n_patients):
        x = rng.normal(mu, np.sqrt(target_var))
        system = BiologicalSystem(identifier=f"p{i}")
        t = t0
        system.observe(Observation(timestamp=t, source="t", values={"x": x}))
        cohort.update(system, representation, timestamp=t)
        for _ in range(n_steps):
            t = t + timedelta(days=dt_fixo)
            x = mu + phi * (x - mu) + rng.normal(0, sigma_eps)
            system.observe(Observation(timestamp=t, source="t", values={"x": x}))
            cohort.update(system, representation, timestamp=t)

    return cohort, representation


def test_digital_twin_simulate_is_deterministic():
    cohort, representation = _build_ou_cohort(mu=20.0, k_true=0.05, target_var=4.0)
    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)

    sid = next(iter(cohort.trajectories))
    twin1 = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)
    twin2 = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)

    path1 = twin1.simulate(evo, horizon_days=200, step_days=50)
    path2 = twin2.simulate(evo, horizon_days=200, step_days=50)

    for (t1, x1), (t2, x2) in zip(path1, path2):
        assert t1 == t2
        assert np.allclose(x1, x2), "simulate() é determinística -- dois gêmeos clonados do mesmo estado devem simular exatamente igual."


def test_simulate_ensemble_requires_sample_method():
    """Um EvolutionOperator sem `.sample()` deve falhar com uma mensagem clara, não um AttributeError confuso."""

    class _SemSample:
        def predict(self, x, dt):
            return x

    cohort, representation = _build_ou_cohort(mu=20.0, k_true=0.05, target_var=4.0, n_patients=5)
    order = representation.domain_names()
    sid = next(iter(cohort.trajectories))
    twin = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)

    with pytest.raises(TypeError, match="sample"):
        twin.simulate_ensemble(_SemSample(), horizon_days=100, step_days=50)


def test_simulate_ensemble_mean_converges_to_true_equilibrium():
    cohort, representation = _build_ou_cohort(mu=20.0, k_true=0.05, target_var=4.0)
    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)

    sid = next(iter(cohort.trajectories))
    twin = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)
    resultado = twin.simulate_ensemble(evo, horizon_days=2000, step_days=50, n_samples=400, seed=1)

    media_final = resultado["mean"][-1, 0]
    assert media_final == pytest.approx(20.0, abs=1.0), f"Média da distribuição preditiva deveria convergir para μ=20, achou {media_final:.3f}"


def test_simulate_ensemble_variance_converges_to_known_stationary_variance():
    """
    O TESTE DECISIVO deste arquivo: a variância da distribuição preditiva,
    após um horizonte longo (muitas meias-vidas), deve convergir para a
    variância ESTACIONÁRIA TEÓRICA CONHECIDA do processo gerador
    (target_var=4.0) — não um múltiplo dela.
    """
    target_var = 4.0
    cohort, representation = _build_ou_cohort(mu=20.0, k_true=0.05, target_var=target_var)
    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)

    sid = next(iter(cohort.trajectories))
    twin = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)
    resultado = twin.simulate_ensemble(evo, horizon_days=2000, step_days=50, n_samples=500, seed=1)

    variancia_final = resultado["std"][-1, 0] ** 2
    assert variancia_final == pytest.approx(target_var, rel=0.3), (
        f"Variância da distribuição preditiva deveria convergir para a variância estacionária "
        f"verdadeira ({target_var}), achou {variancia_final:.3f} — se isto falhar, a escala de "
        f"ruído em sample()/sigma_eps_per_day regrediu para o bug original (~7x o valor real)."
    )


def test_simulate_ensemble_output_shapes():
    cohort, representation = _build_ou_cohort(mu=20.0, k_true=0.05, target_var=4.0, n_patients=30)
    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)

    sid = next(iter(cohort.trajectories))
    twin = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)
    resultado = twin.simulate_ensemble(evo, horizon_days=150, step_days=50, n_samples=20, seed=0)

    n_times = len(resultado["times"])
    assert resultado["paths"].shape == (20, n_times, 1)
    assert resultado["mean"].shape == (n_times, 1)
    assert resultado["std"].shape == (n_times, 1)
    assert resultado["std"][0, 0] == pytest.approx(0.0), "No instante 0 (estado atual, sem simulação), todas as trajetórias deveriam coincidir (std=0)."
