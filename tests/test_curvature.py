"""
tests.test_curvature
========================

Fase 8 — Geometria: Paciente -> Representação -> Variedade -> Trajetória
-> Curvatura -> Estabilidade.

Testa as DUAS formas independentes de estimar curvatura, e reporta
honestamente a limitação real encontrada na validação: a curvatura via
densidade (transversal) é muito mais ruidosa que a curvatura via
dinâmica ajustada (temporal) — não testamos uma coisa que sabíamos que
ia falhar, testamos e o resultado real informou a documentação.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import spearmanr

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.dynamics import MeanRevertingEvolutionOperator
from biospace.geometry import detect_metastability, estimate_density_curvature


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


def _make_space_from_values(values: np.ndarray, feature_name: str = "d.x"):
    from biospace.core import RepresentationSpace, RepresentationVector

    domain, _, atom = feature_name.partition(".")
    space = RepresentationSpace()
    for i, v in enumerate(values):
        vec = RepresentationVector(system_id=f"p{i}", timestamp=datetime(2024, 1, 1), components={domain: [Feature(name=atom, value=float(v))]})
        space.add(vec)
    return space


def test_metastability_detects_single_well_in_unimodal_population():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 300)
    space = _make_space_from_values(values)

    report = detect_metastability(space, "d.x", min_prominence=0.3)
    assert report.n_wells == 1
    assert not report.is_metastable


def test_metastability_detects_two_wells_with_real_barrier_in_bimodal_population():
    """O TESTE DECISIVO da metaestabilidade: 2 grupos bem separados devem virar 2 poços com barreira substancial."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-5, 1, 150), rng.normal(5, 1, 150)])
    space = _make_space_from_values(values)

    report = detect_metastability(space, "d.x", min_prominence=0.3)
    assert report.n_wells == 2
    assert report.is_metastable
    for well in report.wells:
        assert well.escape_barrier is not None
        assert well.escape_barrier > 1.0, "Com grupos tão bem separados, a barreira de escape deveria ser substancial."


def test_metastability_does_not_overfragment_noisy_unimodal_data():
    """Contraprova: ruído dentro de uma única gaussiana não deveria virar vários poços espúrios."""
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, 200)
    space = _make_space_from_values(values)
    report = detect_metastability(space, "d.x", min_prominence=0.3)
    assert report.n_wells == 1, f"Esperava 1 poço em dados unimodais, achou {report.n_wells} (proeminência mínima pode estar baixa demais)."


def test_curvature_from_evolution_operator_recovers_true_k():
    """
    Validação do lado TEMPORAL: com k verdadeiro conhecido (processo de
    Ornstein-Uhlenbeck simulado com ruído auto-consistente com a
    variância estacionária-alvo), `FeatureDynamics.curvature` deve
    recuperar k quase exatamente.
    """
    ks_verdadeiros = {"fraca": 0.02, "media": 0.05, "forte": 0.10, "muito_forte": 0.18}
    mu, dt_fixo, target_var = 20.0, 20, 4.0

    class _Dom(SemanticDomain):
        name = "d"

        def __init__(self):
            super().__init__([_FlagObservable(k) for k in ks_verdadeiros])

        def encode(self, measurements):
            return [Feature(name=k, value=float(measurements[k].value)) for k in ks_verdadeiros if k in measurements]

    representation = Representation([_Dom()])
    cohort = Cohort()
    rng = np.random.default_rng(0)
    t0 = datetime(2020, 1, 1)

    phis = {k: np.exp(-kv * dt_fixo) for k, kv in ks_verdadeiros.items()}
    sigma_eps = {k: np.sqrt(target_var * (1 - phis[k] ** 2)) for k in ks_verdadeiros}

    for i in range(200):
        estados = {k: rng.normal(mu, np.sqrt(target_var)) for k in ks_verdadeiros}
        system = BiologicalSystem(identifier=f"p{i}")
        t = t0
        system.observe(Observation(timestamp=t, source="t", values=dict(estados)))
        cohort.update(system, representation, timestamp=t)
        for _ in range(4):
            t = t + timedelta(days=dt_fixo)
            for k in ks_verdadeiros:
                estados[k] = mu + phis[k] * (estados[k] - mu) + rng.normal(0, sigma_eps[k])
            system.observe(Observation(timestamp=t, source="t", values=dict(estados)))
            cohort.update(system, representation, timestamp=t)

    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)

    k_verdadeiros_lista = list(ks_verdadeiros.values())
    curvaturas = [evo.dynamics_[f"d.{nome}"].curvature for nome in ks_verdadeiros]

    rho, _ = spearmanr(k_verdadeiros_lista, curvaturas)
    assert rho > 0.99, f"Curvatura via EvolutionOperator deveria ordenar perfeitamente com k verdadeiro (rho={rho:.3f})"

    for k_real, curv in zip(k_verdadeiros_lista, curvaturas):
        assert curv == pytest.approx(k_real, rel=0.25), f"Curvatura {curv:.4f} deveria estar perto do k verdadeiro {k_real:.4f}"


def test_density_curvature_runs_but_is_documented_as_imprecise():
    """
    Achado real da validação (ver docstring de `estimate_density_curvature`
    e README): a curvatura via densidade é MUITO mais ruidosa que a
    curvatura via dinâmica temporal — mesmo com variância estacionária
    IDÊNTICA entre Features por construção, o valor retornado variou
    quase 5x. Este teste não afirma precisão que o método não tem —
    só confirma que roda e devolve um número finito e positivo (o
    ponto de densidade máxima é sempre um mínimo local de U, logo
    U''>=0 ali, exceto ruído de borda).
    """
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 300)
    space = _make_space_from_values(values)
    curvature = estimate_density_curvature(space, "d.x")
    assert np.isfinite(curvature)
    assert curvature > 0, "No modo (mínimo de U), a curvatura deveria ser positiva (concavidade para cima)."
