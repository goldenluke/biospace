"""
tests.test_critical_slowing_down
====================================

`biospace.early_warning.CriticalSlowingDownDetector` — o detector
completo de "critical slowing down" (Scheffer et al. 2009; Dakos et
al. 2012), com 3 indicadores (variância, autocorrelação, assimetria),
teste de tendência de Kendall sobre janela móvel, e significância por
dados substitutos AR(1). Existia sem NENHUM teste antes desta rodada
— achado numa auditoria do projeto, junto com uma duplicação real:
`biospace.dynamics.early_warning` (removido) era uma versão muito mais
simples do mesmo conceito, construída sem saber que esta já existia.

Validado contra uma simulação de bifurcação sela-nó GENUÍNA
(dx/dt = r(t) + x², ramo estável em x=-√(-r); Strogatz, "Nonlinear
Dynamics and Chaos") — o teste padrão da própria literatura de CSD —
não um AR(1) com φ arbitrariamente manipulado.

NOTA HONESTA sobre a validação: a primeira tentativa desta validação
falhou por um bug na simulação sintética (sinal de x0 no ramo instável
da bifurcação, causando explosão numérica imediata para o clip) — não
um bug no detector. Corrigido e revalidado antes de aceitar qualquer
conclusão sobre o detector em si.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.early_warning import CriticalSlowingDownDetector


class _ObsX(Observable):
    key = "x"


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_ObsX()])

    def encode(self, measurements):
        v = float(measurements["x"].value)
        return [Feature(name="x", value=v, raw_value=v)]


_REPRESENTATION = Representation([_Dom()])


def _build_trajectory(sid, valores):
    system = BiologicalSystem(identifier=sid)
    cohort = Cohort()
    for i, v in enumerate(valores):
        ts = datetime(2020, 1, 1) + timedelta(days=i)
        system.observe(Observation(timestamp=ts, source="t", values={"x": v}))
        cohort.update(system, _REPRESENTATION, timestamp=ts)
    return system, cohort.trajectories[sid]


def _gerar_fold_bifurcation(n_obs, r_inicial, r_final, x0, sigma, seed, passos_por_obs=20, dt=0.05, clip=10.0):
    """
    dx/dt = r(t) + x^2 -- bifurcacao sela-no quando r cruza 0. Ramo
    ESTAVEL (para r<0) e' x=-sqrt(-r): f'(x)=2x<0 ali, f'(+sqrt(-r))>0
    (instavel). r(t) sobe linearmente de r_inicial ate r_final -- se
    r_final se aproxima de 0, o sistema se aproxima da bifurcacao e
    deveria mostrar sinais de critical slowing down (recuperacao mais
    lenta = |f'(x*)|=2|x*| cai conforme x*->0).
    """
    rng = np.random.default_rng(seed)
    x = x0
    serie = [x]
    for i in range(1, n_obs):
        r = r_inicial + (r_final - r_inicial) * i / n_obs
        for _ in range(passos_por_obs):
            dx = (r + x**2) * dt + sigma * rng.normal(0, np.sqrt(dt))
            x = np.clip(x + dx, -clip, clip)
        serie.append(x)
    return serie


def _cohort_bifurcacao(n_seeds=15, n_obs=80):
    """n_seeds trajetorias se aproximando da bifurcacao (criticas) + n_seeds com r constante, longe da bifurcacao (estaveis, controle)."""
    cohort = Cohort()
    for seed in range(n_seeds):
        sid = f"critico_{seed}"
        sistema, traj = _build_trajectory(sid, _gerar_fold_bifurcation(n_obs, r_inicial=-4.0, r_final=-0.05, x0=-2.0, sigma=0.15, seed=seed))
        cohort.systems[sid], cohort.trajectories[sid] = sistema, traj

        sid2 = f"estavel_{seed}"
        sistema2, traj2 = _build_trajectory(sid2, _gerar_fold_bifurcation(n_obs, r_inicial=-4.0, r_final=-4.0, x0=-2.0, sigma=0.15, seed=seed + 100))
        cohort.systems[sid2], cohort.trajectories[sid2] = sistema2, traj2
    return cohort


def test_detector_discriminates_approaching_bifurcation_from_stable_control():
    """
    TESTE DECISIVO: trajetorias genuinamente se aproximando de uma
    bifurcacao sela-no devem mostrar mais casos de warning=True, e
    tau de autocorrelacao medio maior, que trajetorias de controle com
    r constante (longe da bifurcacao). Nao exige que TODA trajetoria
    critica dispare warning (sensibilidade imperfeita e' esperado e
    documentado na literatura, dado o criterio conservador de
    significancia por substitutos) -- exige uma diferenca populacional
    clara entre os dois grupos.
    """
    cohort = _cohort_bifurcacao(n_seeds=15)
    detector = CriticalSlowingDownDetector.for_feature("d", "x", min_points=8, window_size=10, n_surrogates=100)
    resultados = detector.fit(cohort)

    n_criticos_warning = sum(1 for sid, r in resultados.items() if sid.startswith("critico") and r.warning)
    n_estaveis_warning = sum(1 for sid, r in resultados.items() if sid.startswith("estavel") and r.warning)
    assert n_criticos_warning > n_estaveis_warning, (
        f"Esperava mais warnings entre as criticas -- obteve criticas={n_criticos_warning}/15, estaveis={n_estaveis_warning}/15"
    )

    taus_criticos = [resultados[f"critico_{s}"].tau_autocorrelation for s in range(15) if resultados[f"critico_{s}"].tau_autocorrelation is not None]
    taus_estaveis = [resultados[f"estavel_{s}"].tau_autocorrelation for s in range(15) if resultados[f"estavel_{s}"].tau_autocorrelation is not None]
    assert np.mean(taus_criticos) > np.mean(taus_estaveis), "Esperava tau de autocorrelacao medio maior nas trajetorias criticas."


def test_trajectory_shorter_than_min_points_gives_insufficient_data():
    sistema, traj = _build_trajectory("curta", [1.0, 2.0, 1.5])
    cohort = Cohort()
    cohort.systems["curta"], cohort.trajectories["curta"] = sistema, traj

    detector = CriticalSlowingDownDetector.for_feature("d", "x", min_points=8)
    resultados = detector.fit(cohort)
    assert resultados["curta"].sufficient_data is False
    assert resultados["curta"].n_points == 3


def test_for_distance_from_baseline_convenience_constructor_runs_without_error():
    """A conveniencia for_distance_from_baseline usa distancia geometrica ao 1o ponto como indicador -- confirma que roda e produz resultado bem formado, nao testa discriminacao (isso ja e' coberto pelo teste decisivo acima com for_feature)."""
    cohort = _cohort_bifurcacao(n_seeds=3, n_obs=30)
    detector = CriticalSlowingDownDetector.for_distance_from_baseline(order=["d"], min_points=8, window_size=6, n_surrogates=0)
    resultados = detector.fit(cohort)
    assert len(resultados) == 6
    for r in resultados.values():
        assert r.sufficient_data is True


def test_warning_requires_majority_of_available_indicators_not_unanimity():
    """A propriedade `warning` deveria exigir MAIORIA (>=metade) dos indicadores disponiveis, nao unanimidade -- construido diretamente via EWSResult/IndicatorResult para isolar essa logica da deteccao de tendencia em si."""
    from biospace.early_warning.critical_slowing_down import EWSResult, IndicatorResult

    # 3 indicadores disponiveis, 2 significativos e crescentes -- deveria ser warning=True (maioria)
    resultado_maioria = EWSResult(
        system_id="x", n_points=20, sufficient_data=True,
        indicators={
            "variance": IndicatorResult(tau=0.5, p_surrogate=0.01, n_surrogates=100),
            "autocorrelation": IndicatorResult(tau=0.5, p_surrogate=0.01, n_surrogates=100),
            "skewness": IndicatorResult(tau=-0.2, p_surrogate=0.5, n_surrogates=100),
        },
    )
    assert resultado_maioria.warning is True

    # so 1 de 3 significativo -- NAO deveria ser warning (minoria)
    resultado_minoria = EWSResult(
        system_id="y", n_points=20, sufficient_data=True,
        indicators={
            "variance": IndicatorResult(tau=0.5, p_surrogate=0.01, n_surrogates=100),
            "autocorrelation": IndicatorResult(tau=-0.1, p_surrogate=0.8, n_surrogates=100),
            "skewness": IndicatorResult(tau=-0.2, p_surrogate=0.5, n_surrogates=100),
        },
    )
    assert resultado_minoria.warning is False


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_critical_slowing_down_on_real_uci_data_is_severely_underpowered():
    """
    ACHADO REAL, honesto sobre o limite do proprio metodo mais
    rigoroso: o detector completo (min_points=8 por padrao) so pode
    ser aplicado a pacientes com >=8 encontros -- 320 de 71.518
    (0,45%). Restringindo ainda mais para permitir um holdout real
    (deteccao nos 8 primeiros encontros, checagem de evento nos
    encontros 9+, sem vazar futuro -- mesma disciplina de
    biospace.survival), sobra so 139 pacientes (0,19%), dos quais
    apenas 1 mostra warning=True -- poder estatistico insuficiente
    para qualquer teste de associacao com desfecho futuro. Isto
    REFORCA, com um metodo ainda mais exigente e rigoroso, a mesma
    conclusao do achado documentado com o metodo mais simples: a
    limitacao e' do DADO (poucos encontros de acompanhamento por
    paciente na UCI), nao do metodo escolhido.
    """
    from biospace.core import Trajectory
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", include_diagnosis_category=False)

    cohort_deteccao = Cohort()
    eventos_futuros = {}
    for sid, system in cohort.systems.items():
        obs = system.observations
        if len(obs) < 10:
            continue
        traj_completa = cohort.trajectories[sid]
        nova_traj = Trajectory(system_id=sid)
        for i in range(8):
            nova_traj._points.append(traj_completa.at(i))
        cohort_deteccao.systems[sid] = system
        cohort_deteccao.trajectories[sid] = nova_traj
        eventos_futuros[sid] = any(o.metadata.get("readmitted") == "<30" for o in obs[8:])

    assert 100 < len(cohort_deteccao.trajectories) < 200, (
        f"Esperava ~139 pacientes elegiveis (achado documentado) -- obteve {len(cohort_deteccao.trajectories)}"
    )

    detector = CriticalSlowingDownDetector.for_feature("utilization", "num_medications", min_points=8, window_size=4, n_surrogates=100)
    resultados = detector.fit(cohort_deteccao)

    n_com_warning = sum(1 for r in resultados.values() if r.sufficient_data and r.warning)
    assert n_com_warning <= 5, (
        f"Esperava poucos casos de warning=True nesta subamostra pequena (achado documentado: severamente subamostrado) -- obteve {n_com_warning}"
    )
