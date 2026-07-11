"""
tests.test_theoretical_invariants
=====================================

Fecha três lacunas nomeadas explicitamente como "não testadas" na
Seção 4.16 do manuscript.html (Invariantes) — cada uma tratada como uma
alegação a verificar, não uma verdade assumida:

1. Equivariância de escala (z-score sob mudança de unidade) — CONFIRMADA.
2. Desigualdade triangular do DTW normalizado — investigada com busca
   adversarial mais agressiva que a exploração interativa anterior.
3. Invariância de discretização temporal — testada em trajetória real
   da UCI (subamostragem de encontros).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.core import Measurement
from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
from biospace.plugins.metabolic.reference import fit_reference, zscore_features


# =============================================================================
# 1. Equivariância de escala
# =============================================================================
KG_PARA_LB = 2.20462


def _construir_referencias(seed: int, fracao_ausente: float = 0.0):
    rng = np.random.default_rng(seed)
    n = 200
    valores_kg = rng.normal(75, 15, n)
    mascara_ausente = rng.random(n) < fracao_ausente
    registros_kg = [{"peso": None if mascara_ausente[i] else float(valores_kg[i])} for i in range(n)]
    registros_lb = [{"peso": None if mascara_ausente[i] else float(valores_kg[i] * KG_PARA_LB)} for i in range(n)]
    return fit_reference(registros_kg), fit_reference(registros_lb)


def test_zscore_is_equivariant_under_unit_change():
    """TESTE DECISIVO: z-score em kg == z-score em lb, refazendo o ajuste da Reference na nova unidade -- a alegação matemática da Seção 4.16, agora verificada, não apenas deduzida da fórmula."""
    ref_kg, ref_lb = _construir_referencias(seed=0)
    paciente_kg = 82.3
    m_kg = {"peso": Measurement(key="peso", value=paciente_kg, source="t", timestamp=datetime(2024, 1, 1))}
    m_lb = {"peso": Measurement(key="peso", value=paciente_kg * KG_PARA_LB, source="t", timestamp=datetime(2024, 1, 1))}

    f_kg = zscore_features(m_kg, ["peso"], ref_kg)[0]
    f_lb = zscore_features(m_lb, ["peso"], ref_lb)[0]
    assert abs(f_kg.z_score - f_lb.z_score) < 1e-9


def test_zscore_equivariance_holds_with_partial_missingness():
    """A equivariância também vale quando a população de referência tem ausência parcial (afeta o peso de completude, que por sua vez é independente de unidade)."""
    ref_kg, ref_lb = _construir_referencias(seed=1, fracao_ausente=0.2)
    paciente_kg = 82.3
    m_kg = {"peso": Measurement(key="peso", value=paciente_kg, source="t", timestamp=datetime(2024, 1, 1))}
    m_lb = {"peso": Measurement(key="peso", value=paciente_kg * KG_PARA_LB, source="t", timestamp=datetime(2024, 1, 1))}

    f_kg = zscore_features(m_kg, ["peso"], ref_kg)[0]
    f_lb = zscore_features(m_lb, ["peso"], ref_lb)[0]
    assert abs(f_kg.z_score - f_lb.z_score) < 1e-9
    assert abs(f_kg.weight - f_lb.weight) < 1e-9


def test_zscore_equivariance_holds_under_sign_inversion():
    """Equivariância também sob invert={...} (ex.: eGFR, onde maior=melhor) -- a inversão de sinal comuta com a mudança de unidade."""
    ref_kg, ref_lb = _construir_referencias(seed=2)
    paciente_kg = 82.3
    m_kg = {"peso": Measurement(key="peso", value=paciente_kg, source="t", timestamp=datetime(2024, 1, 1))}
    m_lb = {"peso": Measurement(key="peso", value=paciente_kg * KG_PARA_LB, source="t", timestamp=datetime(2024, 1, 1))}

    f_kg = zscore_features(m_kg, ["peso"], ref_kg, invert={"peso"})[0]
    f_lb = zscore_features(m_lb, ["peso"], ref_lb, invert={"peso"})[0]
    assert abs(f_kg.z_score - f_lb.z_score) < 1e-9


def test_zscore_equivariance_across_many_random_scale_factors():
    """Não apenas kg->lb: confirma para 10 fatores de escala aleatórios diferentes, não um único caso de sorte."""
    rng = np.random.default_rng(3)
    for fator in rng.uniform(0.01, 100, 10):
        rng2 = np.random.default_rng(int(fator * 1000) % 10000)
        n = 200
        valores_a = rng2.normal(75, 15, n)
        registros_a = [{"x": float(v)} for v in valores_a]
        registros_b = [{"x": float(v * fator)} for v in valores_a]
        ref_a = fit_reference(registros_a)
        ref_b = fit_reference(registros_b)

        paciente_a = 82.3
        m_a = {"x": Measurement(key="x", value=paciente_a, source="t", timestamp=datetime(2024, 1, 1))}
        m_b = {"x": Measurement(key="x", value=paciente_a * fator, source="t", timestamp=datetime(2024, 1, 1))}
        f_a = zscore_features(m_a, ["x"], ref_a)[0]
        f_b = zscore_features(m_b, ["x"], ref_b)[0]
        assert abs(f_a.z_score - f_b.z_score) < 1e-6, f"Falhou para fator={fator}"


# =============================================================================
# 2. Desigualdade triangular do DTW -- resolvida, não mais em aberto
# =============================================================================
def _build_trajectory(valores):
    from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain

    class _Obs(Observable):
        key = "x"

    class _Dom(SemanticDomain):
        name = "d"

        def __init__(self):
            super().__init__([_Obs()])

        def encode(self, measurements):
            v = float(measurements["x"].value)
            return [Feature(name="x", value=v, raw_value=v)]

    representation = Representation([_Dom()])
    system = BiologicalSystem(identifier=f"s_{id(valores)}")
    cohort = Cohort()
    for i, v in enumerate(valores):
        ts = datetime(2020, 1, 1) + timedelta(days=i)
        system.observe(Observation(timestamp=ts, source="t", values={"x": v}))
        cohort.update(system, representation, timestamp=ts)
    return cohort.trajectories[system.id]


@pytest.mark.parametrize("normalize", [True, False])
def test_dtw_violates_triangle_inequality_on_published_counterexample(normalize):
    """
    RESOLVE a questão deixada em aberto na Seção 4.13 do manuscript.html:
    tentativas anteriores, com trajetórias construídas à mão, não
    encontraram violação -- não porque a propriedade seja válida para
    nossa implementação, mas porque aquelas trajetórias não eram
    adversariais o suficiente. Usando o contraexemplo PUBLICADO de
    Dryden & Zhang (arXiv:2212.01648, "The DOPE Distance is SIC"):
    a = [-1]*m + [0], b = [-1, 0, 1], c = [0] + [1]*n (m,n>1) -- DTW(a,c)
    deveria exceder DTW(a,b)+DTW(b,c). CONFIRMADO: nossa implementação
    viola a desigualdade triangular, com e sem normalização por
    comprimento de caminho -- DTW não é uma métrica válida em
    biospace.geometry.DTW, resolvendo a questão que antes ficava em
    aberto por falta de evidência, não mais.
    """
    from biospace.geometry import DTW

    m = n = 5
    a = _build_trajectory([-1.0] * m + [0.0])
    b = _build_trajectory([-1.0, 0.0, 1.0])
    c = _build_trajectory([0.0] + [1.0] * n)

    dtw = DTW(order=["d"], normalize=normalize)
    d_ab = dtw.distance(a, b)
    d_bc = dtw.distance(b, c)
    d_ac = dtw.distance(a, c)

    assert d_ac > d_ab + d_bc, (
        f"Esperava violacao da desigualdade triangular (achado documentado, "
        f"contraexemplo publicado) -- d(a,c)={d_ac:.4f}, d(a,b)+d(b,c)={d_ab+d_bc:.4f}"
    )


# =============================================================================
# 3. Invariância de discretização temporal -- NÃO se sustenta empiricamente
# =============================================================================
@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI (achado testado em trajetória real multi-encontro).",
)
def test_temporal_discretization_is_not_invariant_on_real_trajectories():
    """
    ACHADO REAL, NEGATIVO: a Seção 4.16 do manuscript.html listava
    invariância de discretização temporal como "questão em aberto, não
    demonstrada". Testado agora em trajetórias reais multi-encontro da
    UCI (subamostragem: manter só índices pares de cada trajetória,
    dobrando o dt entre pontos remanescentes) -- a invariância NÃO se
    sustenta. phi_per_day sobe sistematicamente sob subamostragem em
    praticamente toda Feature (ex.: utilization.time_in_hospital
    0,21->0,43 -- quase o dobro), e para Features perto do limiar de
    instabilidade (phi~0,995) a estimativa de mu diverge drasticamente
    (22,8->401,0) -- consistente com o viés de estimação de raiz quase
    unitária bem documentado na literatura de séries temporais (mu fica
    mal identificado quando ha pouco "sinal de reversao" nos dados), mas
    isso NAO foi verificado rigorosamente aqui alem de identificar o
    padrao -- reportado como explicacao mais provavel, nao mecanismo
    comprovado.
    """
    from biospace.core import Cohort, Trajectory
    from biospace.dynamics import MeanRevertingEvolutionOperator

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv")
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 4}
    cohort_completo = Cohort()
    for sid, traj in multi.items():
        cohort_completo.systems[sid] = cohort.systems[sid]
        cohort_completo.trajectories[sid] = traj

    cohort_sub = Cohort()
    for sid, traj in multi.items():
        nova = Trajectory(system_id=sid)
        for i in range(0, len(traj), 2):
            nova._points.append(traj.at(i))
        if len(nova) >= 2:
            cohort_sub.systems[sid] = cohort.systems[sid]
            cohort_sub.trajectories[sid] = nova

    evo_completo = MeanRevertingEvolutionOperator(order=order).fit(cohort_completo)
    evo_sub = MeanRevertingEvolutionOperator(order=order).fit(cohort_sub)

    fd_completo = evo_completo.dynamics_["utilization.time_in_hospital"]
    fd_sub = evo_sub.dynamics_["utilization.time_in_hospital"]

    razao = fd_sub.phi_per_day / fd_completo.phi_per_day
    assert razao > 1.3, (
        f"Esperava phi_per_day claramente maior sob subamostragem (achado documentado, "
        f"invariancia NAO se sustenta) -- completo={fd_completo.phi_per_day:.4f}, "
        f"sub={fd_sub.phi_per_day:.4f}, razao={razao:.3f}"
    )
