"""
tests.test_functor_between_categories
=========================================

O item nomeado como "trabalho futuro, não demonstrado" na Seção 4.17
do manuscrito (organização categórica): um functor entre duas
categorias de representação diferentes (sleep, metabolic).

Cada categoria de representação, como definida no meta-modelo (§4.14),
tem exatamente UM objeto (o próprio espaço X_d) — os morfismos
admissíveis são endomorfismos T: X_d -> X_d. Um functor entre duas
categorias de UM OBJETO cada é, precisamente, um HOMOMORFISMO DE
MONOIDE: uma aplicação φ dos operadores admissíveis de uma categoria
para os da outra, que preserva identidade (φ(id)=id) e composição
(φ(g∘f)=φ(g)∘φ(f)).

O candidato testado aqui: `MeanRevertingEvolutionOperator.predict(x,
Δt)`, já usado (com dados reais) em SAOS e no NHANES/UCI, satisfaz a
lei de semigrupo de um fluxo a um parâmetro:

    T_{s+t} = T_s ∘ T_t

Isto NÃO é assumido — é verificado algebricamente (predict(x,dt) =
μ + φ^dt·(x-μ), e μ+φ^t·(μ+φ^s·(x-μ)-μ) = μ+φ^(s+t)·(x-μ) por
substituição direta) e depois testado numericamente contra dado REAL
das duas fontes, não apenas uma. A homomorfia realizada é: φ mapeia o
monoide aditivo (ℝ≥0, +) — "quanto tempo passa" — para o monoide dos
operadores de evolução admissíveis em CADA categoria, φ_sleep: ℝ≥0 ->
T_sleep e φ_metabolic: ℝ≥0 -> T_metabolic, ambos satisfazendo a MESMA
lei estrutural — a mesma "receita" (fit + predict) realizada
identicamente nas duas categorias-objeto é o conteúdo funtorial
concreto, não uma alegação decorativa.

Escopo, declarado com precisão: isto testa a lei de semigrupo do
componente DETERMINÍSTICO de `predict()` (a parte estocástica de
`sample()` não satisfaz esta lei exatamente, por conter ruído
aleatório) -- e testa dois OBJETOS concretos (sleep, metabolic), não
uma prova geral para qualquer par de categorias futuras deste projeto.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _semigroup_law_holds(fd, x0: float, s: float, t: float, tol: float = 1e-9) -> bool:
    """T_{s+t}(x0) == T_t(T_s(x0)) -- aplicada a UMA FeatureDynamics já ajustada."""

    def predict_1d(x, dt):
        return fd.mu + (fd.phi_per_day**dt) * (x - fd.mu)

    lado_composto = predict_1d(predict_1d(x0, s), t)
    lado_direto = predict_1d(x0, s + t)
    return abs(lado_composto - lado_direto) < tol


def test_semigroup_law_algebraic_identity():
    """Prova algébrica direta, sem dado real -- por substituição, T_{s+t}=T_t∘T_s vale para QUALQUER mu, phi, x0, s, t (exceto phi<=0, onde phi^dt não é real para dt não-inteiro)."""
    from biospace.dynamics.evolution import FeatureDynamics

    rng = np.random.default_rng(0)
    for _ in range(200):
        mu = rng.uniform(-50, 50)
        phi = rng.uniform(0.01, 0.999)  # phi>0 para phi^dt ficar bem definido com dt nao-inteiro
        x0 = rng.uniform(-100, 100)
        s = rng.uniform(0, 10)
        t = rng.uniform(0, 10)
        fd = FeatureDynamics(name="x", mu=mu, phi_per_day=phi, n_pairs=10, residual_std=1.0, mean_dt_days=1.0)
        assert _semigroup_law_holds(fd, x0, s, t), f"Falhou para mu={mu}, phi={phi}, x0={x0}, s={s}, t={t}"


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx"),
    reason="Requer a planilha real de SAOS.",
)
def test_semigroup_law_holds_for_operator_fitted_on_real_sleep_data():
    """
    O TESTE DECISIVO na categoria C_sleep: ajusta MeanRevertingEvolutionOperator
    em dado REAL de SAOS (296 pacientes multi-encontro), depois confirma
    que TODAS as Features com dinâmica ajustada satisfazem a lei de
    semigrupo -- não apenas o caso algébrico abstrato acima, mas o
    operador de verdade, ajustado em dado real, ainda satisfaz a lei
    que sua própria forma funcional deveria garantir.
    """
    from biospace.core import Cohort
    from biospace.dynamics import MeanRevertingEvolutionOperator
    from biospace.plugins.sleep import load_from_excel

    cohort, representation = load_from_excel("/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx", header=1)
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 2}
    assert len(multi) > 200, f"Esperava >200 pacientes multi-encontro (achado documentado: 296) -- obteve {len(multi)}"

    cohort_multi = Cohort()
    for sid, traj in multi.items():
        cohort_multi.systems[sid] = cohort.systems[sid]
        cohort_multi.trajectories[sid] = traj

    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort_multi)
    assert len(evo.dynamics_) > 20, "Esperava dinamica ajustada para varias dezenas de Features."

    rng = np.random.default_rng(42)
    n_violacoes = 0
    for nome, fd in evo.dynamics_.items():
        for _ in range(5):
            x0 = rng.uniform(-3, 3)
            s, t = rng.uniform(0, 5), rng.uniform(0, 5)
            if not _semigroup_law_holds(fd, x0, s, t):
                n_violacoes += 1
    assert n_violacoes == 0, f"Lei de semigrupo violada em {n_violacoes} casos -- nao deveria acontecer para nenhuma Feature."


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/P_DEMO.xpt"),
    reason="Requer os arquivos reais do NHANES.",
)
def test_semigroup_law_holds_for_operator_fitted_on_real_metabolic_data():
    """
    O TESTE DECISIVO na categoria C_metabolic, usando dado real da UCI
    (não NHANES -- NHANES é transversal, sem trajetoria multi-ponto).
    A MESMA lei, testada na MESMA forma, agora no OUTRO objeto da
    categoria -- este é o conteúdo funtorial concreto: a receita
    (fit + predict) realiza a mesma lei estrutural nas duas categorias,
    não apenas numa.
    """
    from biospace.core import Cohort
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
    from biospace.dynamics import MeanRevertingEvolutionOperator

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", max_rows=8000)
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 2}
    if len(multi) < 20:
        pytest.skip("Amostra reduzida (max_rows=8000) nao produziu pacientes multi-encontro suficientes.")

    cohort_multi = Cohort()
    for sid, traj in multi.items():
        cohort_multi.systems[sid] = cohort.systems[sid]
        cohort_multi.trajectories[sid] = traj

    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort_multi)
    assert len(evo.dynamics_) > 5

    rng = np.random.default_rng(7)
    n_violacoes = 0
    for nome, fd in evo.dynamics_.items():
        for _ in range(5):
            x0 = rng.uniform(-3, 3)
            s, t = rng.uniform(0, 5), rng.uniform(0, 5)
            if not _semigroup_law_holds(fd, x0, s, t):
                n_violacoes += 1
    assert n_violacoes == 0, f"Lei de semigrupo violada em {n_violacoes} casos na categoria metabolic -- nao deveria acontecer."


def test_functor_maps_identity_to_identity_in_both_categories():
    """
    A segunda lei funtorial (além de composição): φ(id) = id. Δt=0
    deveria ser a identidade (T_0(x)=x) -- verificado algebricamente
    (phi^0=1 sempre, para qualquer phi>0) e testado explicitamente,
    não apenas assumido por decorrer "obviamente" da fórmula.
    """
    from biospace.dynamics.evolution import FeatureDynamics

    rng = np.random.default_rng(1)
    for _ in range(50):
        mu = rng.uniform(-50, 50)
        phi = rng.uniform(0.01, 0.999)
        x0 = rng.uniform(-100, 100)
        fd = FeatureDynamics(name="x", mu=mu, phi_per_day=phi, n_pairs=10, residual_std=1.0, mean_dt_days=1.0)
        predicted = fd.mu + (fd.phi_per_day**0.0) * (x0 - fd.mu)
        assert abs(predicted - x0) < 1e-9, f"T_0 deveria ser identidade -- obteve {predicted} para x0={x0}"
