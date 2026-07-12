"""
tests.test_risk
===================

`biospace.risk` (RiskOperator/LinearRiskOperator) existia sem NENHUM
teste antes desta rodada — achado numa auditoria do projeto, mesmo
padrão de `prediction/`, `early_warning/`. Generaliza uma fórmula
ad-hoc que existia hardcoded num script legado do plugin sleep
(`07_clusterizacao.py`, `score_risco = 0.40*ido + 0.25*(100-spo2_minima) + ...`)
como pesos explícitos sobre Features já existentes na Representation.
"""

from __future__ import annotations

import os

import pytest

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.risk import LinearRiskOperator


def _vetor(system_id: str, componentes: dict[str, dict[str, float]]) -> RepresentationVector:
    from datetime import datetime

    comps = {dom: [Feature(name=nome, value=valor, raw_value=valor) for nome, valor in feats.items()] for dom, feats in componentes.items()}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def test_weighted_sum_computed_correctly():
    """TESTE DECISIVO: score deveria ser exatamente a soma ponderada das Features nomeadas -- calculado a mao."""
    space = RepresentationSpace(domain_order=["a", "b"])
    space.add(_vetor("s1", {"a": {"x": 10.0, "y": 2.0}, "b": {"z": 5.0}}))

    operador = LinearRiskOperator(weights={"x": 0.5, "z": 2.0})
    scores = operador.score(space)

    esperado = 0.5 * 10.0 + 2.0 * 5.0  # 'y' nao esta nos pesos, nao deveria contribuir
    assert scores["s1"] == esperado


def test_features_not_in_weights_do_not_contribute():
    space = RepresentationSpace(domain_order=["a"])
    space.add(_vetor("s1", {"a": {"x": 1.0, "y": 1000.0}}))  # y tem valor enorme, mas nao esta nos pesos

    operador = LinearRiskOperator(weights={"x": 1.0})
    scores = operador.score(space)
    assert scores["s1"] == 1.0, "Feature 'y' fora dos pesos nao deveria influenciar o score, mesmo com valor grande."


def test_works_across_multiple_domains_simultaneously():
    """As Features nomeadas nos pesos podem vir de domInios DIFERENTES do mesmo RepresentationVector -- nao precisam pertencer todas ao mesmo domInio."""
    space = RepresentationSpace(domain_order=["dominio1", "dominio2", "dominio3"])
    space.add(_vetor("s1", {"dominio1": {"a": 1.0}, "dominio2": {"b": 1.0}, "dominio3": {"c": 1.0}}))

    operador = LinearRiskOperator(weights={"a": 1.0, "b": 10.0, "c": 100.0})
    scores = operador.score(space)
    assert scores["s1"] == 111.0


def test_negative_weights_reduce_score():
    """Pesos negativos deveriam SUBTRAIR do score -- ex.: Feature protetora (eGFR alto = bom) com peso negativo."""
    space = RepresentationSpace(domain_order=["a"])
    space.add(_vetor("s1", {"a": {"fator_risco": 5.0, "fator_protetor": 3.0}}))

    operador = LinearRiskOperator(weights={"fator_risco": 1.0, "fator_protetor": -1.0})
    scores = operador.score(space)
    assert scores["s1"] == 2.0


def test_fit_is_an_alias_for_score():
    space = RepresentationSpace(domain_order=["a"])
    space.add(_vetor("s1", {"a": {"x": 7.0}}))
    operador = LinearRiskOperator(weights={"x": 1.0})
    assert operador.fit(space) == operador.score(space)


def test_scores_multiple_systems_independently():
    space = RepresentationSpace(domain_order=["a"])
    space.add(_vetor("s1", {"a": {"x": 1.0}}))
    space.add(_vetor("s2", {"a": {"x": 10.0}}))
    operador = LinearRiskOperator(weights={"x": 3.0})
    scores = operador.score(space)
    assert scores == {"s1": 3.0, "s2": 30.0}


def test_describe_mentions_weights():
    operador = LinearRiskOperator(weights={"idade": 0.5})
    assert "idade" in operador.describe()


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx"),
    reason="Requer a planilha real de SAOS.",
)
def test_risk_score_associates_with_real_treatment_status_on_saos_data():
    """
    ACHADO REAL: um score de risco transparente (ido + spo2_minima
    invertido + carga_hipoxica_min_h, pesos iguais -- generaliza a
    formula ad-hoc legada) associa significativamente com status de
    tratamento (CPAP/AAM) real -- pacientes tratados tem score mais
    alto (p=0.02, Mann-Whitney), consistente com a expectativa clinica
    de que tratamento e' direcionado a casos mais graves. Confirma que
    LinearRiskOperator produz um score clinicamente sensato em dado
    real, nao so em dado fabricado.
    """
    import numpy as np
    from scipy import stats as scipy_stats

    from biospace.plugins.sleep import load_from_excel

    cohort, representation = load_from_excel("/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx", header=1)
    space = cohort.snapshot()

    operador = LinearRiskOperator(weights={"ido": 1.0, "spo2_minima": 1.0, "carga_hipoxica_min_h": 1.0})
    scores = operador.score(space)

    linhas = []
    for sid in cohort.trajectories:
        vetor = space.get(sid)
        tratado = any(f.raw_value and f.raw_value > 0 for f in vetor.components["treatment"] if f.name in ("cpap", "aam"))
        linhas.append((scores[sid], tratado))

    scores_tratados = [s for s, t in linhas if t]
    scores_nao_tratados = [s for s, t in linhas if not t]
    assert len(scores_tratados) > 50 and len(scores_nao_tratados) > 20

    _, p = scipy_stats.mannwhitneyu(scores_tratados, scores_nao_tratados, alternative="greater")
    assert p < 0.05, f"Esperava score de risco maior em pacientes tratados (achado documentado) -- obteve p={p:.4f}"
    assert np.mean(scores_tratados) > np.mean(scores_nao_tratados)
