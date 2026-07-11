"""
tests.test_stability_robustness
===================================

check_feature_stability_robustness: fecha uma lacuna deixada pendente
há muitas sessões — `hypoxia.tempo_total_em_hipoxemia_min` parecia
divergir (φ=1,0039) na coorte real de SAOS, registrado como "mereceria
investigação dedicada". Investigado agora: a Feature tem distribuição
fortemente assimétrica (mediana=0); removendo 1 único paciente (o
outlier de valor=135, de ~296 no total), φ cai para 0,984 (estável) — a
"divergência" nunca foi sinal populacional real, era sensibilidade a um
outlier. Formalizado aqui como diagnóstico reutilizável, testado em
cenários sintéticos com resultado CONHECIDO antes de confiar em dados
reais.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.dynamics import check_feature_stability_robustness


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_FlagObservable("x")])

    def encode(self, measurements):
        v = float(measurements["x"].value)
        return [Feature(name="x", value=v, raw_value=v)]


def _build_mostly_stable_cohort_with_one_outlier_patient(n_patients=60, seed=0):
    """
    N-1 pacientes genuinamente estáveis (reversão à média clara); 1
    paciente com uma trajetória extrema e crescente, isolado, capaz de
    dominar o ajuste se a amostra for pequena o bastante -- reproduz em
    miniatura o achado real de hypoxia.tempo_total_em_hipoxemia_min.
    """
    rng = np.random.default_rng(seed)
    mu, phi_true = 5.0, 0.7
    representation = Representation([_Dom()])
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)

    for i in range(n_patients - 1):
        x = rng.normal(mu, 1.0)
        system = BiologicalSystem(identifier=f"p{i}")
        t = t0
        system.observe(Observation(timestamp=t, source="t", values={"x": x}))
        cohort.update(system, representation, timestamp=t)
        for _ in range(4):
            t = t + timedelta(days=30)
            x = mu + phi_true * (x - mu) + rng.normal(0, 0.5)
            system.observe(Observation(timestamp=t, source="t", values={"x": x}))
            cohort.update(system, representation, timestamp=t)

    outlier = BiologicalSystem(identifier="outlier_extremo")
    t = t0
    x = mu
    outlier.observe(Observation(timestamp=t, source="t", values={"x": x}))
    cohort.update(outlier, representation, timestamp=t)
    for delta in [5, 20, 60, 150]:
        t = t + timedelta(days=30)
        x = x + delta
        outlier.observe(Observation(timestamp=t, source="t", values={"x": x}))
        cohort.update(outlier, representation, timestamp=t)

    return cohort, representation


def test_robustness_report_flags_single_outlier_patient():
    """O TESTE DECISIVO: reproduz em miniatura o achado real -- 1 paciente outlier isolado pode flipar a conclusão de estabilidade."""
    cohort, representation = _build_mostly_stable_cohort_with_one_outlier_patient()
    order = representation.domain_names()

    relatorio = check_feature_stability_robustness(cohort, "d.x", order=order, max_patients_tested=15)

    assert "outlier_extremo" in relatorio.phi_jackknife
    influente = relatorio.most_influential_patient
    assert influente is not None
    assert influente[0] == "outlier_extremo", "O paciente mais influente deveria ser o outlier isolado, não um dos estáveis."


def test_robustness_report_is_robust_when_no_outliers_present():
    """Contraprova: sem nenhum outlier isolado, a conclusão de estabilidade deveria ser robusta à remoção de qualquer paciente individual."""
    rng = np.random.default_rng(1)
    mu, phi_true = 5.0, 0.7
    representation = Representation([_Dom()])
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)
    for i in range(40):
        x = rng.normal(mu, 1.0)
        system = BiologicalSystem(identifier=f"p{i}")
        t = t0
        system.observe(Observation(timestamp=t, source="t", values={"x": x}))
        cohort.update(system, representation, timestamp=t)
        for _ in range(4):
            t = t + timedelta(days=30)
            x = mu + phi_true * (x - mu) + rng.normal(0, 0.5)
            system.observe(Observation(timestamp=t, source="t", values={"x": x}))
            cohort.update(system, representation, timestamp=t)

    order = representation.domain_names()
    relatorio = check_feature_stability_robustness(cohort, "d.x", order=order, max_patients_tested=20)
    assert relatorio.conclusion_is_robust is True


def test_robustness_report_raises_for_unknown_feature():
    cohort, representation = _build_mostly_stable_cohort_with_one_outlier_patient(n_patients=10)
    order = representation.domain_names()
    with pytest.raises(KeyError):
        check_feature_stability_robustness(cohort, "d.feature_que_nao_existe", order=order)


def test_real_sleep_hypoxemia_finding_is_reproducible():
    """
    Registra o achado real (dados reais de SAOS) como teste de
    regressão: 'hypoxia.tempo_total_em_hipoxemia_min' deveria continuar
    aparecendo como não-robusto (achado documentado no README), a menos
    que os dados de entrada mudem.
    """
    import os

    caminho = "/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx"
    if not os.path.exists(caminho):
        pytest.skip("Excel real não disponível neste ambiente de teste (esperado em CI).")

    from biospace.plugins.sleep import load_from_excel

    cohort, representation = load_from_excel(caminho, header=1)
    order = representation.domain_names()
    relatorio = check_feature_stability_robustness(cohort, "hypoxia.tempo_total_em_hipoxemia_min", order=order, max_patients_tested=10)
    assert relatorio.conclusion_is_robust is False
    assert relatorio.most_influential_patient[0] == "sleep_ls_000035"
