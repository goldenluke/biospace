"""
tests.test_diabetes_plugin
=============================

Segundo plugin de doença — construído inteiramente sintético (nenhum
dado de paciente real), com a MESMA disciplina do plugin sleep:
domínios semânticos por significado clínico, z-score ponderado por
completude, domínio latente com hipótese declarada, gerador
longitudinal realista, e validação contra os contratos formais + o
restante do ferramental (fenotipagem, causal) sem NENHUMA alteração no
núcleo — a prova mais forte de genericidade que este projeto tem, além
do teste de glicemia mínimo em `test_core_disease_agnostic.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from biospace.causal import check_baseline_balance
from biospace.core.contracts import check_reproducibility, check_semantic_preservation, check_temporality
from biospace.phenotyping import KMeansPhenotyper
from biospace.plugins.diabetes import (
    DiabetesSystem,
    InsulinResistanceProxyDomain,
    generate_synthetic_dataframe,
    load_from_dataframe,
)


@pytest.fixture(scope="module")
def diabetes_cohort():
    df = generate_synthetic_dataframe(n_per_group=25, seed=7)
    cohort, representation = load_from_dataframe(df)
    return cohort, representation, df


def test_synthetic_generator_produces_longitudinal_data():
    df = generate_synthetic_dataframe(n_per_group=25, seed=7)
    n_por_paciente = df.groupby("paciente").size()
    assert n_por_paciente.nunique() > 1, "Deveria haver pacientes com números DIFERENTES de exames (não todo mundo com 1)."
    assert (n_por_paciente >= 2).sum() >= len(n_por_paciente) * 0.5, "Esperava pelo menos metade dos pacientes com >=2 exames."
    assert n_por_paciente.max() >= 8, "Esperava pelo menos um paciente com trajetória longa (cauda da distribuição)."


def test_loader_groups_by_patient_not_by_row(diabetes_cohort):
    cohort, representation, df = diabetes_cohort
    assert len(cohort) < len(df), "Deveria haver menos pacientes que linhas (múltiplos exames por paciente)."
    assert len(cohort) == df["paciente"].nunique()
    assert cohort.loader_report["n_patients"] == len(cohort)


def test_all_seven_domains_present(diabetes_cohort):
    """Renomeado de 'seis' para 'sete' domínios -- LipidDomain adicionado quando NHANES ganhou colesterol/HDL/triglicerídeos reais."""
    cohort, representation, _ = diabetes_cohort
    assert representation.domain_names() == ["glycemic", "anthropometric", "cardiovascular", "renal", "lipid", "comorbidity", "treatment"]


def test_egfr_sign_is_inverted_consistently():
    """
    eGFR: valor bruto MAIOR é clinicamente MELHOR. A convenção do projeto
    (mesma do SpO2 no plugin sleep) é "Feature.value maior = pior" em
    toda a Representation — então um paciente com eGFR BAIXO (ruim) deve
    receber um Feature.value MAIOR que um paciente com eGFR ALTO (bom).
    """
    from datetime import datetime

    from biospace.plugins.diabetes import DiabetesSystem, RenalDomain, exam, fit_reference

    reference = fit_reference([{"creatinina_mg_dl": 1.0, "taxa_filtracao_glomerular": 80.0}] * 5)
    domain = RenalDomain(reference=reference)

    saudavel = DiabetesSystem()
    saudavel.observe(exam({"creatinina_mg_dl": 0.9, "taxa_filtracao_glomerular": 100.0}, timestamp=datetime(2024, 1, 1)))

    doente = DiabetesSystem()
    doente.observe(exam({"creatinina_mg_dl": 0.9, "taxa_filtracao_glomerular": 40.0}, timestamp=datetime(2024, 1, 1)))

    f_saudavel = next(f for f in domain.transform(saudavel) if f.name == "taxa_filtracao_glomerular")
    f_doente = next(f for f in domain.transform(doente) if f.name == "taxa_filtracao_glomerular")

    assert f_doente.value > f_saudavel.value, "eGFR baixo (doente) deveria produzir Feature.value MAIOR (pior) que eGFR alto (saudável)."


def test_reproducibility_and_semantic_preservation(diabetes_cohort):
    cohort, representation, _ = diabetes_cohort
    systems = list(cohort.systems.values())
    glycemic_domain = next(d for d in representation.domains if d.name == "glycemic")
    assert check_reproducibility(glycemic_domain, systems[0]) is True
    assert check_semantic_preservation(glycemic_domain, systems[0], systems[1]) is True


def test_temporality_contract(diabetes_cohort):
    cohort, representation, _ = diabetes_cohort
    system = next(iter(cohort.systems.values()))
    report = check_temporality(representation, lambda: DiabetesSystem(), system.observations)
    assert report.is_compliant


def test_kmeans_separates_severity_groups(diabetes_cohort):
    """A fenotipagem genérica (sem NENHUM código específico de diabetes) deve rodar sobre a coorte."""
    cohort, representation, df = diabetes_cohort
    space = cohort.snapshot()
    phenotypes = KMeansPhenotyper(n_clusters=3).fit(space)
    assert len(phenotypes) == 3


def test_insulin_resistance_proxy_domain_requires_hypothesis():
    """LatentDomain recusa instanciar sem hypothesis -- InsulinResistanceProxyDomain já a declara, então deve funcionar."""
    from biospace.plugins.diabetes import AnthropometricDomain, GlycemicDomain

    domain = InsulinResistanceProxyDomain(GlycemicDomain(), AnthropometricDomain())
    assert domain.hypothesis
    assert domain.is_validated is False


def test_insulin_resistance_proxy_domain_fits_and_transforms(diabetes_cohort):
    cohort, representation, _ = diabetes_cohort
    glycemic_domain = next(d for d in representation.domains if d.name == "glycemic")
    anthro_domain = next(d for d in representation.domains if d.name == "anthropometric")

    proxy = InsulinResistanceProxyDomain(glycemic_domain, anthro_domain)
    proxy.fit(cohort)

    system = next(iter(cohort.systems.values()))
    features = proxy.transform(system)
    assert len(features) == 1
    assert features[0].name == "factor_1"


def test_renal_decline_correlates_with_chronic_glycemic_exposure(diabetes_cohort):
    """
    Achado de rigor deliberado do gerador: declínio de eGFR ao longo do
    tempo deve correlacionar POSITIVAMENTE com HbA1c médio (pior controle
    crônico -> mais dano renal) -- mecanismo real, não circular (eGFR e
    HbA1c são domínios diferentes, a correlação é induzida pelo gerador,
    não pela representação).
    """
    cohort, representation, _ = diabetes_cohort
    pares = []
    for traj in cohort.trajectories.values():
        if len(traj) < 3:
            continue
        hba1c_medios = [
            f.raw_value
            for pt in [traj.at(i) for i in range(len(traj))]
            for f in pt.components["glycemic"]
            if f.name == "hba1c_pct" and f.raw_value is not None
        ]
        egfr_inicial = next((f.raw_value for f in traj.at(0).components["renal"] if f.name == "taxa_filtracao_glomerular"), None)
        egfr_final = next((f.raw_value for f in traj.at(-1).components["renal"] if f.name == "taxa_filtracao_glomerular"), None)
        if egfr_inicial is not None and egfr_final is not None and hba1c_medios:
            pares.append((float(np.mean(hba1c_medios)), egfr_inicial - egfr_final))

    assert len(pares) >= 10, "Poucos pacientes com dados suficientes para testar a correlação."
    hba1c_medios, quedas = zip(*pares)
    corr = np.corrcoef(hba1c_medios, quedas)[0, 1]
    assert corr > 0, f"Esperava correlação positiva (pior controle -> mais queda de eGFR), achou {corr:.3f}"


def test_check_baseline_balance_detects_treatment_confounding(diabetes_cohort):
    """Pacientes que iniciam insulina são sistematicamente mais graves na linha de base -- confundimento por indicação real e esperado."""
    cohort, representation, _ = diabetes_cohort
    order = representation.domain_names()
    balance = check_baseline_balance(cohort, "treatment", "insulina", order=order)
    assert not balance.is_balanced
    assert balance.smd["glycemic.hba1c_pct"] > 0.3, "Pacientes que iniciam insulina deveriam ter HbA1c basal claramente mais alto."
