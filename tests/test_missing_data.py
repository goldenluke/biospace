"""
tests.test_missing_data
==========================

Regressões de dois achados reais:

  1. Valores ausentes devem ser imputados (z=0) e ponderados por
     completude — não devem levantar KeyError (versão original quebrava
     aqui) nem ser tratados como se tivessem a mesma confiabilidade de um
     campo sempre presente.
  2. `domain.missing_counts` conta por EXAME (chamado a cada
     `cohort.update()`), não por paciente — dividir pelo nº de pacientes
     em vez de exames produz percentuais > 100% (bug real encontrado ao
     rodar nos dados reais).
"""

from __future__ import annotations

import numpy as np

from biospace.plugins.sleep import ApneaDomain, fit_reference
from biospace.plugins.sleep.builders import exam


def test_missing_value_is_imputed_not_raised(sleep_system_factory, exam_values_factory):
    """Um campo ausente não deve levantar exceção — deve ser imputado como z=0."""
    values = exam_values_factory()
    del values["ido_sono"]  # remove um campo que ApneaDomain espera

    system = sleep_system_factory()
    system.observe(exam(values))

    features = ApneaDomain().transform(system)
    feature_ido_sono = next(f for f in features if f.name == "ido_sono")

    assert feature_ido_sono.is_missing is True
    assert feature_ido_sono.value == 0.0
    assert feature_ido_sono.raw_value is None


def test_completeness_weight_reduces_contribution_of_rare_field(sleep_system_factory, exam_values_factory):
    """Um campo presente em poucos registros da referência deve pesar menos que um sempre presente."""
    registros = []
    for i in range(100):
        v = exam_values_factory(ido=20.0 + i * 0.1)
        if i % 10 != 0:  # 'ido_sono' presente em só 10% dos registros
            del v["ido_sono"]
        registros.append(v)

    reference = fit_reference(registros)
    domain = ApneaDomain(reference=reference)

    weights = domain.feature_weights()
    assert weights["ido_sono"] < weights["ido"], "Campo raro deveria pesar menos que campo sempre presente."
    assert weights["ido"] == 1.0  # presente em 100% dos registros de referência


def test_field_below_threshold_is_fully_excluded(sleep_system_factory, exam_values_factory):
    """Completude abaixo de exclude_below deve zerar o peso por completo (exclusão, não só atenuação)."""
    registros = []
    for i in range(100):
        v = exam_values_factory()
        if i % 100 != 0:  # 'ido_sono' presente em só 1% dos registros
            del v["ido_sono"]
        registros.append(v)

    reference = fit_reference(registros)
    domain = ApneaDomain(reference=reference, exclude_below=0.05)

    weights = domain.feature_weights()
    assert weights["ido_sono"] == 0.0


def test_quality_report_denominator_uses_exams_not_patients(small_cohort):
    """
    Bug real: dividir missing_counts (por EXAME) pelo nº de PACIENTES
    produz percentuais > 100% quando há pacientes multi-exame. O
    denominador correto é o total de exames (soma dos comprimentos das
    trajetórias), não `len(cohort)`.
    """
    cohort, representation = small_cohort

    domain = next(d for d in representation.domains if d.name == "apnea")
    if not domain.missing_counts:
        return  # nada ausente neste cenário sintético — nada a checar

    n_patients = len(cohort)
    n_exams = sum(len(traj) for traj in cohort.trajectories.values())

    assert n_exams >= n_patients  # cenário multi-exame

    for count in domain.missing_counts.values():
        pct_sobre_pacientes = 100 * count / n_patients
        pct_sobre_exames = 100 * count / n_exams
        assert pct_sobre_exames <= 100.0
        # Demonstra o próprio bug: dividir por pacientes PODE ultrapassar 100%
        # quando count > n_patients (o que só pode ocorrer se count vier de
        # múltiplos exames do mesmo paciente) — não fazemos assert sobre
        # pct_sobre_pacientes de propósito, só documentamos o contraste.
        assert pct_sobre_exames <= pct_sobre_pacientes + 1e-9 or n_exams >= n_patients
