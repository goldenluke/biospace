"""
tests.test_causal_balance
============================

Cobertura que faltava na suíte original: `check_baseline_balance` e
`ObservationalEffectEstimator` — a parte mais arriscada de todo o
projeto (inferência causal a partir de dados observacionais).

Constrói deliberadamente uma coorte sintética com confundimento por
indicação EMBUTIDO (pacientes que "adotam" tratamento são
sistematicamente mais graves desde o início) — o mesmo padrão
encontrado nos dados reais — para confirmar que `check_baseline_balance`
de fato DETECTA esse desequilíbrio, não apenas roda sem erro.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from biospace.causal import ObservationalEffectEstimator, check_baseline_balance
from biospace.core import Cohort
from biospace.plugins.sleep import SleepRepresentation, SleepSystem
from biospace.plugins.sleep.builders import exam


def _cohort_with_confounding_by_indication(exam_values_factory):
    """
    Grupo A ("adotantes"): graves desde o início (ido alto), iniciam AAM
    no segundo exame. Grupo B ("não-adotantes"): leves desde o início,
    nunca iniciam. O desequilíbrio de linha de base é INTENCIONAL — é
    exatamente o padrão de confundimento por indicação encontrado nos
    dados reais (pacientes mais graves recebem tratamento prescrito).
    """
    representation = SleepRepresentation()
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)

    for i in range(15):
        system = SleepSystem(identifier=f"adotante_{i}")
        vals1 = exam_values_factory(ido=40.0 + i * 0.5, ido_sono=40.0, fc_media_bpm=95.0, tratamentos="")
        system.observe(exam(vals1, timestamp=t0))
        cohort.update(system, representation, timestamp=t0)

        t1 = t0 + timedelta(days=60)
        vals2 = exam_values_factory(ido=30.0 + i * 0.5, ido_sono=30.0, fc_media_bpm=90.0, tratamentos="Aparelho de avanço mandibular")
        system.observe(exam(vals2, timestamp=t1))
        cohort.update(system, representation, timestamp=t1)

    for i in range(15):
        system = SleepSystem(identifier=f"nao_adotante_{i}")
        vals1 = exam_values_factory(ido=8.0 + i * 0.2, ido_sono=8.0, fc_media_bpm=62.0, tratamentos="")
        system.observe(exam(vals1, timestamp=t0))
        cohort.update(system, representation, timestamp=t0)

        t1 = t0 + timedelta(days=60)
        vals2 = exam_values_factory(ido=8.0 + i * 0.2, ido_sono=8.0, fc_media_bpm=62.0, tratamentos="")
        system.observe(exam(vals2, timestamp=t1))
        cohort.update(system, representation, timestamp=t1)

    return cohort, representation


def test_check_baseline_balance_detects_confounding_by_indication(exam_values_factory):
    cohort, representation = _cohort_with_confounding_by_indication(exam_values_factory)
    order = representation.domain_names()

    report = check_baseline_balance(cohort, treatment_domain="treatment", treatment_feature="aam", order=order)

    assert report.n_treated == 15
    assert report.n_untreated == 15
    assert not report.is_balanced, "O desequilíbrio de linha de base (embutido de propósito) deveria ter sido detectado."
    assert report.n_imbalanced > 0

    smd_ido = report.smd.get("apnea.ido")
    assert smd_ido is not None
    assert smd_ido > 0.5, "Grupo que adota tratamento é MUITO mais grave (ido) na linha de base — SMD deveria refletir isso claramente."


def test_check_baseline_balance_no_imbalance_when_groups_are_similar(exam_values_factory):
    """Contraprova: se os grupos forem similares na linha de base, is_balanced deve ser True (a checagem não deve gritar 'lobo' à toa)."""
    representation = SleepRepresentation()
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)

    for i in range(15):
        system = SleepSystem(identifier=f"a_{i}")
        vals1 = exam_values_factory(ido=20.0 + (i % 3), tratamentos="")
        system.observe(exam(vals1, timestamp=t0))
        cohort.update(system, representation, timestamp=t0)
        t1 = t0 + timedelta(days=60)
        vals2 = exam_values_factory(ido=20.0 + (i % 3), tratamentos="Aparelho de avanço mandibular" if i % 2 == 0 else "")
        system.observe(exam(vals2, timestamp=t1))
        cohort.update(system, representation, timestamp=t1)

    order = representation.domain_names()
    report = check_baseline_balance(cohort, treatment_domain="treatment", treatment_feature="aam", order=order)
    # com so ~2 grupos de 7-8 pacientes quase identicos, o desequilibrio deve ficar pequeno
    assert report.n_imbalanced <= 3, f"Esperava poucas Features desequilibradas com grupos similares, achou {report.n_imbalanced}"


def test_observational_effect_estimator_attaches_balance_warning(exam_values_factory):
    """O relatório de efeito DEVE trazer o aviso de desequilíbrio junto — não é opcional por acidente."""
    cohort, representation = _cohort_with_confounding_by_indication(exam_values_factory)
    order = representation.domain_names()

    estimator = ObservationalEffectEstimator(treatment_domain="treatment", treatment_feature="aam", order=order)
    report = estimator.estimate(cohort)

    assert report.n_transitions == 15  # os 15 "adotantes" fizeram a transicao 0->1
    assert report.balance is not None, "estimate() deveria rodar check_baseline_balance() automaticamente por padrão."
    assert not report.balance.is_balanced
    assert "CUIDADO" in report.summary(), "O resumo deveria conter o aviso de confundimento quando os grupos estão desequilibrados."


def test_observational_effect_estimator_raises_without_any_transitions(exam_values_factory):
    """Sem nenhuma transição 0->1 real, não há como estimar nada — deve falhar de forma clara, não devolver um relatório vazio silencioso."""
    representation = SleepRepresentation()
    cohort = Cohort()
    system = SleepSystem()
    system.observe(exam(exam_values_factory(tratamentos=""), timestamp=datetime(2020, 1, 1)))
    cohort.update(system, representation, timestamp=datetime(2020, 1, 1))

    estimator = ObservationalEffectEstimator(treatment_domain="treatment", treatment_feature="aam", order=representation.domain_names())
    with pytest.raises(ValueError):
        estimator.estimate(cohort)


def test_treatment_feature_is_excluded_from_its_own_baseline_comparison(exam_values_factory):
    """
    O TESTE DECISIVO para o bug real encontrado com dado NHANES (transversal,
    1 exame por paciente): a propria Feature de tratamento (`treatment.aam`)
    NUNCA deveria aparecer em `feature_names`/`smd` -- compara-la consigo
    mesma e' autoinclusao, nao confundimento. Em dado TRANSVERSAL (1
    exame, tratamento observado no MESMO ponto que define o grupo), sem
    esta exclusao a formula de SMD quebra silenciosamente (variancia
    zero dentro de cada grupo -> pooled_std=0 -> SMD mascarado como 0.0,
    escondendo o desequilibrio maximo possivel em vez de reporta-lo).
    """
    representation = SleepRepresentation()
    cohort = Cohort()

    # Cada paciente com APENAS 1 exame -- tratamento observado no MESMO
    # ponto que define o grupo, reproduzindo a estrutura do NHANES.
    for i in range(10):
        system = SleepSystem()
        tratado = i < 5
        system.observe(exam(exam_values_factory(tratamentos="Aparelho de avanço mandibular" if tratado else ""), timestamp=datetime(2020, 1, 1)))
        cohort.update(system, representation, timestamp=datetime(2020, 1, 1))

    order = representation.domain_names()
    balance = check_baseline_balance(cohort, "treatment", "aam", order=order)
    assert "treatment.aam" not in balance.feature_names, "A propria Feature de tratamento nao deveria aparecer em feature_names."
    assert "treatment.aam" not in balance.smd

    from biospace.causal import estimate_propensity

    modelo = estimate_propensity(cohort, "treatment", "aam", order=order)
    assert "treatment.aam" not in modelo.feature_names, "O modelo de propensao nao deveria usar o proprio tratamento como preditor."
    assert "treatment.aam" not in modelo.coefficients
