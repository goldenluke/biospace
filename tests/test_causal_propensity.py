"""
tests.test_causal_propensity
===============================

Validação do pareamento por escore de propensão (Rosenbaum & Rubin,
1983) — o "próximo passo real" que tínhamos deixado pendente depois da
primeira versão de `biospace.causal`.

Constrói um cenário sintético com confundimento por indicação FORTE e
CONHECIDO (o confundidor Z afeta tanto a probabilidade de tratamento
quanto o desfecho, independente do efeito do tratamento em si) — o
efeito causal verdadeiro é conhecido de propósito, para poder comparar
a estimativa ingênua (viesada) contra a estimativa pareada (deveria
estar mais perto da verdade).

Não usa nada do plugin sleep — dois domínios de brinquedo mínimos.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.causal import (
    ObservationalEffectEstimator,
    check_baseline_balance,
    estimate_matched_effect,
    match_on_propensity,
)
from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain

TRUE_EFFECT = -5.0
CONFOUND_BETA = 4.0


class _ZObservable(Observable):
    key = "z"


class _OutcomeObservable(Observable):
    key = "outcome"


class _TreatedObservable(Observable):
    key = "treated"


class _ConfounderDomain(SemanticDomain):
    name = "confounder"

    def __init__(self):
        super().__init__([_ZObservable()])

    def encode(self, measurements):
        m = measurements.get("z")
        return [Feature(name="z", value=float(m.value) if m else 0.0)]


class _OutcomeDomain(SemanticDomain):
    name = "outcome_domain"

    def __init__(self):
        super().__init__([_OutcomeObservable()])

    def encode(self, measurements):
        m = measurements.get("outcome")
        return [Feature(name="outcome", value=float(m.value) if m else 0.0)]


class _TreatmentDomain(SemanticDomain):
    name = "treatment"

    def __init__(self):
        super().__init__([_TreatedObservable()])

    def encode(self, measurements):
        m = measurements.get("treated")
        return [Feature(name="treated", value=float(m.value) if m else 0.0)]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def confounded_cohort():
    """
    N=200 pacientes sintéticos. Z (confundidor) afeta tanto P(tratamento)
    quanto o desfecho diretamente — confundimento por indicação forte e
    de magnitude CONHECIDA (CONFOUND_BETA), sobreposto a um efeito causal
    verdadeiro TAMBÉM conhecido (TRUE_EFFECT).
    """
    rng = np.random.default_rng(42)
    representation = Representation([_ConfounderDomain(), _OutcomeDomain(), _TreatmentDomain()])
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)

    for i in range(200):
        z = rng.normal(0, 1)
        treated = 1.0 if rng.random() < _sigmoid(2.0 * z) else 0.0
        baseline_outcome = rng.normal(50, 5)

        system = BiologicalSystem(identifier=f"p{i}")
        system.observe(Observation(timestamp=t0, source="baseline", values={"z": z, "outcome": baseline_outcome, "treated": 0.0}))
        cohort.update(system, representation, timestamp=t0)

        t1 = t0 + timedelta(days=90)
        follow_outcome = baseline_outcome + TRUE_EFFECT * treated + CONFOUND_BETA * z + rng.normal(0, 2)
        system.observe(Observation(timestamp=t1, source="follow_up", values={"z": z, "outcome": follow_outcome, "treated": treated}))
        cohort.update(system, representation, timestamp=t1)

    return cohort, representation


def test_naive_effect_is_biased_by_known_confounding(confounded_cohort):
    """Confirma que o cenário sintético realmente produz confundimento (pré-condição do teste principal, não o próprio ponto)."""
    cohort, representation = confounded_cohort
    order = representation.domain_names()

    naive = ObservationalEffectEstimator("treatment", "treated", order=order).estimate(cohort, run_balance_check=False)
    naive_effect = naive.delta_mean["outcome_domain.outcome"]

    # Viesado em direção a ZERO (menos negativo que o TRUE_EFFECT=-5), porque tratados tem Z medio mais alto,
    # e Z alto por si so ja aumenta o desfecho (CONFOUND_BETA positivo), mascarando parte do efeito real.
    assert naive_effect > TRUE_EFFECT + 1.0, (
        f"Esperava viés claro no efeito ingênuo (效 {naive_effect:.2f} deveria estar bem acima de {TRUE_EFFECT}); "
        "se isso falhar, o cenário sintético não está gerando confundimento como pretendido."
    )


def test_propensity_matching_reduces_bias_toward_true_effect(confounded_cohort):
    """O TESTE PRINCIPAL: o efeito pareado deve estar mais perto do TRUE_EFFECT conhecido do que o efeito ingênuo."""
    cohort, representation = confounded_cohort
    order = representation.domain_names()

    naive = ObservationalEffectEstimator("treatment", "treated", order=order).estimate(cohort, run_balance_check=False)
    naive_effect = naive.delta_mean["outcome_domain.outcome"]
    naive_error = abs(naive_effect - TRUE_EFFECT)

    match_result = match_on_propensity(cohort, "treatment", "treated", order=order, caliper=0.25)
    assert match_result.n_matched >= 20, "Poucos pares formados — teste não é conclusivo com amostra pequena demais."

    matched = estimate_matched_effect(cohort, match_result, order=order)
    matched_effect = matched.effect["outcome_domain.outcome"]
    matched_error = abs(matched_effect - TRUE_EFFECT)

    assert matched_error < naive_error, (
        f"O pareamento deveria reduzir o erro em relação ao efeito verdadeiro "
        f"(ingênuo: erro={naive_error:.2f}, pareado: erro={matched_error:.2f})."
    )
    assert matched_error < 1.5, f"Erro do efeito pareado ainda alto ({matched_error:.2f}) — pareamento não corrigiu o suficiente."


def test_propensity_matching_reduces_mean_absolute_smd(confounded_cohort):
    """
    Achado real ao validar isto: a contagem BINÁRIA de Features acima do limiar pode não mudar mesmo
    com uma melhora enorme (SMD caindo de 1.5 para 0.14, ainda > 0.1) — por isso a métrica de
    referência é o |SMD| MÉDIO (contínuo), não a contagem.
    """
    cohort, representation = confounded_cohort
    order = representation.domain_names()

    match_result = match_on_propensity(cohort, "treatment", "treated", order=order, caliper=0.25)

    assert match_result.balance_after is not None
    assert match_result.mean_absolute_smd_after < match_result.mean_absolute_smd_before
    assert match_result.improved_balance is True


def test_check_baseline_balance_detects_the_synthetic_confounding(confounded_cohort):
    cohort, representation = confounded_cohort
    order = representation.domain_names()

    balance = check_baseline_balance(cohort, "treatment", "treated", order=order)
    assert not balance.is_balanced
    assert abs(balance.smd["confounder.z"]) > 1.0  # confundimento forte e conhecido deve aparecer claramente
