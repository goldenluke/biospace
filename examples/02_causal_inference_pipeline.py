"""
examples/02_causal_inference_pipeline.py
===========================================

Pipeline completo de inferência causal OBSERVACIONAL: balanceamento de
linha de base -> escore de propensão -> pareamento -> efeito pareado
(diferença-em-diferenças) — sobre um cenário SINTÉTICO com confundimento
por indicação FORTE e efeito causal VERDADEIRO conhecido, para que dê
pra comparar a estimativa contra a verdade (impossível com dados reais).

IMPORTANTE: nada aqui é uma prova causal formal (sentido de Pearl) — ver
`biospace/causal/do_operator.py` e o README para o aviso completo.

Rode com: python3 examples/02_causal_inference_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

import numpy as np

from biospace.causal import ObservationalEffectEstimator, check_baseline_balance, estimate_matched_effect, match_on_propensity
from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain

TRUE_EFFECT = -5.0     # efeito causal VERDADEIRO do tratamento sobre o desfecho (conhecido de propósito)
CONFOUND_BETA = 4.0    # confundimento: o confundidor Z também afeta o desfecho, independente do tratamento


class ZObservable(Observable):
    key = "z"


class OutcomeObservable(Observable):
    key = "outcome"


class TreatedObservable(Observable):
    key = "treated"


class ConfounderDomain(SemanticDomain):
    name = "confounder"

    def __init__(self):
        super().__init__([ZObservable()])

    def encode(self, measurements):
        m = measurements.get("z")
        return [Feature(name="z", value=float(m.value) if m else 0.0)]


class OutcomeDomain(SemanticDomain):
    name = "outcome_domain"

    def __init__(self):
        super().__init__([OutcomeObservable()])

    def encode(self, measurements):
        m = measurements.get("outcome")
        return [Feature(name="outcome", value=float(m.value) if m else 0.0)]


class TreatmentDomain(SemanticDomain):
    name = "treatment"

    def __init__(self):
        super().__init__([TreatedObservable()])

    def encode(self, measurements):
        m = measurements.get("treated")
        return [Feature(name="treated", value=float(m.value) if m else 0.0)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def build_confounded_cohort(n: int = 200, seed: int = 42):
    """N pacientes; Z afeta tanto P(tratamento) quanto o desfecho -- confundimento por indicação forte."""
    rng = np.random.default_rng(seed)
    representation = Representation([ConfounderDomain(), OutcomeDomain(), TreatmentDomain()])
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)

    for i in range(n):
        z = rng.normal(0, 1)
        treated = 1.0 if rng.random() < sigmoid(2.0 * z) else 0.0
        baseline_outcome = rng.normal(50, 5)

        system = BiologicalSystem(identifier=f"p{i}")
        system.observe(Observation(timestamp=t0, source="baseline", values={"z": z, "outcome": baseline_outcome, "treated": 0.0}))
        cohort.update(system, representation, timestamp=t0)

        t1 = t0 + timedelta(days=90)
        follow_outcome = baseline_outcome + TRUE_EFFECT * treated + CONFOUND_BETA * z + rng.normal(0, 2)
        system.observe(Observation(timestamp=t1, source="follow_up", values={"z": z, "outcome": follow_outcome, "treated": treated}))
        cohort.update(system, representation, timestamp=t1)

    return cohort, representation


def main():
    print(f"=== Efeito causal VERDADEIRO (conhecido, só possível em dado sintético): {TRUE_EFFECT} ===\n")
    cohort, representation = build_confounded_cohort()
    order = representation.domain_names()

    print("--- Passo 1: balanceamento de linha de base ---")
    balance = check_baseline_balance(cohort, "treatment", "treated", order=order)
    print(balance.summary(), "\n")

    print("--- Passo 2: efeito ingênuo (antes/depois, sem ajuste) ---")
    naive = ObservationalEffectEstimator("treatment", "treated", order=order).estimate(cohort, run_balance_check=False)
    naive_effect = naive.delta_mean["outcome_domain.outcome"]
    print(f"  outcome: {naive_effect:+.3f}  (verdadeiro: {TRUE_EFFECT}, erro: {abs(naive_effect - TRUE_EFFECT):.3f})\n")

    print("--- Passo 3: escore de propensão + pareamento ---")
    match_result = match_on_propensity(cohort, "treatment", "treated", order=order, caliper=0.25)
    print(match_result.summary(), "\n")

    print("--- Passo 4: efeito pareado (diferença-em-diferenças) ---")
    matched = estimate_matched_effect(cohort, match_result, order=order)
    matched_effect = matched.effect["outcome_domain.outcome"]
    print(f"  outcome: {matched_effect:+.3f}  (verdadeiro: {TRUE_EFFECT}, erro: {abs(matched_effect - TRUE_EFFECT):.3f})\n")

    print("=== Conclusão ===")
    print(f"Erro do efeito ingênuo:  {abs(naive_effect - TRUE_EFFECT):.3f}")
    print(f"Erro do efeito pareado:  {abs(matched_effect - TRUE_EFFECT):.3f}  (deveria ser menor)")


if __name__ == "__main__":
    main()
