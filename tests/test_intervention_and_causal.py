"""
tests.test_intervention_and_causal
=====================================

Duas regressões reais:

  1. `FeatureShiftIntervention` casava por `Feature.name` NÃO qualificado
     — usar o nome qualificado (ex.: "apnea.ido", o padrão usado em
     outras partes do sistema) fazia o shift NÃO SER APLICADO, sem erro
     nem aviso. Corrigido para levantar KeyError claro.
  2. `LatentDomain` deve recusar instanciar sem `hypothesis` declarada
     (evita "índice inventado vestido de teoria").

Mais testes básicos de mecânica para DigitalTwin/Scenario.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from biospace.core import Cohort, Feature, LatentDomain
from biospace.intervention import FeatureShiftIntervention
from biospace.plugins.sleep import SleepRepresentation, SleepSystem
from biospace.plugins.sleep.builders import exam


def _build_cohort_and_evolution(exam_values_factory):
    from biospace.dynamics import MeanRevertingEvolutionOperator

    representation = SleepRepresentation()
    cohort = Cohort()
    system = SleepSystem()
    t0 = datetime(2020, 1, 1)
    for i, ido in enumerate([10, 15, 20, 25, 30]):
        system.observe(exam(exam_values_factory(ido=ido, ido_sono=ido), timestamp=t0 + timedelta(days=i * 30)))
        cohort.update(system, representation, timestamp=t0 + timedelta(days=i * 30))

    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)
    return cohort, representation, evo, order, system.id


def test_feature_shift_with_qualified_name_raises_clear_error(exam_values_factory):
    """Regressão: nome qualificado ('apnea.ido') não deve mais falhar silenciosamente — deve levantar KeyError."""
    cohort, representation, evo, order, sid = _build_cohort_and_evolution(exam_values_factory)
    vector = cohort.trajectories[sid].latest()

    intervention = FeatureShiftIntervention(shifts={"apnea.ido": -0.5})
    with pytest.raises(KeyError):
        intervention.apply(vector)


def test_feature_shift_with_correct_name_applies(exam_values_factory):
    cohort, representation, evo, order, sid = _build_cohort_and_evolution(exam_values_factory)
    vector = cohort.trajectories[sid].latest()

    before = vector.as_vector(order)
    intervention = FeatureShiftIntervention(shifts={"ido": -0.5})
    after_vector = intervention.apply(vector)
    after = after_vector.as_vector(order)

    assert not (before == after).all(), "O shift deveria ter alterado o vetor."


def test_feature_shift_does_not_mutate_original_vector(exam_values_factory):
    """`apply()` deve retornar um NOVO RepresentationVector — nunca mutar o original (a trajetória real não pode ser alterada)."""
    cohort, representation, evo, order, sid = _build_cohort_and_evolution(exam_values_factory)
    vector = cohort.trajectories[sid].latest()
    before = vector.as_vector(order).copy()

    FeatureShiftIntervention(shifts={"ido": -0.5}).apply(vector)

    after = vector.as_vector(order)
    assert (before == after).all(), "O vetor ORIGINAL não deveria ter sido alterado por apply()."


def test_latent_domain_requires_hypothesis():
    class DominioSemHipotese(LatentDomain):
        name = "sem_hipotese"

        def infer(self, source_features):
            return [Feature(name="x", value=0.0)]

    with pytest.raises(ValueError):
        DominioSemHipotese([])


def test_latent_domain_instantiates_with_hypothesis():
    class DominioComHipotese(LatentDomain):
        name = "com_hipotese"
        hypothesis = "Justificativa de teste."

        def infer(self, source_features):
            return [Feature(name="x", value=0.0)]

    domain = DominioComHipotese([])
    assert domain.is_validated is False  # padrão seguro


def test_digital_twin_clone_does_not_affect_original_trajectory(exam_values_factory):
    from biospace.causal import DigitalTwin

    cohort, representation, evo, order, sid = _build_cohort_and_evolution(exam_values_factory)
    traj = cohort.trajectories[sid]
    original_vector_before = traj.latest().as_vector(order).copy()

    twin = DigitalTwin.clone_from(traj, order=order)
    twin.do(FeatureShiftIntervention(shifts={"ido": -5.0}))

    original_vector_after = traj.latest().as_vector(order)
    assert (original_vector_before == original_vector_after).all(), (
        "Aplicar do() no gêmeo NUNCA deveria alterar a trajetória real do paciente original."
    )


def test_digital_twin_simulate_produces_expected_number_of_points(exam_values_factory):
    from biospace.causal import DigitalTwin

    cohort, representation, evo, order, sid = _build_cohort_and_evolution(exam_values_factory)
    twin = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)

    path = twin.simulate(evo, horizon_days=180, step_days=60)
    assert len(path) == 4  # t=0, 60, 120, 180
    assert path[0][0] == 0.0
    assert path[-1][0] == 180.0


def test_scenario_control_arm_always_present(exam_values_factory):
    from biospace.causal import Scenario

    cohort, representation, evo, order, sid = _build_cohort_and_evolution(exam_values_factory)
    scenario = Scenario("teste")
    scenario.add_arm("tratamento", FeatureShiftIntervention(shifts={"ido": -1.0}))

    results = scenario.run(cohort.trajectories[sid], evo, horizon_days=90, step_days=30, order=order)
    assert "controle" in results
    assert "tratamento" in results
