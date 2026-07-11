"""
tests.test_derived_variable
===============================

DerivedVariable (biospace.core.derived_variable): camada Observação ->
Trajetória -> Variável Derivada, distinta de Feature (que vem de UM
instante). Testa: (1) valores corretos contra trajetória com resultado
CONHECIDO calculado à mão antes de confiar em qualquer implementação;
(2) casos de borda (trajetória curta demais -> None, nunca um valor
inventado); (3) `augment_with_derived_variables` nunca modifica o
vetor original in-place; (4) retrocompatibilidade — nenhuma variável
derivada é obrigatória, representação pontual funciona sem elas.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.core import Cohort, Trajectory, augment_with_derived_variables
from biospace.plugins.metabolic import (
    GlycemicBurdenVariable,
    HbA1cSlopeVariable,
    HbA1cVariabilityVariable,
    MetabolicRepresentation,
    MetabolicSystem,
    exam,
)


def _build_trajectory(valores: list[float], intervalo_dias: int = 100):
    representation = MetabolicRepresentation()
    cohort = Cohort()
    system = MetabolicSystem()
    t0 = datetime(2020, 1, 1)
    for i, v in enumerate(valores):
        ts = t0 + timedelta(days=intervalo_dias * i)
        system.observe(exam({"hba1c_pct": v, "glicemia_jejum_mg_dl": 100.0}, timestamp=ts))
        cohort.update(system, representation, timestamp=ts)
    return cohort.trajectories[system.id]


def test_slope_matches_hand_calculated_value():
    """TESTE DECISIVO: trajetoria com progressao linear conhecida -- 0.5 a cada 100 dias -- slope esperado calculado a mao: 0.5/100*365.25=1.82625/ano."""
    trajetoria = _build_trajectory([6.0, 6.5, 7.0, 7.5, 8.0])
    feature = HbA1cSlopeVariable().compute(trajetoria)
    assert feature is not None
    assert feature.value == pytest.approx(1.82625, abs=1e-3)


def test_variability_matches_hand_calculated_value():
    """std([6.0,6.5,7.0,7.5,8.0], ddof=1) calculado a mao: media=7.0, soma_quadrados=2.5, /4=0.625, sqrt=0.7906."""
    trajetoria = _build_trajectory([6.0, 6.5, 7.0, 7.5, 8.0])
    feature = HbA1cVariabilityVariable().compute(trajetoria)
    assert feature is not None
    assert feature.value == pytest.approx(0.79057, abs=1e-3)


def test_glycemic_burden_matches_hand_calculated_value():
    """Excesso sobre 7.0: (0)+(0)+(0)+(0.5)+(1.0) = 1.5 -- mesmo mecanismo ja validado em synthetic.py."""
    trajetoria = _build_trajectory([6.0, 6.5, 7.0, 7.5, 8.0])
    feature = GlycemicBurdenVariable().compute(trajetoria)
    assert feature is not None
    assert feature.value == pytest.approx(1.5, abs=1e-6)


def test_flat_trajectory_has_zero_slope_and_zero_variability():
    """Contraprova: trajetoria SEM tendencia nenhuma (todos os valores iguais) deve dar slope=0 e variabilidade=0, nao ruido."""
    trajetoria = _build_trajectory([7.0, 7.0, 7.0, 7.0])
    slope = HbA1cSlopeVariable().compute(trajetoria)
    variabilidade = HbA1cVariabilityVariable().compute(trajetoria)
    assert slope.value == pytest.approx(0.0, abs=1e-9)
    assert variabilidade.value == pytest.approx(0.0, abs=1e-9)


def test_single_point_trajectory_returns_none_for_slope_and_variability():
    """Slope e variabilidade precisam de >=2 pontos -- 1 ponto deve devolver None, nunca um valor inventado."""
    trajetoria = _build_trajectory([7.0])
    assert HbA1cSlopeVariable().compute(trajetoria) is None
    assert HbA1cVariabilityVariable().compute(trajetoria) is None


def test_single_point_trajectory_still_computes_burden():
    """Burden faz sentido ate' com 1 ponto (min_points=1) -- e' uma soma, nao uma tendencia."""
    trajetoria = _build_trajectory([8.0])
    feature = GlycemicBurdenVariable().compute(trajetoria)
    assert feature is not None
    assert feature.value == pytest.approx(1.0, abs=1e-9)


def test_empty_trajectory_returns_none_for_all_variables():
    system = MetabolicSystem()
    trajetoria_vazia = Trajectory(system_id=system.id)
    assert HbA1cSlopeVariable().compute(trajetoria_vazia) is None
    assert HbA1cVariabilityVariable().compute(trajetoria_vazia) is None
    assert GlycemicBurdenVariable().compute(trajetoria_vazia) is None


def test_augment_with_derived_variables_does_not_mutate_original_vector():
    trajetoria = _build_trajectory([6.0, 6.5, 7.0, 7.5, 8.0])
    vetor_original = trajetoria.latest()
    componentes_originais_antes = set(vetor_original.components.keys())

    dv_list = [HbA1cSlopeVariable(), HbA1cVariabilityVariable(), GlycemicBurdenVariable()]
    vetor_aumentado = augment_with_derived_variables(vetor_original, trajetoria, dv_list)

    assert set(vetor_original.components.keys()) == componentes_originais_antes, "augment_with_derived_variables nao deveria alterar o vetor original in-place."
    assert "derived" not in vetor_original.components
    assert "derived" in vetor_aumentado.components
    assert len(vetor_aumentado.components["derived"]) == 3


def test_derived_variables_declare_the_same_process_as_their_source_feature():
    """As 3 variaveis derivadas herdam o processo fisiologico da Feature que resumem -- prova de que DerivedVariable e PhysiologicalProcess compoem corretamente."""
    for dv in [HbA1cSlopeVariable(), HbA1cVariabilityVariable(), GlycemicBurdenVariable()]:
        assert dv.process == "glucose_homeostasis"


def test_point_representation_works_identically_without_any_derived_variable():
    """Retrocompatibilidade: a representacao pontual (sem augment_with_derived_variables) continua identica -- a camada e' aditiva, nunca obrigatoria."""
    trajetoria = _build_trajectory([6.0, 6.5, 7.0])
    vetor = trajetoria.latest()
    assert "derived" not in vetor.components
    assert set(vetor.components.keys()) == {"glycemic", "anthropometric", "cardiovascular", "renal", "comorbidity", "treatment"}
