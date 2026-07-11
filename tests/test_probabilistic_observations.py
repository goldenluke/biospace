"""
tests.test_probabilistic_observations
========================================

Testa a propagação de incerteza (Distribution -> Measurement -> Feature)
com verificação matemática EXATA da fórmula de propagação linear, e
confirma retrocompatibilidade total: sem Distribution, uncertainty deve
ser sempre None.
"""

from __future__ import annotations

from datetime import datetime

from biospace.core import Normal
from biospace.plugins.sleep import ApneaDomain, CardiovascularDomain
from biospace.plugins.sleep.builders import exam


def test_measurement_without_distribution_has_zero_uncertainty(sleep_system_factory, exam_values_factory):
    system = sleep_system_factory()
    system.observe(exam(exam_values_factory()))
    m = system.latest_measurement("ido")
    assert m.uncertainty == 0.0
    assert m.distribution is None


def test_measurement_with_distribution_carries_uncertainty(sleep_system_factory, exam_values_factory):
    system = sleep_system_factory()
    values = exam_values_factory(ido=Normal(20.0, 3.0))
    system.observe(exam(values))

    m = system.latest_measurement("ido")
    assert m.value == 20.0  # ponto estimado = mean
    assert m.uncertainty == 3.0
    assert m.distribution is not None


def test_uncertainty_propagates_linearly_through_zscore(sleep_system_factory, exam_values_factory):
    """Verificação EXATA: sigma_final = (sigma_bruto / std_referencia) * peso (transformação linear)."""
    system = sleep_system_factory()
    values = exam_values_factory(ido=Normal(20.0, 3.0))
    system.observe(exam(values))

    domain = ApneaDomain()  # reference default: ido: mean=18, std=15
    features = domain.transform(system)
    feature_ido = next(f for f in features if f.name == "ido")

    std_ref = domain.reference["ido"].std
    weight = domain.feature_weights()["ido"]
    expected_uncertainty = (3.0 / std_ref) * weight

    assert abs(feature_ido.uncertainty - expected_uncertainty) < 1e-9


def test_uncertainty_propagates_through_amplitude_difference(sleep_system_factory, exam_values_factory):
    """CardiovascularDomain.amplitude_fc = fc_max - fc_min: incerteza deve somar em quadratura (sqrt(a^2+b^2))."""
    import numpy as np

    system = sleep_system_factory()
    values = exam_values_factory(fc_maxima_bpm=Normal(100.0, 2.0), fc_minima_bpm=Normal(55.0, 1.5))
    system.observe(exam(values))

    domain = CardiovascularDomain()
    features = domain.transform(system)
    amplitude = next(f for f in features if f.name == "amplitude_fc")

    std_ref = domain.reference["amplitude_fc"].std
    weight = domain.feature_weights()["amplitude_fc"]
    expected_sigma_amplitude = np.sqrt(2.0**2 + 1.5**2)
    expected_uncertainty = (expected_sigma_amplitude / std_ref) * weight

    assert abs(amplitude.uncertainty - expected_uncertainty) < 1e-9


def test_no_distributions_means_all_uncertainties_none(small_cohort):
    """Retrocompatibilidade: numa coorte sem NENHUMA Distribution usada, uncertainty deve ser None em 100% das Features."""
    cohort, representation = small_cohort
    system = next(iter(cohort.systems.values()))

    for domain in representation.domains:
        for feature in domain.transform(system):
            assert feature.uncertainty is None
