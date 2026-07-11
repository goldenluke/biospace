"""
tests.test_temporality
=========================

Regressão do bug mais sutil e importante encontrado no projeto:
observações fora de ordem cronológica (ex.: preenchimento retroativo de
histórico) faziam uma observação FUTURA contaminar um ponto da
trajetória que deveria refletir só o passado — porque
`Representation.transform()` usava `timestamp` só como rótulo, nunca
como corte real sobre quais observações considerar.

Corrigido propagando `as_of` por toda a cadeia
(BiologicalSystem -> Observable -> SemanticDomain -> Representation).
Estes testes travam essa correção como contrato permanente.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from biospace.core.contracts import check_temporality
from biospace.plugins.sleep import exam


def _make_obs(exam_values_factory, ido: float, dias: int):
    return exam(exam_values_factory(ido=ido, ido_sono=ido), timestamp=datetime(2020, 1, 1) + timedelta(days=dias))


def test_temporality_compliant_with_in_order_observations(sleep_representation, sleep_system_factory, exam_values_factory):
    observations = [
        _make_obs(exam_values_factory, 10, 0),
        _make_obs(exam_values_factory, 20, 30),
        _make_obs(exam_values_factory, 30, 90),
    ]
    report = check_temporality(sleep_representation, sleep_system_factory, observations)
    assert report.is_compliant


def test_temporality_compliant_with_out_of_order_observations(sleep_representation, sleep_system_factory, exam_values_factory):
    """
    O caso que estava genuinamente quebrado: observações fora de ordem
    cronológica não podem mais contaminar pontos anteriores da trajetória.
    """
    observations = [
        _make_obs(exam_values_factory, 30, 90),
        _make_obs(exam_values_factory, 10, 0),
        _make_obs(exam_values_factory, 20, 30),
    ]  # fora de ordem de propósito
    report = check_temporality(sleep_representation, sleep_system_factory, observations)

    assert report.is_compliant
    assert report.no_lookahead, "Regressão: uma observação futura está contaminando um ponto anterior da trajetória."
    assert report.correct_length
    assert report.single_system
    assert report.is_chronologically_sorted
    assert report.timestamps_match_observations


def test_as_of_cutoff_excludes_future_observations(sleep_representation, sleep_system_factory, exam_values_factory):
    """Teste direto e mínimo do mecanismo `as_of`, sem passar pelo check_temporality."""
    system = sleep_system_factory()
    system.observe(_make_obs(exam_values_factory, 10, 0))
    system.observe(_make_obs(exam_values_factory, 999, 100))  # observação "futura" com valor bem diferente

    # As-of o instante da PRIMEIRA observação: o valor "futuro" (999) não deve aparecer
    m = system.latest_measurement("ido", as_of=datetime(2020, 1, 1))
    assert m is not None
    assert m.value == 10, f"Esperava 10 (valor conhecido em t=0), veio {m.value} — a observação futura vazou."

    # Sem as_of (comportamento padrão): deve ver o valor mais recente normalmente
    m_sem_corte = system.latest_measurement("ido")
    assert m_sem_corte.value == 999
