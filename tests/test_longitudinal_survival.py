"""
tests.test_longitudinal_survival
====================================

`biospace.longitudinal.SurvivalAnalyzer` (implementação PRÓPRIA de
Kaplan-Meier, sem `lifelines`) existia sem NENHUM teste antes desta
rodada — achado numa auditoria do projeto.

O teste mais valioso aqui: validação CRUZADA contra `lifelines`
(já usado em `biospace.survival`) sobre o MESMO dado — se as duas
implementações independentes concordam, é evidência forte de que
ambas estão corretas, não coincidência.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.longitudinal import SurvivalAnalyzer


class _Obs(Observable):
    key = "x"


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_Obs()])

    def encode(self, measurements):
        v = float(measurements["x"].value)
        return [Feature(name="x", value=v, raw_value=v)]


_REPRESENTATION = Representation([_Dom()])


def _cohort_controlado():
    """3 pacientes: A tem evento (x>=0.5) no dia 10; B tem evento no dia 20; C nunca tem evento (censurado no dia 15, ultima observacao)."""
    cohort = Cohort()

    a = BiologicalSystem(identifier="A")
    for i, x in [(0, 0.1), (10, 0.6)]:
        a.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={"x": x}))
        cohort.update(a, _REPRESENTATION, timestamp=datetime(2020, 1, 1) + timedelta(days=i))

    b = BiologicalSystem(identifier="B")
    for i, x in [(0, 0.1), (10, 0.2), (20, 0.7)]:
        b.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={"x": x}))
        cohort.update(b, _REPRESENTATION, timestamp=datetime(2020, 1, 1) + timedelta(days=i))

    c = BiologicalSystem(identifier="C")
    for i, x in [(0, 0.1), (15, 0.2)]:
        c.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={"x": x}))
        cohort.update(c, _REPRESENTATION, timestamp=datetime(2020, 1, 1) + timedelta(days=i))

    return cohort


def test_records_correctly_identifies_event_time_and_censoring():
    """TESTE DECISIVO: A e B com evento nos dias 10/20 respectivamente; C censurado no dia 15 (ultima observacao, nunca cruzou o limiar)."""
    cohort = _cohort_controlado()
    analyzer = SurvivalAnalyzer.for_feature_threshold("d", "x", threshold=0.5, above=True)
    registros = {r.system_id: r for r in analyzer.records(cohort)}

    assert registros["A"].event_observed is True
    assert registros["A"].time_days == 10.0
    assert registros["B"].event_observed is True
    assert registros["B"].time_days == 20.0
    assert registros["C"].event_observed is False
    assert registros["C"].time_days == 15.0


def test_kaplan_meier_matches_lifelines_on_same_data():
    """
    Validacao cruzada: a implementacao PROPRIA de Kaplan-Meier deveria
    concordar com `lifelines` (biblioteca externa, ja usada em
    biospace.survival) sobre o MESMO dado de duration/event.
    """
    from lifelines import KaplanMeierFitter

    cohort = _cohort_controlado()
    analyzer = SurvivalAnalyzer.for_feature_threshold("d", "x", threshold=0.5, above=True)
    resultado_proprio = analyzer.kaplan_meier(cohort)

    registros = analyzer.records(cohort)
    durations = [r.time_days for r in registros]
    events = [r.event_observed for r in registros]

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)

    for t_dia in [5, 10, 15, 20, 25]:
        s_proprio = resultado_proprio.survival_at(t_dia)
        s_lifelines = float(kmf.survival_function_at_times(t_dia).iloc[0])
        assert abs(s_proprio - s_lifelines) < 1e-9, f"Divergencia em t={t_dia}: proprio={s_proprio:.6f} lifelines={s_lifelines:.6f}"


def test_median_survival_time_matches_known_value():
    """Com 2 eventos (dias 10 e 20) de 3 pacientes, S(t) cai abaixo de 0.5 exatamente no 2o evento (dia 20) -- valor conhecido por construcao."""
    cohort = _cohort_controlado()
    analyzer = SurvivalAnalyzer.for_feature_threshold("d", "x", threshold=0.5, above=True)
    resultado = analyzer.kaplan_meier(cohort)
    assert resultado.median_survival_time() == 20.0


def test_for_phenotype_convenience_constructor():
    from biospace.core import Phenotype

    cohort = _cohort_controlado()
    fenotipo_alto = Phenotype("alto", membership_fn=lambda x: x[0] >= 0.5)
    analyzer = SurvivalAnalyzer.for_phenotype(fenotipo_alto, order=["d"])
    resultado = analyzer.kaplan_meier(cohort)
    assert resultado.n_events_total == 2
    assert resultado.n_censored_total == 1


def test_fit_is_alias_for_kaplan_meier():
    cohort = _cohort_controlado()
    analyzer = SurvivalAnalyzer.for_feature_threshold("d", "x", threshold=0.5, above=True)
    assert analyzer.fit(cohort).time_days == analyzer.kaplan_meier(cohort).time_days
