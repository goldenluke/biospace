"""
tests.test_transition_analyzer
==================================

`biospace.longitudinal.TransitionAnalyzer` existia sem NENHUM teste
antes desta rodada — achado numa auditoria do projeto.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Phenotype, Representation, SemanticDomain
from biospace.longitudinal import TransitionAnalyzer


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
_ORDER = ["d"]


def _fenotipos():
    return [
        Phenotype("baixo", membership_fn=lambda x: x[0] < 0.5),
        Phenotype("alto", membership_fn=lambda x: x[0] >= 0.5),
    ]


def _build_cohort_com_transicao_conhecida():
    """Paciente 'sobe': comeca baixo (x=0.1), sobe pra alto (x=0.9) 10 dias depois -- transicao baixo->alto conhecida, com gap=10 dias."""
    representation = _REPRESENTATION
    cohort = Cohort()

    system = BiologicalSystem(identifier="sobe")
    system.observe(Observation(timestamp=datetime(2020, 1, 1), source="t", values={"x": 0.1}))
    cohort.update(system, representation, timestamp=datetime(2020, 1, 1))
    system.observe(Observation(timestamp=datetime(2020, 1, 11), source="t", values={"x": 0.9}))
    cohort.update(system, representation, timestamp=datetime(2020, 1, 11))

    # Paciente estavel: fica baixo o tempo todo
    system2 = BiologicalSystem(identifier="estavel")
    system2.observe(Observation(timestamp=datetime(2020, 1, 1), source="t", values={"x": 0.1}))
    cohort.update(system2, representation, timestamp=datetime(2020, 1, 1))
    system2.observe(Observation(timestamp=datetime(2020, 1, 11), source="t", values={"x": 0.2}))
    cohort.update(system2, representation, timestamp=datetime(2020, 1, 11))

    return cohort


def test_matrix_correctly_counts_known_transition():
    """TESTE DECISIVO: com verdade conhecida (1 paciente baixo->alto, 1 paciente baixo->baixo), P(alto|baixo) deveria ser exatamente 0.5."""
    cohort = _build_cohort_com_transicao_conhecida()
    analyzer = TransitionAnalyzer(_fenotipos(), order=_ORDER)
    matriz, nomes = analyzer.matrix(cohort)

    idx_baixo = nomes.index("baixo")
    idx_alto = nomes.index("alto")
    assert matriz[idx_baixo, idx_alto] == 0.5, f"Esperava P(alto|baixo)=0.5 (1 de 2 transicoes de 'baixo') -- obteve {matriz[idx_baixo, idx_alto]}"
    assert matriz[idx_baixo, idx_baixo] == 0.5


def test_time_to_transition_reports_correct_gap():
    """O gap de tempo da transicao baixo->alto deveria ser exatamente 10 dias (conhecido por construcao)."""
    cohort = _build_cohort_com_transicao_conhecida()
    analyzer = TransitionAnalyzer(_fenotipos(), order=_ORDER)
    tempos = analyzer.time_to_transition(cohort)

    gaps_baixo_alto = tempos.get(("baixo", "alto"), [])
    assert len(gaps_baixo_alto) == 1
    assert gaps_baixo_alto[0] == timedelta(days=10)


def test_max_gap_filter_excludes_transitions_outside_window():
    """Filtrar max_gap=5 dias deveria EXCLUIR a transicao de 10 dias -- matriz deveria ficar sem essa contagem."""
    cohort = _build_cohort_com_transicao_conhecida()
    analyzer = TransitionAnalyzer(_fenotipos(), order=_ORDER)
    tempos_filtrados = analyzer.time_to_transition(cohort, max_gap=timedelta(days=5))
    assert ("baixo", "alto") not in tempos_filtrados or len(tempos_filtrados[("baixo", "alto")]) == 0


def test_min_gap_filter_excludes_transitions_too_close():
    """Filtrar min_gap=20 dias tambem deveria excluir a transicao de 10 dias (gap menor que o minimo exigido)."""
    cohort = _build_cohort_com_transicao_conhecida()
    analyzer = TransitionAnalyzer(_fenotipos(), order=_ORDER)
    tempos_filtrados = analyzer.time_to_transition(cohort, min_gap=timedelta(days=20))
    assert ("baixo", "alto") not in tempos_filtrados or len(tempos_filtrados[("baixo", "alto")]) == 0


def test_summary_computes_correct_mean_and_median():
    cohort = _build_cohort_com_transicao_conhecida()
    analyzer = TransitionAnalyzer(_fenotipos(), order=_ORDER)
    resumo = analyzer.summary(cohort)

    stats_baixo_alto = resumo[("baixo", "alto")]
    assert stats_baixo_alto["n"] == 1
    assert stats_baixo_alto["media_dias"] == 10.0
    assert stats_baixo_alto["mediana_dias"] == 10.0


def test_fit_is_alias_for_matrix_without_time_filter():
    cohort = _build_cohort_com_transicao_conhecida()
    analyzer = TransitionAnalyzer(_fenotipos(), order=_ORDER)
    matriz_fit, nomes_fit = analyzer.fit(cohort)
    matriz_direta, nomes_direta = analyzer.matrix(cohort)
    assert np.array_equal(matriz_fit, matriz_direta)
    assert nomes_fit == nomes_direta
