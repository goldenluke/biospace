"""
tests.test_phenotyping
=========================

Regressão real: HDBSCAN com `min_cluster_size` alto (o default original,
10) em ~50 dimensões classificava 100% dos pontos como ruído. Corrigido
reduzindo o default para 5. Este teste trava que o default não volte a
degradar totalmente numa população pequena de alta dimensão.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from biospace.core import Cohort
from biospace.phenotyping import ClinicalKMeansPhenotyper, HDBSCANPhenotyper, KMeansPhenotyper
from biospace.plugins.sleep import SleepRepresentation, SleepSystem
from biospace.plugins.sleep.builders import exam


def _build_space(exam_values_factory, n_patients: int = 30):
    representation = SleepRepresentation()
    cohort = Cohort()
    t0 = datetime(2020, 1, 1)
    for i in range(n_patients):
        system = SleepSystem()
        ido = 10.0 + (i % 3) * 15.0  # 3 grupos de severidade
        system.observe(exam(exam_values_factory(ido=ido, ido_sono=ido), timestamp=t0 + timedelta(days=i)))
        cohort.update(system, representation, timestamp=t0 + timedelta(days=i))
    return cohort.snapshot()


def test_hdbscan_default_does_not_classify_everything_as_noise(exam_values_factory):
    space = _build_space(exam_values_factory)
    detector = HDBSCANPhenotyper()  # default atual (min_cluster_size=5)
    phenotypes = detector.fit(space)

    assert detector.n_noise_ < len(space), (
        "HDBSCAN classificou TODOS os pontos como ruído com o parâmetro padrão — "
        "regressão da degradação em alta dimensionalidade já corrigida uma vez."
    )


def test_kmeans_phenotyper_basic_mechanics(exam_values_factory):
    space = _build_space(exam_values_factory)
    phenotypes = KMeansPhenotyper(n_clusters=3).fit(space)
    assert len(phenotypes) == 3


def test_clinical_kmeans_selects_k_via_silhouette(exam_values_factory):
    space = _build_space(exam_values_factory, n_patients=30)
    phenotyper = ClinicalKMeansPhenotyper(k_range=range(2, 6))
    phenotypes = phenotyper.fit(space)

    assert phenotyper.best_k is not None
    assert len(phenotyper.elbow_table) > 0
    assert len(phenotypes) == phenotyper.best_k
