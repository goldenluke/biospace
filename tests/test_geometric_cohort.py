"""
tests.test_geometric_cohort
===============================

`biospace.geometry.cohort_around`/`GeometricCohort` — a implementação
real do item "coortes automáticas por proximidade geométrica": coortes
como subconjuntos do espaço, não consultas SQL. Testado com dado
fabricado e posições geométricas conhecidas antes de qualquer
aplicação real.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
from biospace.geometry import Euclidean, cohort_around


def _vetor(system_id: str, x: float, y: float) -> RepresentationVector:
    comps = {"d": [Feature(name="x", value=x, raw_value=x), Feature(name="y", value=y, raw_value=y)]}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def _space_em_linha():
    space = RepresentationSpace(domain_order=["d"])
    space.add(_vetor("origem", 0.0, 0.0))
    for i in range(1, 5):
        space.add(_vetor(f"p{i}", float(i), 0.0))
    return space


def test_cohort_around_existing_patient_matches_k_nearest():
    space = _space_em_linha()
    geometria = Euclidean()
    coorte = cohort_around(space, geometria, query="origem", order=["d"], k=2)
    assert coorte.member_ids == ["p1", "p2"]
    assert len(coorte) == 2


def test_cohort_around_arbitrary_vector_not_corresponding_to_any_patient():
    """
    TESTE DECISIVO: consulta por um ponto (1.5, 0) que NAO corresponde
    a nenhum paciente real -- deveria funcionar igual, encontrando
    p1,p2 (distancia 0.5 de cada) dentro de raio=1, excluindo p3,p4
    (distancia 1.5, 2.5).
    """
    space = _space_em_linha()
    geometria = Euclidean()
    coorte = cohort_around(space, geometria, query=np.array([1.5, 0.0]), order=["d"], radius=1.0)
    assert set(coorte.member_ids) == {"p1", "p2"}
    assert "não corresponde a nenhum paciente real" in coorte.query_description


def test_raises_when_both_k_and_radius_given():
    space = _space_em_linha()
    geometria = Euclidean()
    with pytest.raises(ValueError, match="exatamente um"):
        cohort_around(space, geometria, query="origem", order=["d"], k=2, radius=1.0)


def test_raises_when_neither_k_nor_radius_given():
    space = _space_em_linha()
    geometria = Euclidean()
    with pytest.raises(ValueError, match="exatamente um"):
        cohort_around(space, geometria, query="origem", order=["d"])


def test_overlap_with_computes_correct_jaccard_and_asymmetric_fractions():
    """TESTE DECISIVO: coorte geometrica={p1,p2,p3}, coorte externa={p2,p3,p4} -- intersecao={p2,p3} (2), uniao={p1,p2,p3,p4} (4), jaccard=0.5, ambas fracoes = 2/3."""
    from biospace.geometry import GeometricCohort

    coorte = GeometricCohort(member_ids=["p1", "p2", "p3"], query_description="teste")
    resultado = coorte.overlap_with({"p2", "p3", "p4"})

    assert resultado["n_intersecao"] == 2
    assert resultado["jaccard"] == 0.5
    assert abs(resultado["fracao_da_geometrica_tambem_na_outra"] - 2 / 3) < 1e-9
    assert abs(resultado["fracao_da_outra_tambem_na_geometrica"] - 2 / 3) < 1e-9


def test_overlap_with_empty_other_set_does_not_divide_by_zero():
    from biospace.geometry import GeometricCohort

    coorte = GeometricCohort(member_ids=["p1", "p2"], query_description="teste")
    resultado = coorte.overlap_with(set())
    assert resultado["jaccard"] == 0.0
    assert resultado["fracao_da_outra_tambem_na_geometrica"] == 0.0


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_geometric_cohort_around_centroid_diverges_substantially_from_cluster_membership():
    """
    ACHADO REAL, contraintuitivo: uma coorte geometrica definida como
    "k mais proximos do CENTROIDE do fenotipo de alto risco" (kmeans_3,
    achado publicado) NAO recupera a mesma populacao que a associacao
    de cluster original -- overlap de so ~20% (Jaccard~0.11), mesmo
    usando o MESMO tamanho k. Faz sentido matematicamente: K-Means
    particiona por Voronoi entre os 4 centroides SIMULTANEAMENTE
    (mais perto DESTE centroide que de qualquer outro), enquanto
    "k mais proximos de um centroide" so considera distancia a ESSE
    ponto de referencia, ignorando os outros 3. Sao mecanismos de
    definicao de coorte genuinamente diferentes, nao aproximacoes um
    do outro.

    Mesmo assim, a coorte geometrica ainda captura sinal real: taxa de
    readmissao ~1.5x a linha de base populacional (4.51%), mais fraca
    que o cluster completo (~1.9x), mas nao nula -- nao e' so ruido.
    """
    from biospace.geometry import Euclidean, cohort_around
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", include_diagnosis_category=False)
    order = representation.domain_names()
    space = cohort.snapshot()

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space)
    labels = {}
    for sid in space.ids():
        vec = space.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

    ids_kmeans3 = {sid for sid, l in labels.items() if l == "kmeans_3"}
    assert len(ids_kmeans3) > 5000, f"Esperava ~6091 pacientes no fenotipo de alto risco (achado documentado) -- obteve {len(ids_kmeans3)}"

    matriz_kmeans3 = np.stack([space.get(sid).as_vector(order) for sid in ids_kmeans3])
    centroide = matriz_kmeans3.mean(axis=0)

    geom = Euclidean()
    coorte_geometrica = cohort_around(space, geom, query=centroide, order=order, k=len(ids_kmeans3))

    comparacao = coorte_geometrica.overlap_with(ids_kmeans3)
    assert comparacao["jaccard"] < 0.25, f"Esperava divergencia substancial (achado documentado, Jaccard~0.11) -- obteve {comparacao['jaccard']:.3f}"

    def taxa_readmissao(ids):
        return sum(1 for sid in ids if cohort.systems[sid].observations[-1].metadata.get("readmitted") == "<30") / len(ids)

    taxa_geometrica = taxa_readmissao(set(coorte_geometrica.member_ids))
    taxa_geral = sum(1 for s in cohort.systems.values() if s.observations[-1].metadata.get("readmitted") == "<30") / len(cohort.systems)
    assert taxa_geometrica > taxa_geral * 1.2, (
        f"Esperava a coorte geometrica ainda capturar sinal real acima da linha de base (achado documentado: ~1.5x) -- "
        f"geometrica={taxa_geometrica:.4f}, geral={taxa_geral:.4f}"
    )
