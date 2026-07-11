"""
tests.test_core_contracts
============================

Um teste por contrato formal (Seção 5 da teoria + Estabilidade
Fenotípica, Seção 8.5) — cada um usando o cenário mínimo necessário para
exercitar a propriedade, não apenas confirmar "não deu erro".
"""

from __future__ import annotations

import numpy as np
import pytest

from biospace.core import Feature, Representation, SemanticDomain
from biospace.core.contracts import (
    check_extensibility,
    check_injectivity,
    check_lipschitz_continuity,
    check_reproducibility,
    check_representation_compatibility,
    check_semantic_preservation,
)
from biospace.plugins.sleep import ApneaDomain, fit_reference
from biospace.plugins.sleep.builders import exam


class _ConstantDomain(SemanticDomain):
    """Domínio trivial (sempre a mesma Feature) — usado só para testar Extensibilidade sem colidir de nome com domínios reais."""

    name = "dominio_de_teste"

    def __init__(self):
        super().__init__([])

    def encode(self, measurements):
        return [Feature(name="constante", value=1.0)]


def test_reproducibility(sleep_system_factory, exam_values_factory):
    system = sleep_system_factory()
    system.observe(exam(exam_values_factory()))
    assert check_reproducibility(ApneaDomain(), system) is True


def test_semantic_preservation_detects_different_states(sleep_system_factory, exam_values_factory):
    system_a = sleep_system_factory()
    system_a.observe(exam(exam_values_factory(ido=10.0)))
    system_b = sleep_system_factory()
    system_b.observe(exam(exam_values_factory(ido=45.0)))

    assert check_semantic_preservation(ApneaDomain(), system_a, system_b) is True


def test_semantic_preservation_identical_states_trivially_ok(sleep_system_factory, exam_values_factory):
    """Dois sistemas com os MESMOS valores não têm nada a preservar — o contrato deve retornar True (nada a violar)."""
    vals = exam_values_factory(ido=20.0)
    system_a = sleep_system_factory()
    system_a.observe(exam(dict(vals)))
    system_b = sleep_system_factory()
    system_b.observe(exam(dict(vals)))

    assert check_semantic_preservation(ApneaDomain(), system_a, system_b) is True


def test_lipschitz_continuity_is_finite(sleep_system_factory, exam_values_factory):
    systems = []
    for ido in [10.0, 15.0, 20.0, 25.0, 30.0]:
        s = sleep_system_factory()
        s.observe(exam(exam_values_factory(ido=ido)))
        systems.append(s)

    pairs = [(systems[i], systems[i + 1]) for i in range(len(systems) - 1)]

    def raw_distance(a, b):
        return abs(a.latest_values()["ido"] - b.latest_values()["ido"])

    L = check_lipschitz_continuity(ApneaDomain(), pairs, raw_distance)
    assert np.isfinite(L), "Constante de Lipschitz deveria ser finita para uma sequência suave de estados."


def test_extensibility_preserves_old_components(sleep_representation, sleep_system_factory, exam_values_factory):
    system = sleep_system_factory()
    system.observe(exam(exam_values_factory()))
    assert check_extensibility(sleep_representation, _ConstantDomain(), system) is True


def test_extensibility_catches_name_collision(sleep_representation, sleep_system_factory, exam_values_factory):
    """
    Achado real do projeto: usar um novo domínio com o MESMO NOME de um já
    existente é um erro de modelagem. Depois de endurecer o núcleo
    (`Representation.__init__` agora valida nomes únicos), isso passou a
    ser pego MAIS CEDO — `extend()` levanta ValueError imediatamente, em
    vez de silenciosamente construir uma Representation quebrada para
    `check_extensibility()` detectar via comparação de valores depois.
    """
    system = sleep_system_factory()
    system.observe(exam(exam_values_factory()))

    referencia_diferente = fit_reference([exam_values_factory(ido=v) for v in [100.0, 200.0, 300.0]])
    domain_com_nome_colidindo = ApneaDomain(reference=referencia_diferente)  # mesmo nome "apnea"

    with pytest.raises(ValueError):
        check_extensibility(sleep_representation, domain_com_nome_colidindo, system)


def test_representation_rejects_duplicate_domain_names(sleep_system_factory, exam_values_factory):
    """Regressão direta: Representation não deve aceitar dois domínios com o mesmo `.name` — nunca, não só via extend()."""
    from biospace.core import Representation

    with pytest.raises(ValueError):
        Representation([ApneaDomain(), ApneaDomain()])


def test_injectivity_no_collisions_among_distinct_systems(sleep_system_factory, exam_values_factory):
    systems = []
    for ido in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
        s = sleep_system_factory()
        s.observe(exam(exam_values_factory(ido=ido)))
        systems.append(s)

    domain = ApneaDomain()
    report = check_injectivity(
        transform=lambda s: np.array([f.value for f in domain.transform(s)]),
        systems=systems,
    )
    assert report.is_injective
    assert report.n_pairs_checked == 28  # C(8,2)


def test_injectivity_detects_forced_collision(sleep_system_factory, exam_values_factory):
    """Dois sistemas com valores quase idênticos (dentro da tolerância) devem ser reportados como colisão."""
    s_a = sleep_system_factory()
    s_a.observe(exam(exam_values_factory(ido=20.0)))
    s_b = sleep_system_factory()
    s_b.observe(exam(exam_values_factory(ido=20.0 + 1e-12)))

    domain = ApneaDomain()
    report = check_injectivity(
        transform=lambda s: np.array([f.value for f in domain.transform(s)]),
        systems=[s_a, s_b],
    )
    assert not report.is_injective
    assert len(report.violations) == 1


def test_representation_compatibility_schema_stable_across_reference_refits(sleep_system_factory, exam_values_factory):
    """Reajustar a Reference (ex.: sobre uma nova população) não deve mudar QUAIS Features existem, só seus valores."""
    system = sleep_system_factory()
    system.observe(exam(exam_values_factory()))

    ref_a = fit_reference([exam_values_factory(ido=v) for v in [10, 20, 30]])
    ref_b = fit_reference([exam_values_factory(ido=v) for v in [15, 45, 60]])

    domain_v1 = ApneaDomain(reference=ref_a)
    domain_v2 = ApneaDomain(reference=ref_b)

    report = check_representation_compatibility(domain_v1, domain_v2, system)
    assert report.is_compatible
    assert report.only_in_v1 == []
    assert report.only_in_v2 == []


def test_phenotype_stability_report_shape(small_cohort):
    """Só testa a MECÂNICA (roda sem erro, produz um ARI em [-1,1]) — não a interpretação clínica."""
    from biospace.phenotyping import KMeansPhenotyper
    from biospace.phenotyping.contracts import check_phenotype_stability

    cohort, representation = small_cohort
    space = cohort.snapshot()

    report = check_phenotype_stability(
        operator_factory=lambda: KMeansPhenotyper(n_clusters=2),
        space=space,
        seed=0,
    )
    assert -1.0 <= report.adjusted_rand_index <= 1.0
    assert report.n_common_points == len(space)


def test_run_contract_suite_skips_missing_inputs_gracefully(small_cohort, sleep_system_factory, exam_values_factory):
    """Sem NENHUM dado extra fornecido, a suíte não deve falhar — todos os contratos ficam em `skipped`."""
    from biospace.ontology import run_contract_suite

    cohort, representation = small_cohort
    report = run_contract_suite(representation=representation)

    assert report.reproducibility is None
    assert len(report.skipped) == 11  # todos os 11 contratos pulados (nenhum dado extra foi passado)


def test_run_contract_suite_runs_all_eleven_with_full_inputs(small_cohort, sleep_system_factory):
    """Com todos os dados fornecidos, os 11 contratos devem rodar — regressão da suíte consolidada (Seção README)."""
    from biospace.core import CompositeRepresentation, Representation
    from biospace.geometry import Euclidean
    from biospace.ontology import run_contract_suite
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = small_cohort
    space = cohort.snapshot()
    systems = list(cohort.systems.values())
    apnea_domain = next(d for d in representation.domains if d.name == "apnea")

    def raw_distance(a, b):
        va, vb = a.latest_values(), b.latest_values()
        return abs(va.get("ido", 0) - vb.get("ido", 0))

    pairs = [(systems[i], systems[i + 1]) for i in range(min(3, len(systems) - 1))]
    rep_v2 = Representation([apnea_domain])
    composite = CompositeRepresentation("grupo_teste", [apnea_domain])
    vetor = cohort.trajectories[systems[0].id].latest()

    def aplicar_euclidiana(geometria, esp):
        ids = esp.ids()
        order = esp.order()
        a = esp.get(ids[0]).as_vector(order)
        b = esp.get(ids[1]).as_vector(order) if len(ids) > 1 else a
        return geometria.distance(a, b)

    report = run_contract_suite(
        representation=representation,
        domain=apnea_domain,
        systems=systems,
        pairs_for_continuity=pairs,
        raw_distance_fn=raw_distance,
        new_domain_for_extensibility=_ConstantDomain(),  # nome NOVO -- não colide com "apnea" já existente
        representation_v2=rep_v2,
        space=space,
        phenotyping_operator_factory=lambda: KMeansPhenotyper(n_clusters=2),
        system_factory=sleep_system_factory,
        observations_for_temporality=systems[0].observations,
        composite_for_compositionality=composite,
        algorithm_factory=Euclidean,
        algorithm_apply_fn=aplicar_euclidiana,
        space_b_for_independence=space,
        vector_for_interoperability=vetor,
    )

    assert report.skipped == []
    assert report.reproducibility is True
    assert report.extensibility is True  # confirma que o nome novo NÃO colide (ver docstring de _ConstantDomain)
    assert report.temporality is not None
    assert report.phenotype_stability is not None
    assert report.compositionality is not None
    assert report.algorithmic_independence is not None
    assert report.interoperability is not None
