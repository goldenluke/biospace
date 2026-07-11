"""
tests.test_core_disease_agnostic
===================================

Não usa NADA do plugin sleep. Constrói do zero, usando só
`biospace.core` (+ geometria/fenotipagem genéricas, que também não
conhecem doença nenhuma), um domínio de doença TOTALMENTE diferente
(glicemia/diabetes) — para provar EMPIRICAMENTE que o núcleo é genérico
de verdade, não só na teoria (AGENT.md: "o núcleo nunca conhece uma
doença específica").

Se este teste quebrar no futuro porque alguém colocou, sem querer, uma
suposição específica de SAOS dentro de `biospace.core`, é AQUI que isso
vai aparecer — esse é o propósito deste arquivo.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from biospace.core import (
    BiologicalSystem,
    Cohort,
    Feature,
    Measurement,
    Observable,
    Observation,
    Representation,
    SemanticDomain,
)
from biospace.core.contracts import check_reproducibility, check_semantic_preservation
from biospace.geometry import Euclidean


class GlucoseObservable(Observable):
    key = "glicemia_mg_dl"
    unit = "mg/dL"
    description = "Glicemia de jejum"


class HbA1cObservable(Observable):
    key = "hba1c_pct"
    unit = "%"
    description = "Hemoglobina glicada"


class GlycemicDomain(SemanticDomain):
    """
    Domínio de exemplo mínimo — glicemia + HbA1c, z-score contra uma
    referência fixa. Deliberadamente simplificado (não é sobre a lógica
    clínica de diabetes — isso pertenceria a um plugin de verdade; este
    teste é só sobre a ARQUITETURA do núcleo funcionar para uma doença
    que ela nunca viu).
    """

    name = "glycemic"

    def __init__(self):
        super().__init__([GlucoseObservable(), HbA1cObservable()])

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key, ref_mean, ref_std in [("glicemia_mg_dl", 100.0, 20.0), ("hba1c_pct", 5.5, 1.0)]:
            m = measurements.get(key)
            if m is None or m.is_missing():
                features.append(Feature(name=key, value=0.0, is_missing=True))
            else:
                z = (float(m.value) - ref_mean) / ref_std
                features.append(Feature(name=key, value=z, raw_value=float(m.value), z_score=z))
        return features


def _make_patient(identifier: str, glicemia: float, hba1c: float, timestamp: datetime) -> BiologicalSystem:
    """Note: BiologicalSystem USADO DIRETAMENTE — nem precisa de uma subclasse como SleepSystem."""
    system = BiologicalSystem(identifier=identifier)
    system.observe(
        Observation(timestamp=timestamp, source="exame_laboratorial", values={"glicemia_mg_dl": glicemia, "hba1c_pct": hba1c})
    )
    return system


def test_core_supports_a_completely_new_disease_without_any_sleep_dependency():
    """Constrói uma coorte de 'diabetes' do zero, usando só biospace.core + geometria genérica."""
    representation = Representation([GlycemicDomain()])
    cohort = Cohort()

    pacientes = [
        ("normal_1", 90.0, 5.2),
        ("normal_2", 95.0, 5.4),
        ("normal_3", 88.0, 5.1),
        ("diabetico_1", 180.0, 8.5),
        ("diabetico_2", 190.0, 9.0),
        ("diabetico_3", 175.0, 8.2),
    ]
    t0 = datetime(2023, 1, 1)
    for pid, glicemia, hba1c in pacientes:
        cohort.update(_make_patient(pid, glicemia, hba1c, t0), representation, timestamp=t0)

    assert len(cohort) == 6

    space = cohort.snapshot()
    matrix, ids = space.matrix()
    assert matrix.shape == (6, 2)  # 6 "pacientes", 2 Features (glicemia + hba1c) — nenhuma suposição de SAOS aqui

    euclid = Euclidean()
    order = space.order()
    d_entre_grupos = euclid.distance(space.get("normal_1").as_vector(order), space.get("diabetico_1").as_vector(order))
    d_mesmo_grupo = euclid.distance(space.get("normal_1").as_vector(order), space.get("normal_2").as_vector(order))
    assert d_entre_grupos > d_mesmo_grupo, "Sistemas de grupos fisiologicamente diferentes devem estar mais distantes que os do mesmo grupo."


def test_core_contracts_work_on_a_disease_core_has_never_seen():
    """Reprodutibilidade e Preservação Semântica devem valer para QUALQUER domínio, não só os do plugin sleep."""
    domain = GlycemicDomain()
    normal = _make_patient("n1", 90.0, 5.2, datetime(2023, 1, 1))
    diabetico = _make_patient("d1", 190.0, 9.0, datetime(2023, 1, 1))

    assert check_reproducibility(domain, normal) is True
    assert check_semantic_preservation(domain, normal, diabetico) is True


def test_core_phenotyping_works_on_a_disease_core_has_never_seen():
    """KMeansPhenotyper (genérico, em biospace.phenotyping) deve separar os 2 grupos sem NENHUM código específico de doença."""
    from biospace.phenotyping import KMeansPhenotyper

    representation = Representation([GlycemicDomain()])
    cohort = Cohort()
    t0 = datetime(2023, 1, 1)
    rng = np.random.default_rng(0)

    for i in range(15):
        cohort.update(
            _make_patient(f"normal_{i}", float(rng.normal(90, 5)), float(rng.normal(5.3, 0.3)), t0),
            representation, timestamp=t0,
        )
    for i in range(15):
        cohort.update(
            _make_patient(f"diabetico_{i}", float(rng.normal(190, 10)), float(rng.normal(9.0, 0.5)), t0),
            representation, timestamp=t0,
        )

    space = cohort.snapshot()
    phenotypes = KMeansPhenotyper(n_clusters=2).fit(space)
    assert len(phenotypes) == 2

    order = space.order()
    labels = {
        sid: next((ph.name for ph in phenotypes if ph.contains(space.get(sid).as_vector(order))), None)
        for sid in space.ids()
    }

    normal_labels = {v for k, v in labels.items() if k.startswith("normal_")}
    diabetico_labels = {v for k, v in labels.items() if k.startswith("diabetico_")}
    assert len(normal_labels) == 1, "Todos os 'normais' deveriam cair no MESMO cluster entre si."
    assert len(diabetico_labels) == 1, "Todos os 'diabéticos' deveriam cair no MESMO cluster entre si."
    assert normal_labels != diabetico_labels, "Os dois grupos deveriam cair em clusters DIFERENTES."


def test_representation_extend_composes_across_unrelated_domains():
    """Contrato 5.5 (Extensibilidade) também deve valer fora do plugin sleep — estender com um 2º domínio não relacionado."""

    class BloodPressureObservable(Observable):
        key = "pressao_sistolica_mmhg"
        unit = "mmHg"

    class BloodPressureDomain(SemanticDomain):
        name = "blood_pressure"

        def __init__(self):
            super().__init__([BloodPressureObservable()])

        def encode(self, measurements):
            m = measurements.get("pressao_sistolica_mmhg")
            if m is None or m.is_missing():
                return [Feature(name="pressao_sistolica_mmhg", value=0.0, is_missing=True)]
            z = (float(m.value) - 120.0) / 15.0
            return [Feature(name="pressao_sistolica_mmhg", value=z, raw_value=float(m.value))]

    representation = Representation([GlycemicDomain()])
    system = _make_patient("p1", 100.0, 5.5, datetime(2023, 1, 1))
    system.observe(
        Observation(timestamp=datetime(2023, 1, 1), source="exame_laboratorial", values={"pressao_sistolica_mmhg": 130.0})
    )

    extended = representation.extend(BloodPressureDomain())
    vector = extended.transform(system)

    assert set(vector.components.keys()) == {"glycemic", "blood_pressure"}
    assert len(vector.as_vector(["glycemic", "blood_pressure"])) == 3  # 2 (glicemia+hba1c) + 1 (pressão)
