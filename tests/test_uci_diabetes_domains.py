"""
tests.test_uci_diabetes_domains
===================================

Testes unitários da lógica de codificação dos domínios UCI (encode(),
mapeamento ordinal) com dado FABRICADO — rodam em CI, não dependem do
arquivo real de 101.766 linhas (esse fica em test_uci_diabetes_real_data.py,
skip-if-absent).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from biospace.core import Measurement, Observation
from biospace.datasets.uci_diabetes import (
    GlycemicTestingDomain,
    HospitalUtilizationDomain,
    UCIHospitalRepresentation,
    UCIHospitalSystem,
    _row_to_values,
)


def test_row_to_values_maps_a1c_ordinal_correctly():
    row = pd.Series({
        "time_in_hospital": 3, "num_lab_procedures": 40, "num_procedures": 1, "num_medications": 12,
        "number_diagnoses": 7, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
        "A1Cresult": ">8", "max_glu_serum": None, "insulin": "Steady", "change": "Ch", "diabetesMed": "Yes",
    })
    valores = _row_to_values(row)
    assert valores["A1Cresult_ordinal"] == 2.0
    assert "max_glu_serum_ordinal" not in valores
    assert valores["insulin_ordinal"] == 1.0
    assert valores["change_flag"] == 1.0
    assert valores["diabetes_med_flag"] == 1.0


def test_row_to_values_handles_all_missing_glycemic():
    row = pd.Series({
        "time_in_hospital": 1, "num_lab_procedures": 10, "num_procedures": 0, "num_medications": 5,
        "number_diagnoses": 3, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
        "A1Cresult": None, "max_glu_serum": None, "insulin": "No", "change": "No", "diabetesMed": "No",
    })
    valores = _row_to_values(row)
    assert "A1Cresult_ordinal" not in valores
    assert "max_glu_serum_ordinal" not in valores
    assert valores["insulin_ordinal"] == 0.0


def test_glycemic_testing_domain_declares_glucose_homeostasis_process():
    """A conexao deliberada com o NHANES: mesmo nome de processo."""
    domain = GlycemicTestingDomain()
    assert domain.processes() == {"glucose_homeostasis"}


def test_utilization_domain_marks_missing_features_explicitly():
    domain = HospitalUtilizationDomain(mean_std={"time_in_hospital": (4.0, 2.0)})
    measurements = {"time_in_hospital": Measurement(key="time_in_hospital", value=6.0, source="t", timestamp=datetime(2024, 1, 1))}
    features = domain.encode(measurements)
    por_nome = {f.name: f for f in features}
    assert por_nome["time_in_hospital"].is_missing is False
    assert por_nome["num_lab_procedures"].is_missing is True


def test_uci_representation_has_three_domains():
    representation = UCIHospitalRepresentation()
    assert set(representation.domain_names()) == {"utilization", "glycemic_testing", "medication_intensity"}


def test_uci_system_accumulates_multiple_encounters():
    system = UCIHospitalSystem(identifier="uci_teste")
    representation = UCIHospitalRepresentation()
    from biospace.core import Cohort

    cohort = Cohort()
    for i, num_med in enumerate([10, 12, 15]):
        ts = datetime(2020, 1, 1 + i)
        system.observe(Observation(timestamp=ts, source="encontro_hospitalar", values={
            "time_in_hospital": 2, "num_lab_procedures": 30, "num_procedures": 1, "num_medications": num_med,
            "number_diagnoses": 5, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
        }))
        cohort.update(system, representation, timestamp=ts)

    traj = cohort.trajectories[system.id]
    assert len(traj) == 3
