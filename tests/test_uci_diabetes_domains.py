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
import pytest

from biospace.core import Measurement, Observation
from biospace.datasets.uci_diabetes import (
    GlycemicTestingDomain,
    HospitalUtilizationDomain,
    UCIHospitalRepresentation,
    UCIHospitalSystem,
    _row_to_values,
    icd9_to_category,
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


class TestIcd9Categoria:
    """icd9_to_category: agrupamento padrao de Strack et al. (2014), testado contra codigos com categoria CONHECIDA."""

    @pytest.mark.parametrize("codigo,esperado", [
        ("250.8", "diabetes"), ("250", "diabetes"),
        ("428", "circulatory"), ("410", "circulatory"), ("785", "circulatory"),
        ("486", "respiratory"), ("491", "respiratory"), ("786", "respiratory"),
        ("787", "digestive"), ("530", "digestive"),
        ("996", "injury"), ("850", "injury"),
        ("715", "musculoskeletal"), ("720", "musculoskeletal"),
        ("580", "genitourinary"), ("788", "genitourinary"),
        ("174", "neoplasms"), ("153", "neoplasms"),
        ("V57", "other"), ("E888", "other"), ("276", "other"), ("38", "other"),
    ])
    def test_known_codes_map_to_expected_category(self, codigo, esperado):
        assert icd9_to_category(codigo) == esperado

    @pytest.mark.parametrize("codigo", ["?", None, "", "nan"])
    def test_missing_or_invalid_codes_map_to_none(self, codigo):
        assert icd9_to_category(codigo) is None


def test_diagnosis_category_domain_flags_any_of_three_diagnoses():
    """DiagnosisCategoryDomain deve marcar uma categoria como presente se QUALQUER um dos 3 diagnosticos cair nela -- nao so o primeiro."""
    row = pd.Series({
        "time_in_hospital": 2, "num_lab_procedures": 30, "num_procedures": 1, "num_medications": 10,
        "number_diagnoses": 3, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
        "diag_1": "428", "diag_2": "250.8", "diag_3": "715",  # circulatory, diabetes, musculoskeletal
    })
    valores = _row_to_values(row)
    assert valores["diag_cat_circulatory"] == 1.0
    assert valores["diag_cat_diabetes"] == 1.0
    assert valores["diag_cat_musculoskeletal"] == 1.0
    assert valores["diag_cat_respiratory"] == 0.0
    assert valores["diag_cat_neoplasms"] == 0.0


def test_diagnosis_category_domain_handles_missing_diagnosis_slots():
    """Encontros com diag_2/diag_3 ausentes (comum -- nem todo encontro tem 3 diagnosticos) nao devem quebrar, nem inventar categoria."""
    row = pd.Series({
        "time_in_hospital": 2, "num_lab_procedures": 30, "num_procedures": 1, "num_medications": 10,
        "number_diagnoses": 1, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
        "diag_1": "428", "diag_2": "?", "diag_3": None,
    })
    valores = _row_to_values(row)
    assert valores["diag_cat_circulatory"] == 1.0
    assert sum(v for k, v in valores.items() if k.startswith("diag_cat_")) == 1.0, "So circulatory deveria estar marcado -- os outros 2 slots sao ausentes/invalidos."


def test_uci_representation_now_has_four_domains():
    representation = UCIHospitalRepresentation()
    assert set(representation.domain_names()) == {"utilization", "glycemic_testing", "medication_intensity", "diagnosis_category"}


def test_demographics_extraction_uses_first_non_missing_value(tmp_path):
    """
    TESTE DECISIVO da extração de demografia: paciente 1 tem race="?"
    no primeiro encontro e um valor real no segundo -- deve usar o
    valor real (primeiro NÃO ausente), não "?" nem uma string vazia.
    Paciente 2 tem gender="Unknown/Invalid" -- deve virar None, não uma
    terceira categoria.
    """
    import csv as csv_module

    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    colunas_base = {
        "weight": "?", "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 1,
        "time_in_hospital": 2, "payer_code": "?", "medical_specialty": "?", "num_lab_procedures": 30,
        "num_procedures": 1, "num_medications": 10, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "diag_1": "428", "diag_2": "?", "diag_3": "?", "number_diagnoses": 1,
        "max_glu_serum": "None", "A1Cresult": "None", "metformin": "No", "repaglinide": "No",
        "nateglinide": "No", "chlorpropamide": "No", "glimepiride": "No", "acetohexamide": "No",
        "glipizide": "No", "glyburide": "No", "tolbutamide": "No", "pioglitazone": "No",
        "rosiglitazone": "No", "acarbose": "No", "miglitol": "No", "troglitazone": "No",
        "tolazamide": "No", "examide": "No", "citoglipton": "No", "insulin": "No",
        "glyburide-metformin": "No", "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No", "change": "No",
        "diabetesMed": "No", "readmitted": "NO",
    }
    linhas = [
        {"encounter_id": 1, "patient_nbr": 100, "race": "?", "gender": "Female", "age": "[40-50)", **colunas_base},
        {"encounter_id": 2, "patient_nbr": 100, "race": "Caucasian", "gender": "Female", "age": "[40-50)", **colunas_base},
        {"encounter_id": 3, "patient_nbr": 200, "race": "AfricanAmerican", "gender": "Unknown/Invalid", "age": "[60-70)", **colunas_base},
    ]
    caminho_csv = tmp_path / "fabricado.csv"
    with open(caminho_csv, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)

    cohort, representation = load_uci_diabetes_cohort(str(caminho_csv), include_demographics=True)

    sistema_100 = cohort.systems["uci_100"]
    assert sistema_100.metadata["race"] == "Caucasian", "Deveria usar o primeiro valor NAO ausente ('?' do 1o encontro ignorado)."

    sistema_200 = cohort.systems["uci_200"]
    assert sistema_200.metadata["gender"] is None, "'Unknown/Invalid' deveria virar None, nao uma terceira categoria."
    assert sistema_200.metadata["race"] == "AfricanAmerican"
