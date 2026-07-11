"""
tests.test_nhanes_loader
============================

biospace.datasets.nhanes: testado com dados FABRICADOS, estruturalmente
idênticos ao NHANES real (mesmas colunas, mesmos tipos) — não dados
reais, que exigiriam acesso de rede a cdc.gov (indisponível neste
ambiente, `host_not_allowed` confirmado empiricamente) ou arquivos
.XPT enviados manualmente. `_merge_nhanes_frames` foi desenhada
especificamente para ser testável assim: separa a lógica de
merge/renomeação (testada aqui, com dados fabricados) da leitura de
arquivo (`pandas.read_sas`, funcionalidade já estabelecida do pandas,
não retestada).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biospace.datasets.nhanes import _merge_nhanes_frames, _select_and_rename


def _fabricated_frames(n=20, seed=0, omitir_glicemia_de=None):
    """Dados fabricados com a MESMA estrutura de coluna do NHANES 2017-2018 real (confirmada contra a documentação oficial do ciclo) -- nao sao valores reais de nenhum participante."""
    rng = np.random.default_rng(seed)
    seqns = np.arange(100000, 100000 + n)

    demo = pd.DataFrame({"SEQN": seqns, "RIDAGEYR": rng.integers(20, 80, n).astype(float), "RIAGENDR": rng.integers(1, 3, n).astype(float)})
    seqns_glu = seqns[:omitir_glicemia_de] if omitir_glicemia_de is not None else seqns
    glu = pd.DataFrame({"SEQN": seqns_glu, "LBXGLU": rng.normal(100, 20, len(seqns_glu))})
    ghb = pd.DataFrame({"SEQN": seqns, "LBXGH": rng.normal(6.0, 1.5, n)})
    bmx = pd.DataFrame({"SEQN": seqns, "BMXBMI": rng.normal(28, 5, n), "BMXWAIST": rng.normal(95, 12, n)})
    bpxo = pd.DataFrame({"SEQN": seqns, "BPXOSY1": rng.normal(125, 15, n), "BPXODI1": rng.normal(80, 10, n)})
    diq = pd.DataFrame({"SEQN": seqns, "DIQ010": rng.integers(1, 4, n).astype(float)})
    return {"demo": demo, "ghb": ghb, "glu": glu, "bmx": bmx, "bpxo": bpxo, "diq": diq}


def test_select_and_rename_keeps_only_mapped_columns():
    df = pd.DataFrame({"SEQN": [1, 2], "LBXGH": [5.5, 6.0], "COLUNA_IRRELEVANTE": [1, 2]})
    resultado = _select_and_rename(df, "ghb")
    assert list(resultado.columns) == ["SEQN", "hba1c_pct"]
    assert "COLUNA_IRRELEVANTE" not in resultado.columns


def test_select_and_rename_raises_on_missing_expected_column():
    df = pd.DataFrame({"SEQN": [1, 2], "COLUNA_ERRADA": [1, 2]})
    with pytest.raises(KeyError):
        _select_and_rename(df, "ghb")


def test_merge_produces_one_row_per_seqn():
    frames = _fabricated_frames(n=25)
    resultado = _merge_nhanes_frames(frames)
    assert len(resultado) == 25
    assert resultado["paciente"].nunique() == 25


def test_merge_renames_to_biospace_column_names():
    frames = _fabricated_frames(n=10)
    resultado = _merge_nhanes_frames(frames)
    esperadas = {
        "paciente", "idade", "hba1c_pct", "glicemia_jejum_mg_dl", "imc",
        "circunferencia_abdominal_cm", "pressao_sistolica_mmhg", "pressao_diastolica_mmhg",
        "diabetes_autorreferido", "data_exame",
    }
    assert esperadas <= set(resultado.columns)


def test_merge_handles_participants_missing_from_one_file():
    """NHANES real tem participantes sem exame de glicemia de jejum -- o merge (how='left') deve preservar o participante com NaN, nao descarta-lo."""
    frames = _fabricated_frames(n=20, omitir_glicemia_de=15)
    resultado = _merge_nhanes_frames(frames)
    assert len(resultado) == 20, "Participantes sem glicemia nao deveriam ser descartados pelo merge."
    assert resultado["glicemia_jejum_mg_dl"].isna().sum() == 5


def test_patient_id_is_prefixed_and_stable():
    frames = _fabricated_frames(n=5)
    resultado = _merge_nhanes_frames(frames)
    assert all(p.startswith("nhanes_") for p in resultado["paciente"])


def test_end_to_end_fabricated_data_produces_valid_metabolic_representation():
    """
    Fecha o ciclo completo: dados FABRICADOS -> merge -> load_from_dataframe
    -> Cohort/Representation validos. O dominio renal deve aparecer
    AUSENTE (imputado por completude), nao um erro -- nao baixamos
    creatinina/eGFR neste conjunto inicial de arquivos.
    """
    from biospace.plugins.metabolic import load_from_dataframe

    frames = _fabricated_frames(n=15)
    df_nhanes = _merge_nhanes_frames(frames)
    cohort, representation = load_from_dataframe(df_nhanes)

    assert len(cohort.trajectories) == 15
    assert set(representation.domain_names()) == {"glycemic", "anthropometric", "cardiovascular", "renal", "comorbidity", "treatment"}

    sid = next(iter(cohort.trajectories))
    vetor = cohort.trajectories[sid].latest()
    features_renal = {f.name: f for f in vetor.components["renal"]}
    assert features_renal["creatinina_mg_dl"].is_missing is True
    assert features_renal["taxa_filtracao_glomerular"].is_missing is True

    features_glycemic = {f.name: f for f in vetor.components["glycemic"]}
    assert features_glycemic["hba1c_pct"].is_missing is False


def test_missing_file_raises_with_download_url():
    from biospace.datasets.nhanes import load_nhanes_metabolic_cohort

    with pytest.raises(FileNotFoundError) as exc_info:
        load_nhanes_metabolic_cohort("/tmp/pasta_que_nao_existe_12345")
    assert "cdc.gov" in str(exc_info.value)
