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
    diq = pd.DataFrame({"SEQN": seqns, "DIQ010": rng.integers(1, 4, n).astype(float), "DIQ050": rng.choice([1.0, 2.0, np.nan], n)})
    biopro = pd.DataFrame({"SEQN": seqns, "LBXSCR": rng.normal(0.9, 0.3, n).clip(0.3, 3.0)})
    tchol = pd.DataFrame({"SEQN": seqns, "LBXTC": rng.normal(190, 35, n)})
    hdl = pd.DataFrame({"SEQN": seqns, "LBDHDD": rng.normal(50, 15, n)})
    seqns_trig = seqns[: n // 2]  # so ~metade tem triglicerideos (subamostra em jejum, achado real replicado no fabricador)
    triglycerides = pd.DataFrame({"SEQN": seqns_trig, "LBXTR": rng.normal(120, 50, len(seqns_trig))})
    return {"demo": demo, "ghb": ghb, "glu": glu, "bmx": bmx, "bpxo": bpxo, "diq": diq, "biopro": biopro, "tchol": tchol, "hdl": hdl, "triglycerides": triglycerides}


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
        "paciente", "idade", "sexo", "hba1c_pct", "glicemia_jejum_mg_dl", "imc",
        "circunferencia_abdominal_cm", "pressao_sistolica_mmhg", "pressao_diastolica_mmhg",
        "diabetes_autorreferido", "insulina", "data_exame",
        "creatinina_mg_dl", "taxa_filtracao_glomerular", "colesterol_total_mg_dl", "hdl_mg_dl", "trigliceridios_mg_dl",
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
    -> Cohort/Representation validos. O dominio renal agora deve estar
    PRESENTE (creatinina + eGFR calculado via CKD-EPI 2021) -- diferente
    de antes, quando ficava sempre ausente (nao tinhamos baixado esses
    arquivos ainda).
    """
    from biospace.plugins.metabolic import load_from_dataframe

    frames = _fabricated_frames(n=15)
    df_nhanes = _merge_nhanes_frames(frames)
    cohort, representation = load_from_dataframe(df_nhanes)

    assert len(cohort.trajectories) == 15
    assert set(representation.domain_names()) == {"glycemic", "anthropometric", "cardiovascular", "renal", "lipid", "comorbidity", "treatment"}

    sid = next(iter(cohort.trajectories))
    vetor = cohort.trajectories[sid].latest()
    features_renal = {f.name: f for f in vetor.components["renal"]}
    assert features_renal["creatinina_mg_dl"].is_missing is False
    assert features_renal["taxa_filtracao_glomerular"].is_missing is False
    assert features_renal["taxa_filtracao_glomerular"].raw_value > 0

    features_glycemic = {f.name: f for f in vetor.components["glycemic"]}
    assert features_glycemic["hba1c_pct"].is_missing is False


def test_missing_file_raises_with_download_url():
    from biospace.datasets.nhanes import load_nhanes_metabolic_cohort

    with pytest.raises(FileNotFoundError) as exc_info:
        load_nhanes_metabolic_cohort("/tmp/pasta_que_nao_existe_12345")
    assert "cdc.gov" in str(exc_info.value)


# =============================================================================
# CKD-EPI 2021 (eGFR) -- formula clinica, testada com rigor extra
# =============================================================================
def test_egfr_matches_hand_calculated_reference_value_male():
    """TESTE DECISIVO: Scr=kappa exatamente (razao=1) -- os dois termos min/max colapsam para 1, sobra so 142*0.9938^idade. Calculado a mao: 142*0.9938^50 = 104.049."""
    from biospace.datasets.nhanes import _calcular_egfr_ckd_epi_2021

    resultado = _calcular_egfr_ckd_epi_2021(creatinina_mg_dl=0.9, idade=50, sexo=1.0)  # kappa masculino = 0.9
    assert resultado == pytest.approx(104.049, abs=0.01)


def test_egfr_matches_hand_calculated_reference_value_female():
    """Mesma logica, kappa feminino=0.7: Scr=0.7 exatamente -> 142*0.9938^50*1.012 (multiplicador feminino)."""
    from biospace.datasets.nhanes import _calcular_egfr_ckd_epi_2021

    resultado = _calcular_egfr_ckd_epi_2021(creatinina_mg_dl=0.7, idade=50, sexo=2.0)
    esperado = 142 * (0.9938**50) * 1.012
    assert resultado == pytest.approx(esperado, abs=0.01)


def test_egfr_returns_none_for_missing_or_invalid_inputs():
    from biospace.datasets.nhanes import _calcular_egfr_ckd_epi_2021

    assert _calcular_egfr_ckd_epi_2021(None, 50, 1.0) is None
    assert _calcular_egfr_ckd_epi_2021(0.9, None, 1.0) is None
    assert _calcular_egfr_ckd_epi_2021(0.9, 50, None) is None
    assert _calcular_egfr_ckd_epi_2021(0.9, 50, 9.0) is None, "sexo=9.0 (recusa/nao sabe, codificacao NHANES) nao deveria assumir um sexo default."
    assert _calcular_egfr_ckd_epi_2021(0.0, 50, 1.0) is None, "creatinina <= 0 e clinicamente impossivel."
    assert _calcular_egfr_ckd_epi_2021(0.9, 0, 1.0) is None, "idade <= 0 e clinicamente impossivel."
    assert _calcular_egfr_ckd_epi_2021(np.nan, 50, 1.0) is None, "NaN do pandas (nao None do Python) tambem deve virar None -- bug real corrigido."
    assert _calcular_egfr_ckd_epi_2021(0.9, np.nan, 1.0) is None


def test_egfr_decreases_with_age_holding_creatinine_fixed():
    """Propriedade monotonica esperada: eGFR cai com idade, mesma creatinina -- 0.9938^idade e' estritamente decrescente."""
    from biospace.datasets.nhanes import _calcular_egfr_ckd_epi_2021

    egfr_jovem = _calcular_egfr_ckd_epi_2021(0.9, 30, 1.0)
    egfr_idoso = _calcular_egfr_ckd_epi_2021(0.9, 70, 1.0)
    assert egfr_idoso < egfr_jovem


def test_egfr_decreases_with_creatinine_above_kappa():
    """Acima do kappa de referencia, eGFR deve cair com creatinina mais alta (pior funcao renal)."""
    from biospace.datasets.nhanes import _calcular_egfr_ckd_epi_2021

    egfr_normal = _calcular_egfr_ckd_epi_2021(0.9, 50, 1.0)
    egfr_alterado = _calcular_egfr_ckd_epi_2021(2.5, 50, 1.0)
    assert egfr_alterado < egfr_normal


def test_egfr_computed_correctly_end_to_end_via_merge():
    """O eGFR calculado dentro de _merge_nhanes_frames bate com chamar a funcao diretamente -- nao ha divergencia entre o pipeline e a formula isolada."""
    from biospace.datasets.nhanes import _calcular_egfr_ckd_epi_2021

    frames = _fabricated_frames(n=10, seed=99)
    resultado = _merge_nhanes_frames(frames)

    for _, linha in resultado.iterrows():
        esperado = _calcular_egfr_ckd_epi_2021(linha.get("creatinina_mg_dl"), linha.get("idade"), linha.get("sexo"))
        obtido = linha.get("taxa_filtracao_glomerular")
        if esperado is None:
            assert pd.isna(obtido)
        else:
            assert obtido == pytest.approx(esperado, abs=1e-6)
