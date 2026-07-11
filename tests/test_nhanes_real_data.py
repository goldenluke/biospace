"""
tests.test_nhanes_real_data
===============================

Testes contra os arquivos NHANES REAIS (ciclo Pre-pandemic ago/2017-mar/2020),
enviados pelo usuário — pytest.mark.skipif quando ausentes (mesmo
padrão de tests/test_stability_robustness.py e
tests/test_phenotype_stability_sweep.py).

MOTIVAÇÃO REAL para o primeiro teste: os testes em test_nhanes_loader.py
mockam `_read_xpt` inteiramente (testam `_merge_nhanes_frames` com
DataFrames já carregados) — nunca exercitam `pandas.read_sas` de
verdade. Foi exatamente aí que um bug real (`format="xpt"` em vez de
`format="xport"`) ficou escondido até os arquivos reais chegarem.
`test_read_xpt_works_against_real_files` fecha essa lacuna.
"""

from __future__ import annotations

import os

import pytest

CAMINHO_UPLOADS = "/mnt/user-data/uploads"
ARQUIVOS_NHANES = {"demo": "P_DEMO.xpt", "ghb": "P_GHB.xpt", "glu": "P_GLU.xpt", "bmx": "P_BMX.xpt", "bpxo": "P_BPXO.xpt", "diq": "P_DIQ.xpt"}

_todos_presentes = all(os.path.exists(os.path.join(CAMINHO_UPLOADS, f)) for f in ARQUIVOS_NHANES.values())
pytestmark = pytest.mark.skipif(not _todos_presentes, reason="Arquivos NHANES reais não disponíveis neste ambiente (esperado em CI).")


@pytest.fixture(scope="module")
def real_nhanes_dataframe():
    from biospace.datasets.nhanes import load_nhanes_metabolic_cohort

    return load_nhanes_metabolic_cohort(CAMINHO_UPLOADS, files=ARQUIVOS_NHANES)


@pytest.fixture(scope="module")
def real_nhanes_adults(real_nhanes_dataframe):
    return real_nhanes_dataframe[real_nhanes_dataframe["idade"] >= 20].copy()


def test_read_xpt_works_against_real_files():
    """
    Fecha a lacuna real: testes com dado fabricado mockam _read_xpt
    inteiramente e nunca teriam pego o bug format='xpt' vs 'xport'.
    Este teste exercita pandas.read_sas de verdade.
    """
    from biospace.datasets.nhanes import _read_xpt

    df = _read_xpt(os.path.join(CAMINHO_UPLOADS, "P_DEMO.xpt"))
    assert "SEQN" in df.columns
    assert "RIDAGEYR" in df.columns
    assert len(df) > 1000


def test_real_nhanes_loads_with_plausible_shape(real_nhanes_dataframe):
    assert len(real_nhanes_dataframe) > 10000
    assert set(real_nhanes_dataframe.columns) >= {
        "paciente", "idade", "hba1c_pct", "glicemia_jejum_mg_dl", "imc",
        "circunferencia_abdominal_cm", "pressao_sistolica_mmhg", "pressao_diastolica_mmhg",
    }


def test_real_nhanes_includes_children_by_design(real_nhanes_dataframe):
    """NHANES amostra toda a populacao, incluindo criancas -- achado real, nao um bug (idade minima proxima de 0)."""
    assert real_nhanes_dataframe["idade"].min() < 1.0


def test_derived_variables_return_none_on_cross_sectional_nhanes_data(real_nhanes_dataframe):
    """NHANES e transversal -- qualquer DerivedVariable que precise de >=2 pontos deve devolver None, nunca inventar uma trajetoria que nao existe."""
    from biospace.plugins.metabolic import HbA1cSlopeVariable, load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_dataframe.head(50))
    sid = next(iter(cohort.trajectories))
    traj = cohort.trajectories[sid]
    assert len(traj) == 1
    assert HbA1cSlopeVariable().compute(traj) is None


def test_classify_diabetes_status_against_real_self_reported_diagnosis(real_nhanes_adults):
    """
    O TESTE DECISIVO com dado real: classify_diabetes_status (criterio
    ADA sobre HbA1c/glicemia) comparado contra diagnostico
    AUTORREFERIDO (DIQ010) em ~9 mil adultos reais. Resultado
    documentado como regressao -- sensibilidade/especificidade
    plausiveis clinicamente (subdiagnostico de diabetes e
    pre-diabetes e um fenomeno real e bem documentado, nao um erro
    do classificador).
    """
    from biospace.plugins.metabolic import classify_diabetes_status, load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)

    tp = fn = fp = tn = 0
    for sid, traj in cohort.trajectories.items():
        paciente_original = cohort.systems[sid].metadata.get("paciente_original")
        linha = real_nhanes_adults[real_nhanes_adults["paciente"] == paciente_original]
        if linha.empty:
            continue
        autorreferido = linha.iloc[0]["diabetes_autorreferido"]
        if autorreferido not in (1.0, 2.0):
            continue
        status = classify_diabetes_status(traj.latest())
        if status == "indeterminado":
            continue
        predisse_diabetes = status == "diabetes"
        tem_diabetes_real = autorreferido == 1.0
        if predisse_diabetes and tem_diabetes_real:
            tp += 1
        elif not predisse_diabetes and tem_diabetes_real:
            fn += 1
        elif predisse_diabetes and not tem_diabetes_real:
            fp += 1
        else:
            tn += 1

    sensibilidade = tp / (tp + fn)
    especificidade = tn / (tn + fp)

    assert 0.55 < sensibilidade < 0.80, f"Sensibilidade {sensibilidade:.3f} fora da faixa clinicamente plausivel."
    assert especificidade > 0.90, f"Especificidade {especificidade:.3f} abaixo do esperado."


def test_process_coherence_confirmed_on_real_population_unlike_synthetic_data(real_nhanes_adults):
    """
    ACHADO REAL, contrastando com test_metabolic_synthetic_data_does_not_confirm_coherence
    (tests/test_process_coherence.py): em populacao REAL, a coerencia
    de processo SE CONFIRMA -- HbA1c e glicemia (glucose_homeostasis)
    correlacionam genuinamente mais entre si que com pressao arterial/IMC
    (outros processos). O gerador sintetico de diabetes NAO tinha essa
    propriedade (variaveis sorteadas independentemente); dados reais tem.
    """
    from biospace.core import check_process_coherence
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()

    relatorio = check_process_coherence(representation, space)
    assert relatorio.is_coherent is True, (
        "Esperava coerencia de processo confirmada em populacao real -- se isto falhar, "
        "investigar se algo mudou na forma como os dados sao carregados ou normalizados."
    )
