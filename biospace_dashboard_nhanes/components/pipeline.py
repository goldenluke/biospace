"""
components.pipeline
======================

Ponte entre o biospace (Cohort / RepresentationSpace) e o Streamlit,
para a coorte NHANES real. Nunca reimplementa lógica clínica aqui —
toda classificação vem de `biospace.plugins.metabolic`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from biospace.core import Cohort, RepresentationSpace
from biospace.datasets.nhanes import NHANES_PREPANDEMIC_FILES, load_nhanes_metabolic_cohort
from biospace.plugins.metabolic import (
    MetabolicRepresentation,
    classify_diabetes_status,
    classify_metabolic_syndrome_risk,
    classify_metabolic_syndrome_risk_full,
    load_from_dataframe,
)


@dataclass
class Pipeline:
    cohort: Cohort
    representation: MetabolicRepresentation
    space: RepresentationSpace
    display_df: pd.DataFrame
    n_total: int
    n_adultos: int


def _linha_paciente(cohort: Cohort, space: RepresentationSpace, sid: str, order: list[str]) -> dict:
    system = cohort.systems[sid]
    valores = system.latest_values()
    vetor = space.get(sid)
    status_diabetes = classify_diabetes_status(vetor)
    sindrome = classify_metabolic_syndrome_risk(vetor)
    sindrome_completa = classify_metabolic_syndrome_risk_full(vetor)
    return {
        "id": sid,
        "paciente": system.metadata.get("paciente_original", sid),
        "idade": valores.get("idade"),
        "sexo": valores.get("sexo"),
        "hba1c_pct": valores.get("hba1c_pct"),
        "glicemia_jejum_mg_dl": valores.get("glicemia_jejum_mg_dl"),
        "imc": valores.get("imc"),
        "circunferencia_abdominal_cm": valores.get("circunferencia_abdominal_cm"),
        "pressao_sistolica_mmhg": valores.get("pressao_sistolica_mmhg"),
        "pressao_diastolica_mmhg": valores.get("pressao_diastolica_mmhg"),
        "creatinina_mg_dl": valores.get("creatinina_mg_dl"),
        "taxa_filtracao_glomerular": valores.get("taxa_filtracao_glomerular"),
        "colesterol_total_mg_dl": valores.get("colesterol_total_mg_dl"),
        "hdl_mg_dl": valores.get("hdl_mg_dl"),
        "trigliceridios_mg_dl": valores.get("trigliceridios_mg_dl"),
        "diabetes_autorreferido": valores.get("diabetes_autorreferido"),
        "status_diabetes_laboratorial": status_diabetes,
        "sindrome_metabolica_risco": sindrome["risco_elevado"],
        "sindrome_metabolica_n_criterios": sindrome["n_criterios_presentes"],
        "sindrome_metabolica_completa_risco": sindrome_completa["risco_elevado"],
        "sindrome_metabolica_completa_n_avaliaveis": sindrome_completa["n_criterios_avaliaveis"],
    }


def run_pipeline(data_dir: str, idade_minima: int = 20, max_pacientes: int | None = None) -> Pipeline:
    """Carrega os 10 arquivos NHANES reais (6 originais + creatinina/colesterol/HDL/triglicerídeos), filtra por idade mínima, roda a representação e as interpretações clínicas."""
    df_bruto = load_nhanes_metabolic_cohort(data_dir, files=NHANES_PREPANDEMIC_FILES)
    n_total = len(df_bruto)

    df_adultos = df_bruto[df_bruto["idade"] >= idade_minima].copy()
    if max_pacientes is not None:
        df_adultos = df_adultos.head(max_pacientes)
    n_adultos = len(df_adultos)

    cohort, representation = load_from_dataframe(df_adultos)
    space = cohort.snapshot()
    order = representation.domain_names()

    linhas = [_linha_paciente(cohort, space, sid, order) for sid in cohort.trajectories]
    display_df = pd.DataFrame(linhas)

    return Pipeline(cohort=cohort, representation=representation, space=space, display_df=display_df, n_total=n_total, n_adultos=n_adultos)
