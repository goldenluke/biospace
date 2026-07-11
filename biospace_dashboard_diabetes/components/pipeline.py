"""
components.pipeline
======================

Ponte entre o biospace (Cohort / RepresentationSpace / Phenotype) e o
Streamlit — mesmo papel do dashboard de sleep, agora sobre o plugin de
diabetes. Nunca reimplementa lógica clínica aqui.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from biospace.core import Cohort, Phenotype, RepresentationSpace
from biospace.phenotyping import KMeansPhenotyper
from biospace.plugins.diabetes import DiabetesRepresentation, load_from_dataframe


@dataclass
class Pipeline:
    cohort: Cohort
    representation: DiabetesRepresentation
    space: RepresentationSpace
    phenotyper: KMeansPhenotyper
    phenotypes: list[Phenotype]
    display_df: pd.DataFrame


def _classificar_controle(hba1c: float | None) -> str | None:
    """Classificação clínica padrão de controle glicêmico por HbA1c (ADA) — só para exibição, não usada na representação."""
    if hba1c is None:
        return None
    if hba1c < 7.0:
        return "Controlado"
    if hba1c < 9.0:
        return "Moderado"
    return "Descompensado"


def _phenotype_of(space: RepresentationSpace, phenotypes: list[Phenotype], system_id: str) -> str | None:
    if system_id not in space.ids():
        return None
    vec = space.get(system_id).as_vector(space.order())
    return next((ph.name for ph in phenotypes if ph.contains(vec)), None)


def build_display_dataframe(cohort: Cohort, space: RepresentationSpace, phenotypes: list[Phenotype]) -> pd.DataFrame:
    rows = []
    for sid, system in cohort.systems.items():
        values = system.latest_values()
        rows.append(
            {
                "id": sid,
                "paciente": system.metadata.get("paciente_original", sid) if hasattr(system, "metadata") else sid,
                "idade": values.get("idade"),
                "imc": values.get("imc"),
                "circunferencia_abdominal_cm": values.get("circunferencia_abdominal_cm"),
                "glicemia_jejum_mg_dl": values.get("glicemia_jejum_mg_dl"),
                "hba1c_pct": values.get("hba1c_pct"),
                "pressao_sistolica_mmhg": values.get("pressao_sistolica_mmhg"),
                "pressao_diastolica_mmhg": values.get("pressao_diastolica_mmhg"),
                "fc_repouso_bpm": values.get("fc_repouso_bpm"),
                "creatinina_mg_dl": values.get("creatinina_mg_dl"),
                "taxa_filtracao_glomerular": values.get("taxa_filtracao_glomerular"),
                "hipertensao": values.get("hipertensao", 0),
                "retinopatia": values.get("retinopatia", 0),
                "neuropatia": values.get("neuropatia", 0),
                "doenca_cardiovascular": values.get("doenca_cardiovascular", 0),
                "metformina": values.get("metformina", 0),
                "insulina": values.get("insulina", 0),
                "classe_controle": _classificar_controle(values.get("hba1c_pct")),
                "fenotipo": _phenotype_of(space, phenotypes, sid),
                "n_exames": len(cohort.trajectories[sid]) if sid in cohort.trajectories else 0,
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(raw_df: pd.DataFrame) -> Pipeline:
    """Roda o pipeline biospace inteiro: ingestão -> representação -> espaço -> fenotipagem -> dataframe de exibição."""
    cohort, representation = load_from_dataframe(raw_df)
    space = cohort.snapshot()
    phenotyper = KMeansPhenotyper(n_clusters=3)
    phenotypes = phenotyper.fit(space)
    display_df = build_display_dataframe(cohort, space, phenotypes)
    return Pipeline(
        cohort=cohort,
        representation=representation,
        space=space,
        phenotyper=phenotyper,
        phenotypes=phenotypes,
        display_df=display_df,
    )
