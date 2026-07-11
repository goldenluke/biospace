"""
components.pipeline
======================

Ponte entre o biospace (Cohort / RepresentationSpace / Phenotype) e o
Streamlit. O dashboard NUNCA reimplementa lógica clínica aqui — apenas
projeta o que já existe no biospace em um DataFrame conveniente para
plotly/st.dataframe. A modelagem em si (representação, geometria,
fenotipagem) acontece inteiramente em `biospace`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from biospace.core import Cohort, Phenotype, RepresentationSpace
from biospace.phenotyping import ClinicalKMeansPhenotyper
from biospace.plugins.sleep import SleepRepresentation, classificar_apneia, classificar_hipoxemia, load_from_dataframe


@dataclass
class Pipeline:
    cohort: Cohort
    representation: SleepRepresentation
    space: RepresentationSpace
    phenotyper: ClinicalKMeansPhenotyper
    phenotypes: list[Phenotype]
    display_df: pd.DataFrame


def _phenotype_of(space: RepresentationSpace, phenotypes: list[Phenotype], system_id: str) -> str | None:
    if system_id not in space.ids():
        return None
    vec = space.get(system_id).as_vector(space.order())
    return next((ph.name for ph in phenotypes if ph.contains(vec)), None)


def build_display_dataframe(cohort: Cohort, space: RepresentationSpace, phenotypes: list[Phenotype]) -> pd.DataFrame:
    """
    Projeta a Cohort em um DataFrame plano — só para visualização
    (histogramas, tabelas, scatter plots). A fonte da verdade continua
    sendo `cohort`/`space`, não este DataFrame.
    """
    rows = []
    for sid, system in cohort.systems.items():
        values = system.latest_values()
        rows.append(
            {
                "id": sid,
                "paciente": system.metadata.get("paciente_original", sid) if hasattr(system, "metadata") else sid,
                "idade": values.get("idade"),
                "genero": values.get("genero"),
                "peso_kg": values.get("peso_kg"),
                "altura_cm": values.get("altura_cm"),
                "imc": values.get("imc"),
                "ido": values.get("ido"),
                "ido_sono": values.get("ido_sono"),
                "no_de_dessaturacoes": values.get("no_de_dessaturacoes"),
                "tempo_total_de_ronco_min": values.get("tempo_total_de_ronco_min"),
                "spo2_minima": values.get("spo2_minima"),
                "spo2_media": values.get("spo2_media"),
                "spo2_maxima": values.get("spo2_maxima"),
                "tempo_spo2_90": values.get("tempo_spo2_90"),
                "carga_hipoxica_min_h": values.get("carga_hipoxica_min_h"),
                "no_de_eventos_de_hipoxemia": values.get("no_de_eventos_de_hipoxemia"),
                "tempo_total_em_hipoxemia_min": values.get("tempo_total_em_hipoxemia_min"),
                "tempo_para_dormir_min": values.get("tempo_para_dormir_min"),
                "tempo_total_de_sono_min": values.get("tempo_total_de_sono_min"),
                "eficiencia_do_sono": values.get("eficiencia_do_sono"),
                "fc_minima_bpm": values.get("fc_minima_bpm"),
                "fc_media_bpm": values.get("fc_media_bpm"),
                "fc_maxima_bpm": values.get("fc_maxima_bpm"),
                "doencas": values.get("doencas", "") or "",
                "sintomas": values.get("sintomas", "") or "",
                "tratamentos": values.get("tratamentos", "") or "",
                "classe_apneia": classificar_apneia(values.get("ido")),
                "classe_hipoxemia": classificar_hipoxemia(values.get("spo2_minima")),
                "fenotipo": _phenotype_of(space, phenotypes, sid),
            }
        )
    df = pd.DataFrame(rows)
    if "fc_maxima_bpm" in df.columns and "fc_minima_bpm" in df.columns:
        df["amplitude_fc"] = df["fc_maxima_bpm"] - df["fc_minima_bpm"]
    return df


def run_pipeline(raw_df: pd.DataFrame) -> Pipeline:
    """Roda o pipeline biospace inteiro: ingestão -> representação -> espaço -> fenotipagem -> dataframe de exibição."""
    cohort, representation = load_from_dataframe(raw_df)
    space = cohort.snapshot()
    phenotyper = ClinicalKMeansPhenotyper()
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
