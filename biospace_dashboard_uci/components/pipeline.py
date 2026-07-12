"""
components.pipeline
======================

Ponte entre o biospace (Cohort / RepresentationSpace) e o Streamlit,
para a coorte UCI Diabetes 130-US Hospitals real.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from biospace.core import Cohort, RepresentationSpace
from biospace.datasets.uci_diabetes import UCIHospitalRepresentation, load_uci_diabetes_cohort
from biospace.phenotyping import KMeansPhenotyper


@dataclass
class Pipeline:
    cohort: Cohort
    representation: UCIHospitalRepresentation
    space: RepresentationSpace
    phenotyper: KMeansPhenotyper
    display_df: pd.DataFrame
    n_encontros: int
    n_pacientes: int


def _linha_paciente(cohort: Cohort, space: RepresentationSpace, labels: dict, sid: str) -> dict:
    system = cohort.systems[sid]
    traj = cohort.trajectories[sid]
    ultima_obs = system.observations[-1]
    valores = system.latest_values()
    return {
        "id": sid,
        "paciente": system.metadata.get("paciente_original", sid),
        "n_encontros": len(traj),
        "readmitted_ultimo_encontro": ultima_obs.metadata.get("readmitted"),
        "time_in_hospital": valores.get("time_in_hospital"),
        "num_medications": valores.get("num_medications"),
        "num_lab_procedures": valores.get("num_lab_procedures"),
        "number_outpatient": valores.get("number_outpatient"),
        "number_emergency": valores.get("number_emergency"),
        "number_inpatient": valores.get("number_inpatient"),
        "insulin_ordinal": valores.get("insulin_ordinal"),
        "fenotipo": labels.get(sid),
    }


def run_pipeline(csv_path: str, max_rows: int | None = None, n_clusters: int = 4, include_diagnosis_category: bool = True) -> Pipeline:
    """
    Carrega o CSV real, agrupa por paciente, roda representação e fenotipagem.

    `include_diagnosis_category`: ACHADO REAL -- incluir esse domínio
    (default True, igual ao loader) muda o que K-Means encontra como
    fenótipo dominante -- de um organizado por utilização PRÉVIA
    (associação forte com readmissão, ~2,2x) para um organizado mais
    por tempo de internação extremo (associação mais fraca, ~1,5x). Ver
    a página 'Fenótipos e Readmissão', que expõe as duas representações
    lado a lado em vez de esconder a diferença.
    """
    cohort, representation = load_uci_diabetes_cohort(csv_path, max_rows=max_rows, include_diagnosis_category=include_diagnosis_category)
    space = cohort.snapshot()
    order = representation.domain_names()

    phenotyper = KMeansPhenotyper(n_clusters=n_clusters)
    phenotypes = phenotyper.fit(space)

    labels = {}
    for sid in space.ids():
        vec = space.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

    linhas = [_linha_paciente(cohort, space, labels, sid) for sid in cohort.trajectories]
    display_df = pd.DataFrame(linhas)

    return Pipeline(
        cohort=cohort, representation=representation, space=space, phenotyper=phenotyper, display_df=display_df,
        n_encontros=cohort.loader_report["n_encontros"], n_pacientes=cohort.loader_report["n_pacientes"],
    )
