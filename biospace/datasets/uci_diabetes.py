"""
biospace.datasets.uci_diabetes
=================================

UCI Diabetes 130-US Hospitals (Strack et al., 2014) — 101.766 encontros
hospitalares, 71.518 pacientes. Estrutura REAL, inspecionada antes de
qualquer mapeamento (não assumida por analogia ao NHANES):

- SEM HbA1c/glicemia contínuas — só categorias (`A1Cresult`: None/Norm/>7/>8,
  83,3% ausente; `max_glu_serum`: None/Norm/>200/>300, 94,8% ausente).
- SEM IMC, circunferência abdominal, pressão arterial, creatinina —
  nenhuma dessas variáveis existe nesta base.
- Idade em FAIXAS de 10 anos (`[70-80)`), não contínua.
- Variáveis de UTILIZAÇÃO hospitalar 100% completas (time_in_hospital,
  num_lab_procedures, num_procedures, num_medications, number_diagnoses,
  number_outpatient, number_emergency, number_inpatient) — o oposto do
  padrão de completude do NHANES.
- `patient_nbr` (não `encounter_id`) é o paciente — 16.773 de 71.518
  pacientes (23%) têm MÚLTIPLOS encontros (até 39 no máximo observado),
  uma oportunidade real de trajetória que o NHANES (transversal) não tinha.

CONSEQUÊNCIA HONESTA: isto NÃO é "a mesma MetabolicRepresentation do
NHANES aplicada a uma segunda fonte" — a estrutura de dados é
estruturalmente incompatível com aquele conjunto de domínios (sem lab
contínuo, sem antropometria, sem pressão). É uma representação
GENUINAMENTE DIFERENTE, apropriada ao que esta fonte realmente mede:
utilização hospitalar e testes glicêmicos categóricos esparsos — ainda
assim testável dentro do mesmo framework BioSpace.

CONEXÃO DELIBERADA com o NHANES: `GlycemicTestingDomain` declara
`process="glucose_homeostasis"` — o MESMO nome de processo usado em
`plugins.metabolic.processes.GLUCOSE_HOMEOSTASIS` — porque A1Cresult e
max_glu_serum medem o mesmo mecanismo biológico que HbA1c/glicemia no
NHANES, só que categorizado e muito mais esparso. Isso não junta as
duas representações automaticamente (são objetos `Representation`
diferentes), mas registra formalmente que medem o mesmo processo.

`encounter_id` NÃO é uma data real — é um identificador sequencial que
cresce monotonicamente com o tempo (verificado empiricamente num
paciente com 39 encontros). Usado como PROXY de ordem cronológica para
construir trajetórias multi-encontro, com datas sintéticas espaçadas
por 1 dia cada (preservam ORDEM, não intervalo real) — documentado
explicitamente onde essa suposição é usada.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from biospace.core import BiologicalSystem, Cohort, Feature, Measurement, Observable, Observation, Representation, SemanticDomain

__all__ = [
    "HospitalUtilizationDomain",
    "GlycemicTestingDomain",
    "MedicationIntensityDomain",
    "UCIHospitalSystem",
    "UCIHospitalRepresentation",
    "load_uci_diabetes_cohort",
]

_UTILIZACAO_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient",
]

_A1C_ORDINAL = {"Norm": 0.0, ">7": 1.0, ">8": 2.0}
_GLU_ORDINAL = {"Norm": 0.0, ">200": 1.0, ">300": 2.0}
_INSULIN_ORDINAL = {"No": 0.0, "Steady": 1.0, "Down": 2.0, "Up": 2.0}


class _CountObservable(Observable):
    def __init__(self, key: str):
        self.key = key


class HospitalUtilizationDomain(SemanticDomain):
    """Intensidade de utilização hospitalar -- 100% completo nesta base (o oposto do padrão de completude do NHANES)."""

    name = "utilization"
    description = "Intensidade de utilização hospitalar no encontro (procedimentos, medicações, diagnósticos, visitas prévias)."

    def __init__(self, mean_std: Optional[dict[str, tuple[float, float]]] = None):
        super().__init__([_CountObservable(k) for k in _UTILIZACAO_COLS])
        self._mean_std = mean_std or {}

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key in _UTILIZACAO_COLS:
            m = measurements.get(key)
            if m is None or m.is_missing():
                features.append(Feature(name=key, value=0.0, raw_value=None, is_missing=True, weight=0.0, provenance=(key,)))
                continue
            raw = float(m.value)
            mean, std = self._mean_std.get(key, (0.0, 1.0))
            z = (raw - mean) / std if std > 1e-9 else 0.0
            features.append(Feature(name=key, value=z, raw_value=raw, z_score=z, provenance=(key,)))
        return features


class GlycemicTestingDomain(SemanticDomain):
    """
    A1Cresult/max_glu_serum recodificados ordinalmente (Norm=0, elevado
    moderado=1, elevado alto=2) — ESPARSO POR DESENHO: 83-95% ausente
    nesta base. `process="glucose_homeostasis"` -- o MESMO processo do
    NHANES (ver docstring do módulo).
    """

    name = "glycemic_testing"
    description = "Resultado categórico de HbA1c e glicemia máxima sérica, quando testados neste encontro (maioria ausente)."

    def __init__(self):
        obs_a1c = _CountObservable("A1Cresult_ordinal")
        obs_a1c.process = "glucose_homeostasis"
        obs_glu = _CountObservable("max_glu_serum_ordinal")
        obs_glu.process = "glucose_homeostasis"
        super().__init__([obs_a1c, obs_glu])

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key in ["A1Cresult_ordinal", "max_glu_serum_ordinal"]:
            m = measurements.get(key)
            if m is None or m.is_missing():
                features.append(Feature(name=key, value=0.0, raw_value=None, is_missing=True, weight=0.0, provenance=(key,)))
            else:
                v = float(m.value)
                features.append(Feature(name=key, value=v, raw_value=v, provenance=(key,)))
        return features


class MedicationIntensityDomain(SemanticDomain):
    """Status ordinal de insulina (No=0, Steady=1, Down/Up=2) e se houve mudança de medicação no encontro."""

    name = "medication_intensity"
    description = "Intensidade/instabilidade do regime medicamentoso no encontro."

    def __init__(self):
        super().__init__([_CountObservable("insulin_ordinal"), _CountObservable("change_flag"), _CountObservable("diabetes_med_flag")])

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key in ["insulin_ordinal", "change_flag", "diabetes_med_flag"]:
            m = measurements.get(key)
            v = float(m.value) if (m is not None and not m.is_missing()) else 0.0
            features.append(Feature(name=key, value=v, raw_value=v, provenance=(key,)))
        return features


class UCIHospitalSystem(BiologicalSystem):
    """Um paciente (patient_nbr) desta base -- pode ter 1 a 39 encontros observados, ordenados por encounter_id (proxy de ordem cronológica, não datas reais)."""

    pass


class UCIHospitalRepresentation(Representation):
    def __init__(self, mean_std_utilizacao: Optional[dict[str, tuple[float, float]]] = None):
        super().__init__([
            HospitalUtilizationDomain(mean_std_utilizacao),
            GlycemicTestingDomain(),
            MedicationIntensityDomain(),
        ])


def _fit_utilization_mean_std(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {col: (float(df[col].mean()), float(df[col].std())) for col in _UTILIZACAO_COLS}


def _row_to_values(row: pd.Series) -> dict:
    valores = {col: row[col] for col in _UTILIZACAO_COLS}
    valores["A1Cresult_ordinal"] = _A1C_ORDINAL.get(row.get("A1Cresult"))
    valores["max_glu_serum_ordinal"] = _GLU_ORDINAL.get(row.get("max_glu_serum"))
    valores["insulin_ordinal"] = _INSULIN_ORDINAL.get(row.get("insulin"), 0.0)
    valores["change_flag"] = 1.0 if row.get("change") == "Ch" else 0.0
    valores["diabetes_med_flag"] = 1.0 if row.get("diabetesMed") == "Yes" else 0.0
    return {k: v for k, v in valores.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}


def load_uci_diabetes_cohort(csv_path: str, max_rows: Optional[int] = None) -> tuple[Cohort, UCIHospitalRepresentation]:
    """
    Agrupa por `patient_nbr` (não `encounter_id`), ordena por
    `encounter_id` dentro de cada paciente (proxy de ordem cronológica),
    produz UM `UCIHospitalSystem` por paciente com trajetória completa
    (1 a 39 pontos observados).

    Datas sintéticas: primeiro encontro em 2020-01-01, cada encontro
    subsequente +1 dia -- preserva ORDEM, não intervalo real.
    """
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    mean_std = _fit_utilization_mean_std(df)
    representation = UCIHospitalRepresentation(mean_std_utilizacao=mean_std)
    cohort = Cohort()

    grupos = df.sort_values("encounter_id").groupby("patient_nbr")
    for patient_nbr, grupo in grupos:
        system = UCIHospitalSystem(identifier=f"uci_{patient_nbr}")
        system.metadata = {"paciente_original": str(patient_nbr), "n_encontros": len(grupo)}
        data_base = datetime(2020, 1, 1)
        for i, (_, row) in enumerate(grupo.iterrows()):
            ts = data_base + timedelta(days=i)
            valores = _row_to_values(row)
            system.observe(Observation(timestamp=ts, source="encontro_hospitalar", values=valores, metadata={"encounter_id": int(row["encounter_id"]), "readmitted": row.get("readmitted")}))
            cohort.update(system, representation, timestamp=ts)

    cohort.loader_report = {"n_encontros": len(df), "n_pacientes": df["patient_nbr"].nunique()}
    return cohort, representation
