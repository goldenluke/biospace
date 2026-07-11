"""
biospace.datasets.nhanes
===========================

Loader para o ciclo NHANES "Pre-pandemic" (agosto/2017–março/2020,
prefixo "P_" — arquivos combinados de dois ciclos bienais, lançados
juntos pelo CDC por causa da interrupção da coleta pela pandemia).
Mapeado para as colunas que `biospace.plugins.metabolic.load_from_dataframe`
espera.

CORREÇÃO REAL (não suposição): `pandas.read_sas` exige `format="xport"`,
não `format="xpt"` como uma versão anterior deste módulo assumia —
descoberto ao testar contra os 6 arquivos reais enviados, que falhavam
com `ValueError: unknown SAS format` até a correção.

CORREÇÃO REAL #2: o arquivo de pressão arterial neste ciclo é `P_BPXO`
(Oscilométrica), não `P_BPX` como no ciclo 2017-2018 isolado — a
metodologia de medição mudou, e as variáveis são `BPXOSY1`/`BPXODI1`
(com "O" extra), não `BPXSY1`/`BPXDI1`. Descoberto inspecionando as
colunas reais dos arquivos enviados antes de assumir que o mapeamento
antigo (baseado em documentação do ciclo 2017-2018 isolado) bateria.

Nomes de variável CONFIRMADOS diretamente nos 6 arquivos reais
enviados (`pd.read_sas(..., format="xport").columns`), não apenas
contra documentação:

| Arquivo    | Variável NHANES | Nome no BioSpace              |
|------------|------------------|--------------------------------|
| P_DEMO     | SEQN             | paciente (chave de junção)    |
| P_DEMO     | RIDAGEYR         | idade                          |
| P_GHB      | LBXGH            | hba1c_pct                      |
| P_GLU      | LBXGLU           | glicemia_jejum_mg_dl           |
| P_BMX      | BMXBMI           | imc                             |
| P_BMX      | BMXWAIST         | circunferencia_abdominal_cm    |
| P_BPXO     | BPXOSY1          | pressao_sistolica_mmhg         |
| P_BPXO     | BPXODI1          | pressao_diastolica_mmhg        |
| P_DIQ      | DIQ010           | diabetes_autorreferido (1=Sim, 2=Não, 3=Borderline, 7/9=Recusa/NSei) |
| P_DIQ      | DIQ050           | insulina (recodificado 1=Sim/2=Não→1.0/0.0; só perguntado a quem respondeu Sim em DIQ010 — ~93% ausente por desenho do questionário, não erro) |

NÃO incluído (fora do conjunto de arquivos baixados): creatinina, eGFR,
frequência cardíaca de repouso (`P_BPXO` tem pulso — `BPXOPLS1` — mas
não foi mapeado ainda), perfil lipídico, hipoglicemiante oral (DIQ070
existe no arquivo mas mede "qualquer agente oral", não metformina
especificamente — não mapeado para não forçar uma equivalência
imprecisa), comorbidades além de diabetes autorreferido. Essas Features
do `MetabolicRepresentation` ficarão ausentes — imputadas por
completude (mecanismo já existente no
núcleo), não um erro.

NHANES é TRANSVERSAL, não longitudinal por desenho: cada SEQN aparece
uma vez, sem trajetória. `load_nhanes_metabolic_cohort` produz um
DataFrame de um registro por participante — `MetabolicSystem`s
resultantes terão `len(trajectory) == 1`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

__all__ = ["NHANES_PREPANDEMIC_FILES", "load_nhanes_metabolic_cohort"]

NHANES_PREPANDEMIC_FILES = {
    "demo": "P_DEMO.xpt",
    "ghb": "P_GHB.xpt",
    "glu": "P_GLU.xpt",
    "bmx": "P_BMX.xpt",
    "bpxo": "P_BPXO.xpt",
    "diq": "P_DIQ.xpt",
}

_COLUMN_MAP: dict[str, dict[str, str]] = {
    "demo": {"RIDAGEYR": "idade"},
    "ghb": {"LBXGH": "hba1c_pct"},
    "glu": {"LBXGLU": "glicemia_jejum_mg_dl"},
    "bmx": {"BMXBMI": "imc", "BMXWAIST": "circunferencia_abdominal_cm"},
    "bpxo": {"BPXOSY1": "pressao_sistolica_mmhg", "BPXODI1": "pressao_diastolica_mmhg"},
    "diq": {"DIQ010": "diabetes_autorreferido", "DIQ050": "insulina_bruta"},
}

# DIQ050 (uso de insulina) só é perguntado a quem já respondeu "Sim" para
# diabetes em DIQ010 -- por isso a alta taxa de ausência (~93%) não é um
# problema de qualidade do dado, é o desenho do questionário. Codificação
# NHANES: 1=Sim, 2=Não, 9=Não sabe. Recodificado para o padrão binário
# 1.0/0.0 que `TreatmentDomain.encode()` espera -- "Não sabe" (9) vira
# ausente, não um terceiro valor arbitrário.
_SIM_NAO_NHANES = {1.0: 1.0, 2.0: 0.0}

_ORDEM_JUNCAO = ["ghb", "glu", "bmx", "bpxo", "diq"]


def _read_xpt(path: Path) -> pd.DataFrame:
    """Wrapper fino sobre pandas.read_sas — format='xport', não 'xpt' (bug real, corrigido depois de testar contra arquivos reais)."""
    return pd.read_sas(path, format="xport")


def _select_and_rename(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Renomeia as colunas de interesse de UM arquivo já carregado e descarta o resto -- função pura, testável sem I/O de arquivo."""
    mapa = _COLUMN_MAP.get(key, {})
    renomeado = df.rename(columns=mapa)
    colunas = ["SEQN"] + list(mapa.values())
    faltando = [c for c in colunas if c not in renomeado.columns]
    if faltando:
        raise KeyError(f"Arquivo '{key}' não tem as colunas esperadas {faltando} — confira se é mesmo o ciclo/arquivo certo.")
    return renomeado[colunas]


def _merge_nhanes_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Recebe os 6 DataFrames JÁ CARREGADOS (chaves: demo, ghb, glu, bmx,
    bpxo, diq — as mesmas de NHANES_PREPANDEMIC_FILES), já com as colunas
    originais do NHANES (não renomeadas) — faz a seleção/renomeação e o
    merge por SEQN. Separado de `load_nhanes_metabolic_cohort` (que lida
    com arquivos em disco) precisamente para ser testável com dados
    FABRICADOS, sem precisar de arquivos .XPT reais nem de um escritor
    XPT (que pandas não tem).
    """
    processados = {key: _select_and_rename(df, key) for key, df in frames.items()}

    merged = processados["demo"]
    for key in _ORDEM_JUNCAO:
        merged = merged.merge(processados[key], on="SEQN", how="left")

    merged = merged.rename(columns={"SEQN": "paciente"})
    merged["paciente"] = merged["paciente"].astype(int).astype(str).apply(lambda s: f"nhanes_{s}")
    merged["data_exame"] = pd.Timestamp("2018-09-01")  # ciclo combinado ago/2017-mar/2020, sem data exata por participante -- ponto medio aproximado
    merged["insulina"] = merged["insulina_bruta"].map(_SIM_NAO_NHANES)  # 9.0 (Nao sabe) e NaN (nao perguntado) viram NaN aqui, tratados como ausencia pelo TreatmentDomain
    merged = merged.drop(columns=["insulina_bruta"])
    return merged


def load_nhanes_metabolic_cohort(data_dir: str, files: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """
    Lê os 6 arquivos .XPT de `data_dir` (nomes default em
    `NHANES_PREPANDEMIC_FILES`, sobrescrevíveis via `files`), junta por
    SEQN, produz um DataFrame compatível com
    `biospace.plugins.metabolic.load_from_dataframe`.

    Levanta `FileNotFoundError` com a URL exata de onde baixar, se
    algum arquivo estiver faltando — não falha silenciosamente nem
    inventa dado ausente.
    """
    files = files or NHANES_PREPANDEMIC_FILES
    base = Path(data_dir)

    frames_brutos: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo NHANES não encontrado: {path}\n"
                f"Baixe de: https://wwwn.cdc.gov/nchs/nhanes/tutorials/PreliminaryData.aspx (ciclo Pre-pandemic ago/2017-mar/2020, arquivo {filename})"
            )
        frames_brutos[key] = _read_xpt(path)

    return _merge_nhanes_frames(frames_brutos)
