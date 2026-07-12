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

Nomes de variável CONFIRMADOS diretamente nos arquivos reais enviados
(`pd.read_sas(..., format="xport").columns`) e contra a documentação
oficial do CDC via busca (não assumidos de memória):

| Arquivo    | Variável NHANES | Nome no BioSpace              |
|------------|------------------|--------------------------------|
| P_DEMO     | SEQN             | paciente (chave de junção)    |
| P_DEMO     | RIDAGEYR         | idade                          |
| P_DEMO     | RIAGENDR         | sexo (1.0=masculino, 2.0=feminino, codificação NHANES padrão) |
| P_GHB      | LBXGH            | hba1c_pct                      |
| P_GLU      | LBXGLU            | glicemia_jejum_mg_dl           |
| P_BMX      | BMXBMI           | imc                             |
| P_BMX      | BMXWAIST         | circunferencia_abdominal_cm    |
| P_BPXO     | BPXOSY1          | pressao_sistolica_mmhg         |
| P_BPXO     | BPXODI1          | pressao_diastolica_mmhg        |
| P_DIQ      | DIQ010           | diabetes_autorreferido (1=Sim, 2=Não, 3=Borderline, 7/9=Recusa/NSei) |
| P_DIQ      | DIQ050           | insulina (recodificado 1=Sim/2=Não→1.0/0.0; só perguntado a quem respondeu Sim em DIQ010 — ~93% ausente por desenho do questionário, não erro) |
| P_BIOPRO   | LBXSCR           | creatinina_mg_dl (LBXSCH/LBXSTR do mesmo arquivo NÃO usados — documentação do CDC recomenda explicitamente os arquivos de referência TCHOL/TRIGLY em vez do painel bioquímico padrão para colesterol/triglicerídeos) |
| P_TCHOL    | LBXTC            | colesterol_total_mg_dl         |
| P_HDL      | LBDHDD           | hdl_mg_dl                       |
| P_TRIGLY   | LBXTR            | trigliceridios_mg_dl (só medido em subamostra EM JEJUM — n muito menor que os demais arquivos, ~42% da coorte, não erro de carregamento) |

`taxa_filtracao_glomerular` (eGFR) é CALCULADA, não observada diretamente —
fórmula CKD-EPI 2021 sem raça (Inker et al., N Engl J Med 2021;
κ=0,7♀/0,9♂, α=-0,241♀/-0,302♂, confirmados contra a National Kidney
Foundation e múltiplas fontes independentes antes de implementar,
não a versão de 2009 com coeficientes diferentes) — requer
creatinina + idade + sexo simultaneamente; ausente se qualquer um
dos três estiver ausente.

NÃO incluído: frequência cardíaca de repouso (`P_BPXO` tem pulso —
`BPXOPLS1` — mas não foi mapeado ainda), hipoglicemiante oral (DIQ070
existe no arquivo mas mede "qualquer agente oral", não metformina
especificamente — não mapeado para não forçar uma equivalência
imprecisa), comorbidades além de diabetes autorreferido, LDL (as 3
variantes de cálculo — Friedewald/Martin-Hopkins/NIH — existem no
arquivo TRIGLY mas não são necessárias para os 5 critérios de síndrome
metabólica NCEP ATP III, que usam triglicerídeos e HDL diretamente,
não LDL). Essas Features do `MetabolicRepresentation` ficarão
ausentes — imputadas por completude (mecanismo já existente no
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
    "biopro": "P_BIOPRO.xpt",
    "tchol": "P_TCHOL.xpt",
    "hdl": "P_HDL.xpt",
    "triglycerides": "P_TRIGLY.xpt",
}

_COLUMN_MAP: dict[str, dict[str, str]] = {
    "demo": {"RIDAGEYR": "idade", "RIAGENDR": "sexo"},
    "ghb": {"LBXGH": "hba1c_pct"},
    "glu": {"LBXGLU": "glicemia_jejum_mg_dl"},
    "bmx": {"BMXBMI": "imc", "BMXWAIST": "circunferencia_abdominal_cm"},
    "bpxo": {"BPXOSY1": "pressao_sistolica_mmhg", "BPXODI1": "pressao_diastolica_mmhg"},
    "diq": {"DIQ010": "diabetes_autorreferido", "DIQ050": "insulina_bruta"},
    "biopro": {"LBXSCR": "creatinina_mg_dl"},
    "tchol": {"LBXTC": "colesterol_total_mg_dl"},
    "hdl": {"LBDHDD": "hdl_mg_dl"},
    "triglycerides": {"LBXTR": "trigliceridios_mg_dl"},
}

# DIQ050 (uso de insulina) só é perguntado a quem já respondeu "Sim" para
# diabetes em DIQ010 -- por isso a alta taxa de ausência (~93%) não é um
# problema de qualidade do dado, é o desenho do questionário. Codificação
# NHANES: 1=Sim, 2=Não, 9=Não sabe. Recodificado para o padrão binário
# 1.0/0.0 que `TreatmentDomain.encode()` espera -- "Não sabe" (9) vira
# ausente, não um terceiro valor arbitrário.
_SIM_NAO_NHANES = {1.0: 1.0, 2.0: 0.0}

_ORDEM_JUNCAO = ["ghb", "glu", "bmx", "bpxo", "diq", "biopro", "tchol", "hdl", "triglycerides"]


def _calcular_egfr_ckd_epi_2021(creatinina_mg_dl: float, idade: float, sexo: float) -> Optional[float]:
    """
    CKD-EPI 2021 sem coeficiente de raça (Inker et al., New Creatinine-
    and Cystatin C-Based Equations to Estimate GFR without Race, N Engl
    J Med 2021;385:1737-1749) -- κ e α CONFIRMADOS contra a National
    Kidney Foundation e múltiplas fontes independentes antes de
    implementar (não a versão de 2009, que tem α diferente: -0,329♀/-0,411♂).

    eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1,200) × 0,9938^idade × (1,012 se feminino)
    κ = 0,7 (feminino) / 0,9 (masculino); α = -0,241 (feminino) / -0,302 (masculino).

    `sexo`: codificação NHANES (1.0=masculino, 2.0=feminino) -- devolve
    None se `sexo` não for 1.0 nem 2.0 (ausente ou código de recusa),
    nunca assume um sexo por default.
    """
    import math

    if creatinina_mg_dl is None or idade is None or sexo is None:
        return None
    if (isinstance(creatinina_mg_dl, float) and math.isnan(creatinina_mg_dl)) or (isinstance(idade, float) and math.isnan(idade)):
        return None
    if creatinina_mg_dl <= 0 or idade <= 0:
        return None
    if sexo == 2.0:
        kappa, alpha, mult_feminino = 0.7, -0.241, 1.012
    elif sexo == 1.0:
        kappa, alpha, mult_feminino = 0.9, -0.302, 1.0
    else:
        return None

    razao = creatinina_mg_dl / kappa
    termo_min = min(razao, 1.0) ** alpha
    termo_max = max(razao, 1.0) ** (-1.200)
    egfr = 142.0 * termo_min * termo_max * (0.9938**idade) * mult_feminino
    return float(egfr)


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
    merged["taxa_filtracao_glomerular"] = merged.apply(
        lambda linha: _calcular_egfr_ckd_epi_2021(linha.get("creatinina_mg_dl"), linha.get("idade"), linha.get("sexo")), axis=1
    )
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
