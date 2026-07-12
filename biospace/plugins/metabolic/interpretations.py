"""
biospace.plugins.metabolic.interpretations
=============================================

Interpretações clínicas: funções puras que tomam um vetor JÁ
REPRESENTADO (RepresentationVector, produzido por MetabolicRepresentation)
e devolvem um rótulo clínico — nunca o contrário. Isto é N(R(B)), não R.

Este módulo é a prova de que o pacote metabólico não é "diabetes
disfarçado": duas interpretações clínicas DIFERENTES e
independentemente motivadas — diabetes (critério glicêmico, ADA) e
síndrome metabólica (critério composto, adaptado do NCEP ATP III) —
operam sobre exatamente o mesmo MetabolicRepresentation, sem que nenhum
domínio precise saber qual interpretação será aplicada depois.

Nenhuma função aqui é a representação. Todas recebem um vetor já
construído por `MetabolicRepresentation` e devolvem uma interpretação —
a distinção formal defendida no framework: B -> R(B) -> N(R(B)).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationVector

__all__ = ["classify_diabetes_status", "classify_metabolic_syndrome_risk"]

# Limiares em VALOR BRUTO (não z-score) -- por isso as interpretações usam
# raw_value das Features, nunca o value normalizado, que depende da
# referência populacional escolhida e mudaria o rótulo clínico conforme a coorte.
_LIMIAR_HBA1C_DIABETES = 6.5
_LIMIAR_HBA1C_PRE_DIABETES = 5.7
_LIMIAR_GLICEMIA_DIABETES = 126.0
_LIMIAR_GLICEMIA_PRE_DIABETES = 100.0


def _raw_value(vector: "RepresentationVector", domain_name: str, feature_name: str) -> Optional[float]:
    """
    BUG REAL CORRIGIDO: a versão anterior fazia
    `f.raw_value if f.raw_value is not None else f.value` -- para uma
    Feature AUSENTE (`is_missing=True`), `f.value` vale 0.0 (não None,
    ver `zscore_features`), então essa função nunca devolvia None de
    verdade para ausência real, sempre 0.0 silenciosamente. Isso tornava
    o ramo "indeterminado" de `classify_diabetes_status` inalcançável
    (pacientes sem HbA1c NEM glicemia caíam em "normal" em vez de
    "indeterminado") -- corrigido checando `is_missing` explicitamente.
    """
    for f in vector.components.get(domain_name, []):
        if f.name == feature_name:
            return None if f.is_missing else f.raw_value
    return None


def classify_diabetes_status(vector: "RepresentationVector") -> str:
    """
    Critério glicêmico da American Diabetes Association (ADA), aplicado
    sobre valores BRUTOS de HbA1c e glicemia de jejum já presentes no
    GlycemicDomain de um MetabolicRepresentation — não introduz nenhuma
    variável nova, apenas interpreta duas que já existem na representação.

    Retorna "diabetes", "pre_diabetes", ou "normal". Simplificação
    reconhecida: o critério ADA real também usa teste oral de tolerância
    à glicose (OGTT) e permite diagnóstico em consultas separadas — não
    modelados aqui.
    """
    hba1c = _raw_value(vector, "glycemic", "hba1c_pct")
    glicemia = _raw_value(vector, "glycemic", "glicemia_jejum_mg_dl")

    if hba1c is not None and hba1c >= _LIMIAR_HBA1C_DIABETES:
        return "diabetes"
    if glicemia is not None and glicemia >= _LIMIAR_GLICEMIA_DIABETES:
        return "diabetes"
    if hba1c is not None and hba1c >= _LIMIAR_HBA1C_PRE_DIABETES:
        return "pre_diabetes"
    if glicemia is not None and glicemia >= _LIMIAR_GLICEMIA_PRE_DIABETES:
        return "pre_diabetes"
    if hba1c is None and glicemia is None:
        return "indeterminado"
    return "normal"


def classify_metabolic_syndrome_risk(vector: "RepresentationVector") -> dict:
    """
    Versão ADAPTADA dos critérios NCEP ATP III para síndrome metabólica
    (>=3 de 5 critérios) -- adaptada porque este pacote não tem
    triglicerídeos nem HDL (não coletados nas Features disponíveis).
    Usa os 3 critérios disponíveis (circunferência abdominal, pressão
    arterial, glicemia de jejum) mais IMC como proxy adicional de
    adiposidade -- reportado explicitamente como ADAPTAÇÃO, não o
    critério clínico literal, para não superestimar validade clínica.

    Retorna um dict com cada critério avaliado individualmente e a
    contagem total -- nunca só um booleano, para que a interpretação
    permaneça auditável.
    """
    circunferencia = _raw_value(vector, "anthropometric", "circunferencia_abdominal_cm")
    imc = _raw_value(vector, "anthropometric", "imc")
    sistolica = _raw_value(vector, "cardiovascular", "pressao_sistolica_mmhg")
    diastolica = _raw_value(vector, "cardiovascular", "pressao_diastolica_mmhg")
    glicemia = _raw_value(vector, "glycemic", "glicemia_jejum_mg_dl")

    criterios = {
        "adiposidade_central": circunferencia is not None and circunferencia >= 102.0,
        "imc_elevado": imc is not None and imc >= 30.0,
        "pressao_elevada": (sistolica is not None and sistolica >= 130.0) or (diastolica is not None and diastolica >= 85.0),
        "glicemia_elevada": glicemia is not None and glicemia >= 100.0,
    }
    n_criterios = sum(1 for v in criterios.values() if v)
    return {
        "criterios": criterios,
        "n_criterios_presentes": n_criterios,
        "risco_elevado": n_criterios >= 2,  # limiar ajustado (4 criterios disponiveis, nao 5 -- ver docstring)
        "adaptacao_reconhecida": "Critério adaptado do NCEP ATP III -- sem triglicerídeos/HDL, usa IMC como proxy adicional.",
    }


def classify_metabolic_syndrome_risk_full(vector: "RepresentationVector") -> dict:
    """
    Critério NCEP ATP III COMPLETO (>=3 de 5), sexo-específico onde o
    critério clínico original exige -- diferente de
    `classify_metabolic_syndrome_risk` (a versão adaptada de 4
    critérios, sem lipídios). Requer LipidDomain (colesterol, HDL,
    triglicerídeos) e `sexo` em AnthropometricDomain, ambos só
    disponíveis a partir dos arquivos NHANES P_BIOPRO/P_TCHOL/P_HDL/
    P_TRIGLY -- se ausentes, o critério correspondente fica indefinido
    (não conta nem a favor nem contra), nunca assumido.

    5 critérios (Grundy et al., 2005 — AHA/NHLBI): circunferência
    abdominal >102cm(M)/>88cm(F); triglicerídeos >=150mg/dL; HDL
    <40mg/dL(M)/<50mg/dL(F); pressão >=130/85mmHg; glicemia de jejum
    >=100mg/dL. Retorna dict auditável, igual à versão adaptada.
    """
    circunferencia = _raw_value(vector, "anthropometric", "circunferencia_abdominal_cm")
    sexo = _raw_value(vector, "anthropometric", "sexo")  # NHANES: 1.0=masculino, 2.0=feminino
    sistolica = _raw_value(vector, "cardiovascular", "pressao_sistolica_mmhg")
    diastolica = _raw_value(vector, "cardiovascular", "pressao_diastolica_mmhg")
    glicemia = _raw_value(vector, "glycemic", "glicemia_jejum_mg_dl")
    trigliceridios = _raw_value(vector, "lipid", "trigliceridios_mg_dl")
    hdl = _raw_value(vector, "lipid", "hdl_mg_dl")

    limiar_cintura = None
    if sexo == 1.0:
        limiar_cintura = 102.0
    elif sexo == 2.0:
        limiar_cintura = 88.0

    limiar_hdl = None
    if sexo == 1.0:
        limiar_hdl = 40.0
    elif sexo == 2.0:
        limiar_hdl = 50.0

    criterios = {
        "adiposidade_central": None if (circunferencia is None or limiar_cintura is None) else circunferencia >= limiar_cintura,
        "trigliceridios_elevados": None if trigliceridios is None else trigliceridios >= 150.0,
        "hdl_baixo": None if (hdl is None or limiar_hdl is None) else hdl < limiar_hdl,
        "pressao_elevada": None
        if (sistolica is None and diastolica is None)
        else ((sistolica is not None and sistolica >= 130.0) or (diastolica is not None and diastolica >= 85.0)),
        "glicemia_elevada": None if glicemia is None else glicemia >= 100.0,
    }
    criterios_avaliaveis = {k: v for k, v in criterios.items() if v is not None}
    n_criterios = sum(1 for v in criterios_avaliaveis.values() if v)

    return {
        "criterios": criterios,
        "n_criterios_avaliaveis": len(criterios_avaliaveis),
        "n_criterios_presentes": n_criterios,
        "risco_elevado": None if len(criterios_avaliaveis) < 3 else n_criterios >= 3,
        "sexo_disponivel": sexo is not None,
        "nota": "Critério NCEP ATP III completo (Grundy et al., 2005) -- não uma adaptação. Critérios com dado ausente ficam None, não contam a favor nem contra.",
    }
