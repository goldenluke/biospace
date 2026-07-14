"""
biospace.ontology.terminology
=================================

Liga `Observable.key` (o vocabulário interno deste projeto) a códigos
de terminologias clínicas formais — LOINC (testes laboratoriais e
sinais vitais) e SNOMED CT (condições/diagnósticos). Camada
estritamente aditiva sobre `Observable` (mesmo espírito de `process`
em `biospace.core.observation`): não modifica a classe base, não é
exigida por nenhum contrato do núcleo, e um `Observable` sem entrada
aqui continua funcionando de forma idêntica.

Todo código listado em `LOINC_CODES`/`SNOMED_CODES` foi verificado por
busca contra fonte pública (loinc.org, snomed CT via terminologia
pública) antes de ser incluído aqui — não um número lembrado de
memória de treinamento. Onde não há confiança suficiente na
correspondência exata (por exemplo, sinalizadores de comorbidade que
não mapeiam limpamente para um único conceito), a entrada é
deliberadamente omitida, não preenchida com um palpite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["TerminologyCode", "LOINC_CODES", "SNOMED_CODES", "lookup_loinc", "lookup_snomed", "coverage_report"]


@dataclass(frozen=True)
class TerminologyCode:
    system: str  # "LOINC" ou "SNOMED CT"
    code: str
    display: str  # nome oficial do conceito, na terminologia de origem


# Verificado por busca contra loinc.org antes de incluir aqui (nao memoria).
LOINC_CODES: dict[str, TerminologyCode] = {
    "hba1c_pct": TerminologyCode("LOINC", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "glicemia_jejum_mg_dl": TerminologyCode("LOINC", "1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma"),
    "idade": TerminologyCode("LOINC", "30525-0", "Age"),
    "imc": TerminologyCode("LOINC", "39156-5", "Body mass index (BMI) [Ratio]"),
    "circunferencia_abdominal_cm": TerminologyCode("LOINC", "56115-9", "Waist Circumference"),
    "pressao_sistolica_mmhg": TerminologyCode("LOINC", "8480-6", "Systolic blood pressure"),
    "pressao_diastolica_mmhg": TerminologyCode("LOINC", "8462-4", "Diastolic blood pressure"),
    "fc_repouso_bpm": TerminologyCode("LOINC", "8867-4", "Heart rate"),
    "creatinina_mg_dl": TerminologyCode("LOINC", "2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
    "taxa_filtracao_glomerular": TerminologyCode("LOINC", "48642-3", "Glomerular filtration rate (creatinine-based formula)"),
    "colesterol_total_mg_dl": TerminologyCode("LOINC", "2093-3", "Cholesterol [Mass/volume] in Serum or Plasma"),
    "hdl_mg_dl": TerminologyCode("LOINC", "2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma"),
    "trigliceridios_mg_dl": TerminologyCode("LOINC", "2571-8", "Triglyceride [Mass/volume] in Serum or Plasma"),
    # UCI Diabetes 130-US Hospitals: mesmo conceito de teste que as entradas acima do NHANES --
    # o mesmo analito/grandeza clinica (hemoglobina glicada, glicose), so codificado como
    # categoria ordinal (normal/elevado/alto) em vez de valor continuo. O codigo LOINC
    # identifica O TESTE, nao a convencao de encoding escolhida por uma fonte especifica --
    # ligar as duas eh o proprio ponto de uma terminologia formal (interoperabilidade real
    # entre fontes estruturalmente incompativeis, nao coincidencia de nome de coluna).
    "A1Cresult_ordinal": TerminologyCode("LOINC", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "max_glu_serum_ordinal": TerminologyCode("LOINC", "2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
}

# Verificado por busca contra fonte publica de SNOMED CT antes de incluir aqui.
SNOMED_CODES: dict[str, TerminologyCode] = {
    "diabetes_tipo_2": TerminologyCode("SNOMED CT", "44054006", "Diabetes mellitus type 2"),
    "hipertensao": TerminologyCode("SNOMED CT", "38341003", "Hypertensive disorder"),
    "retinopatia": TerminologyCode("SNOMED CT", "4855003", "Retinopathy due to diabetes mellitus"),
    "neuropatia": TerminologyCode("SNOMED CT", "230572002", "Neuropathy due to diabetes mellitus"),
    "doenca_cardiovascular": TerminologyCode("SNOMED CT", "49601007", "Disorder of cardiovascular system"),
}

# Verificado por busca contra RxNorm (National Library of Medicine) antes de incluir aqui --
# nao sao codigos LOINC/SNOMED, mas RxCUI (identificador de ingrediente farmaceutico), a
# terminologia formal correta pra medicamentos, nao pra testes ou condicoes.
RXNORM_CODES: dict[str, TerminologyCode] = {
    "metformina": TerminologyCode("RxNorm", "6809", "Metformin"),
    "insulina": TerminologyCode("RxNorm", "5856", "Insulin (Regular)"),
}


def lookup_loinc(observable_key: str) -> Optional[TerminologyCode]:
    return LOINC_CODES.get(observable_key)


def lookup_snomed(observable_key: str) -> Optional[TerminologyCode]:
    return SNOMED_CODES.get(observable_key)


def lookup_rxnorm(observable_key: str) -> Optional[TerminologyCode]:
    return RXNORM_CODES.get(observable_key)


def coverage_report(observable_keys: list[str]) -> dict[str, Optional[TerminologyCode]]:
    """Para uma lista de chaves de Observable (ex.: as de uma Representation real),
    devolve qual tem código de terminologia mapeado e qual não -- relatado
    explicitamente, não escondido, para não passar a impressão de cobertura
    maior do que a real."""
    resultado = {}
    for key in observable_keys:
        resultado[key] = lookup_loinc(key) or lookup_snomed(key) or lookup_rxnorm(key)
    return resultado
