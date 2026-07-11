"""
biospace.plugins.diabetes
============================

Diabetes Tipo 2 NÃO é mais um pacote de representação próprio — é uma
INTERPRETAÇÃO CLÍNICA (N) aplicada sobre a representação genérica do
sistema metabólico (`biospace.plugins.metabolic`, que é R). Nenhum
domínio, sistema biológico ou representação vive aqui: tudo isso foi
migrado para `metabolic/`, que não sabe o que é diabetes.

O que este pacote de fato contribui, além de reexportar `metabolic`
para manter compatibilidade com código existente:

- `classify_diabetes_status` (reexportado de `metabolic.interpretations`)
  — o critério clínico específico de diabetes, uma função pura sobre
  um vetor já representado, nunca uma propriedade da representação.
- `generate_synthetic_dataframe` — um CENÁRIO clínico sintético
  específico de diabetes (progressão glicêmica, adoção de metformina/
  insulina, declínio renal por exposição glicêmica acumulada). O
  cenário é específico de diabetes; a representação que o consome não é.

Ver `biospace/plugins/metabolic/` para a arquitetura completa (B, O, R)
e `tests/test_metabolic_package_genericity.py` para a prova de que a
mesma representação sustenta múltiplas interpretações clínicas
independentes (diabetes E síndrome metabólica), não apenas esta.
"""

from __future__ import annotations

from biospace.plugins.metabolic import (
    AnthropometricDomain,
    CardiovascularDomain,
    ComorbidityDomain,
    FieldStats,
    GlycemicDomain,
    InsulinResistanceProxyDomain,
    MetabolicRepresentation,
    MetabolicSystem,
    Reference,
    RenalDomain,
    TreatmentDomain,
    classify_diabetes_status,
    exam,
    fit_reference,
    load_from_dataframe,
)

from .synthetic import generate_synthetic_dataframe

# Aliases preservando os nomes antigos (compatibilidade com testes, dashboard e
# exemplos já existentes que importam DiabetesSystem/DiabetesRepresentation) --
# ambos são literalmente as mesmas classes do pacote metabólico, não cópias.
DiabetesSystem = MetabolicSystem
DiabetesRepresentation = MetabolicRepresentation

__all__ = [
    "DiabetesSystem",
    "DiabetesRepresentation",
    "MetabolicSystem",
    "MetabolicRepresentation",
    "exam",
    "GlycemicDomain",
    "AnthropometricDomain",
    "CardiovascularDomain",
    "RenalDomain",
    "ComorbidityDomain",
    "TreatmentDomain",
    "InsulinResistanceProxyDomain",
    "FieldStats",
    "Reference",
    "fit_reference",
    "load_from_dataframe",
    "generate_synthetic_dataframe",
    "classify_diabetes_status",
]
