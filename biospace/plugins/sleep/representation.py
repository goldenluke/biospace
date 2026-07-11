"""
biospace.plugins.sleep.representation
========================================

R : SleepSystem -> X

Composição dos 8 domínios reais de SAOS (5 numéricos codificados por
z-score + 3 textuais estruturados).
"""

from __future__ import annotations

from typing import Optional

from biospace.core import Representation

from .domains import (
    AnthropometricDomain,
    ApneaDomain,
    CardiovascularDomain,
    ComorbidityDomain,
    HypoxiaDomain,
    Reference,
    SleepArchitectureDomain,
    SymptomsDomain,
    TreatmentDomain,
)

__all__ = ["SleepRepresentation"]


class SleepRepresentation(Representation):
    """
    Representação padrão de SAOS. Se `reference` não for informada, cada
    domínio numérico usa as estatísticas de fallback ilustrativas — para
    uso com dados reais, prefira construir via
    `biospace.plugins.sleep.loader.load_from_dataframe/load_from_excel`,
    que ajustam `reference` diretamente sobre a população carregada.
    """

    def __init__(self, reference: Optional[Reference] = None):
        super().__init__(
            [
                AnthropometricDomain(reference),
                ApneaDomain(reference),
                HypoxiaDomain(reference),
                SleepArchitectureDomain(reference),
                CardiovascularDomain(reference),
                ComorbidityDomain(),
                SymptomsDomain(),
                TreatmentDomain(),
            ]
        )
