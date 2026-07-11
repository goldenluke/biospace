"""
biospace.plugins.metabolic.representation
============================================

MetabolicRepresentation: compõe os 6 domínios do pacote em uma única
Representation, aceitando uma Reference compartilhada (ajustada uma vez
sobre a população, como no plugin sleep). Esta é R(B) — não sabe nada
sobre diabetes, síndrome metabólica, ou qualquer outra interpretação
clínica; produz um espaço de representação único sobre o qual
QUALQUER interpretação pode operar depois (ver `interpretations.py`).
"""

from __future__ import annotations

from typing import Optional

from biospace.core import Representation

from .domains import AnthropometricDomain, CardiovascularDomain, ComorbidityDomain, GlycemicDomain, RenalDomain, TreatmentDomain
from .reference import Reference

__all__ = ["MetabolicRepresentation"]


class MetabolicRepresentation(Representation):
    def __init__(self, reference: Optional[Reference] = None):
        super().__init__(
            [
                GlycemicDomain(reference),
                AnthropometricDomain(reference),
                CardiovascularDomain(reference),
                RenalDomain(reference),
                ComorbidityDomain(),
                TreatmentDomain(),
            ]
        )
