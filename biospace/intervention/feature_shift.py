"""
biospace.intervention.feature_shift
======================================

FeatureShiftIntervention — τ(x) = x, exceto nas Features nomeadas em
`shifts`, que recebem um deslocamento aditivo fixo. Formaliza, como um
Operator reutilizável, o padrão que vínhamos aplicando manualmente em
`demo_sleep.py` para simular uma segunda visita pós-tratamento (reduzir
IDO, aumentar SpO2 mínima etc.) — agora como uma transformação nomeada,
auditável e reaproveitável, em vez de um bloco de código solto.

IMPORTANTE: o deslocamento em `shifts` é aplicado a `feature.value` — a
coordenada já normalizada/ponderada do espaço de representação X, não a
unidade clínica bruta. Um shift de `-10` em "ido" significa "-10 em
z-score ponderado", não "-10 eventos/hora de IDO". `feature.raw_value`
permanece inalterado (é só um registro do valor originalmente medido) —
use-o para conferir o que a Measurement original dizia, não para
interpretar o efeito do shift.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .base import InterventionOperator

if TYPE_CHECKING:
    from biospace.core import RepresentationVector

__all__ = ["FeatureShiftIntervention"]


class FeatureShiftIntervention(InterventionOperator):
    def __init__(self, shifts: dict[str, float]):
        """`shifts`: dict Feature.name -> deslocamento aditivo aplicado a `feature.value`."""
        self.shifts = shifts

    def apply(self, vector: "RepresentationVector") -> "RepresentationVector":
        from biospace.core import RepresentationVector

        matched_keys: set[str] = set()
        new_components = {}
        for domain_name, features in vector.components.items():
            new_features = []
            for feature in features:
                if feature.name in self.shifts:
                    matched_keys.add(feature.name)
                    new_features.append(replace(feature, value=feature.value + self.shifts[feature.name]))
                else:
                    new_features.append(feature)
            new_components[domain_name] = new_features

        unmatched = set(self.shifts.keys()) - matched_keys
        if unmatched:
            raise KeyError(
                f"{sorted(unmatched)} não corresponde a nenhuma Feature.name em `vector` — o shift NÃO foi "
                "aplicado para essas chaves (falharia silenciosamente sem esta checagem). `shifts` usa o nome "
                "NÃO qualificado da Feature (ex.: 'ido'), não o nome qualificado por domínio (ex.: 'apnea.ido' "
                "— esse formato é usado em outras partes do sistema, como FactorAnalysisLatentDomain, mas não aqui)."
            )

        return RepresentationVector(system_id=vector.system_id, timestamp=vector.timestamp, components=new_components)

    def describe(self) -> str:
        shifts_str = ", ".join(f"{k}{v:+.2f}" for k, v in self.shifts.items())
        return f"FeatureShiftIntervention({shifts_str})"
