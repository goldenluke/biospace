"""
biospace.plugins.sleep.hierarchical
======================================

HierarchicalSleepRepresentation: demonstração concreta de
CompositeRepresentation, reorganizando os domínios do plugin sleep na
hierarquia proposta:

    patient.representation
    ├── RespiratoryRepresentation    (ApneaDomain)
    ├── HypoxiaRepresentation        (HypoxiaDomain)
    ├── CardiovascularRepresentation (CardiovascularDomain)
    └── MetabolicRepresentation      (AnthropometricDomain — ver ressalva abaixo)
    + domínios não agrupados: SleepArchitecture, Comorbidity, Symptoms, Treatment

IMPORTANTE — esta é uma representação ALTERNATIVA, não um substituto de
`SleepRepresentation`. O pipeline padrão (`loader.py`, dashboard,
`Ontology`, scripts) continua usando `SleepRepresentation` (nomes de
domínio "planos": "apnea", "hypoxia", ...) — trocar isso quebraria tudo
que já depende desses nomes. `HierarchicalSleepRepresentation` existe
para demonstrar e testar a arquitetura de composição, não para
substituir silenciosamente o que já está em produção.

RESSALVA SOBRE "MetabolicRepresentation": esta planilha não tem
marcadores metabólicos reais (glicemia, HbA1c, perfil lipídico) — o
único proxy disponível é `AnthropometricDomain` (IMC), que é, na melhor
das hipóteses, um indicador indireto e fraco de risco metabólico, não
uma representação genuína do sistema metabólico. Nomeei assim só para
seguir literalmente a hierarquia proposta; um plugin de diabetes, por
exemplo, teria um MetabolicRepresentation muito mais completo (glicemia,
HbA1c, perfil lipídico).
"""

from __future__ import annotations

from biospace.core import CompositeRepresentation, Representation

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

__all__ = ["HierarchicalSleepRepresentation"]


class HierarchicalSleepRepresentation(Representation):
    """Ver docstring do módulo. Aceita `reference` como `SleepRepresentation`, para reaproveitar uma `Reference` já ajustada."""

    def __init__(self, reference: "Reference | None" = None):
        respiratory = CompositeRepresentation(
            name="respiratory",
            description="Sistema respiratório — eventos obstrutivos e ronco",
            children=[ApneaDomain(reference)],
        )
        hypoxia = CompositeRepresentation(
            name="hypoxia_system",
            description="Sistema de oxigenação — dessaturação e carga hipóxica",
            children=[HypoxiaDomain(reference)],
        )
        cardiovascular = CompositeRepresentation(
            name="cardiovascular_system",
            description="Sistema cardiovascular — frequência cardíaca durante o sono",
            children=[CardiovascularDomain(reference)],
        )
        metabolic = CompositeRepresentation(
            name="metabolic_system",
            description=(
                "Sistema metabólico — PROXY FRACO via IMC apenas; esta planilha não tem "
                "glicemia/HbA1c/perfil lipídico. Ver ressalva no módulo."
            ),
            children=[AnthropometricDomain(reference)],
        )

        # Os demais domínios permanecem "soltos" (não agrupados) — a hierarquia
        # cobre apenas os 4 grupos explicitamente propostos.
        super().__init__(
            [
                respiratory,
                hypoxia,
                cardiovascular,
                metabolic,
                SleepArchitectureDomain(reference),
                ComorbidityDomain(),
                SymptomsDomain(),
                TreatmentDomain(),
            ]
        )
        self.respiratory = respiratory
        self.hypoxia_group = hypoxia
        self.cardiovascular_group = cardiovascular
        self.metabolic_group = metabolic
