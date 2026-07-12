"""
biospace.plugins.metabolic.processes
=======================================

Registro dos PhysiologicalProcess que os Observables deste pacote
declaram (ver `biospace.core.process` e `observables.py`). Puramente
documental — nenhuma lógica de `encode()`/`transform()` depende deste
módulo; ele existe para dar um nome e uma descrição estável a cada
processo referenciado por string em `Observable.process`.
"""

from __future__ import annotations

from biospace.core import PhysiologicalProcess

__all__ = ["GLUCOSE_HOMEOSTASIS", "BODY_COMPOSITION", "CARDIOVASCULAR_REGULATION", "RENAL_FILTRATION", "LIPID_METABOLISM", "ALL_PROCESSES"]

GLUCOSE_HOMEOSTASIS = PhysiologicalProcess(
    name="glucose_homeostasis",
    description="Regulação da concentração de glicose no sangue — medida direta (glicemia de jejum) e indireta acumulada (HbA1c, reflete ~3 meses de exposição glicêmica).",
)

BODY_COMPOSITION = PhysiologicalProcess(
    name="body_composition",
    description="Distribuição de massa corporal e adiposidade — IMC (proxy geral) e circunferência abdominal (proxy de adiposidade central, mais especificamente associada a risco metabólico que IMC isoladamente).",
)

CARDIOVASCULAR_REGULATION = PhysiologicalProcess(
    name="cardiovascular_regulation",
    description="Regulação de pressão arterial e frequência cardíaca de repouso pelo sistema cardiovascular autônomo.",
)

RENAL_FILTRATION = PhysiologicalProcess(
    name="renal_filtration",
    description="Capacidade de filtração glomerular dos rins — creatinina sérica (elevação indica pior função) e eGFR (queda indica pior função; direção invertida em relação à creatinina).",
)

LIPID_METABOLISM = PhysiologicalProcess(
    name="lipid_metabolism",
    description="Metabolismo de lipoproteínas — colesterol total, HDL (proteção cardiovascular, direção invertida) e triglicerídeos (medido só em subamostra em jejum).",
)

ALL_PROCESSES = [GLUCOSE_HOMEOSTASIS, BODY_COMPOSITION, CARDIOVASCULAR_REGULATION, RENAL_FILTRATION, LIPID_METABOLISM]
