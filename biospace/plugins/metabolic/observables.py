"""
biospace.plugins.metabolic.observables
========================================

Observables do pacote metabólico — cada um extrai UMA grandeza
clinicamente nomeada de um MetabolicSystem. Nomes de campo em
português, mesmo padrão do plugin sleep.

Cada Observable declara opcionalmente `process` (ver
`biospace.core.process.PhysiologicalProcess`): o mecanismo biológico
que a grandeza mede, independente de qual SemanticDomain a consome.
Isto é anotação aditiva — nada aqui muda o comportamento de `encode()`
nem do resto do pipeline; existe só para habilitar
`Representation.processes()`/`features_by_process()`.
"""

from __future__ import annotations

from biospace.core import Observable

__all__ = [
    "GlicemiaJejumObservable",
    "HbA1cObservable",
    "IdadeObservable",
    "ImcObservable",
    "CircunferenciaAbdominalObservable",
    "PressaoSistolicaObservable",
    "PressaoDiastolicaObservable",
    "FrequenciaCardiacaObservable",
    "CreatininaObservable",
    "TaxaFiltracaoGlomerularObservable",
]


class GlicemiaJejumObservable(Observable):
    key = "glicemia_jejum_mg_dl"
    unit = "mg/dL"
    description = "Glicemia de jejum"
    process = "glucose_homeostasis"


class HbA1cObservable(Observable):
    key = "hba1c_pct"
    unit = "%"
    description = "Hemoglobina glicada"
    process = "glucose_homeostasis"


class IdadeObservable(Observable):
    key = "idade"
    unit = "anos"
    # Sem `process`: idade é um covariável, não a medida de um mecanismo
    # fisiológico específico -- deixar sem processo é a escolha correta
    # aqui, não uma omissão (ver docstring de PhysiologicalProcess: a
    # camada é opcional por Observable, não obrigatória).


class ImcObservable(Observable):
    key = "imc"
    unit = "kg/m²"
    process = "body_composition"


class CircunferenciaAbdominalObservable(Observable):
    key = "circunferencia_abdominal_cm"
    unit = "cm"
    process = "body_composition"


class PressaoSistolicaObservable(Observable):
    key = "pressao_sistolica_mmhg"
    unit = "mmHg"
    process = "cardiovascular_regulation"


class PressaoDiastolicaObservable(Observable):
    key = "pressao_diastolica_mmhg"
    unit = "mmHg"
    process = "cardiovascular_regulation"


class FrequenciaCardiacaObservable(Observable):
    key = "fc_repouso_bpm"
    unit = "bpm"
    process = "cardiovascular_regulation"


class CreatininaObservable(Observable):
    key = "creatinina_mg_dl"
    unit = "mg/dL"
    description = "Creatinina sérica — quanto MAIOR, pior a função renal"
    process = "renal_filtration"


class TaxaFiltracaoGlomerularObservable(Observable):
    key = "taxa_filtracao_glomerular"
    unit = "mL/min/1.73m²"
    description = "eGFR — quanto MENOR, pior a função renal (sinal invertido no domínio)"
    process = "renal_filtration"
