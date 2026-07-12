"""
biospace.survival
====================

Análise de sobrevivência em tempo discreto (Kaplan-Meier, log-rank,
Cox proportional hazards) sobre a infraestrutura de Cohort/Trajectory
do meta-modelo — genérica, não específica de nenhuma doença.

Ver `discrete_time.py` para a extração do dataset (tempo = índice
ordinal, nunca calendário real a menos que explicitamente vindo de
timestamps genuínos) e `models.py` para os ajustes estatísticos.

DISTINÇÃO DE DESIGN, registrada explicitamente após uma auditoria do
projeto (não uma duplicação a resolver — os dois têm propósitos
genuinamente diferentes, mantidos ambos):

- **Este módulo** (`biospace.survival`): tempo = ÍNDICE ORDINAL
  (posição na sequência de observações), depende de `lifelines`, e
  suporta Cox (regressão com covariáveis, hazard ratios). Certo
  quando o timestamp não representa um intervalo de calendário real
  — o caso da UCI Diabetes 130-US Hospitals, onde `encounter_id` é só
  proxy de ordem, não data verdadeira.
- **`biospace.longitudinal.survival`** (`SurvivalAnalyzer`/
  `SurvivalOperator`): tempo = DIAS DE CALENDÁRIO reais
  (`timestamp` de cada `Observation`), implementação própria de
  Kaplan-Meier sem dependência externa (mais alinhado com a filosofia
  de núcleo autocontido do projeto), sem Cox. Certo quando os
  timestamps são genuinamente significativos como intervalo de tempo
  — o caso do plugin sleep, onde as datas de exame são reais.

Ao escolher qual usar: pergunte se o tempo entre observações da fonte
de dados é um intervalo real (calendário) ou só uma ordem (proxy) —
essa distinção, não preferência de API, é o critério.
"""

from __future__ import annotations

from .discrete_time import DiscreteTimeToEventDataset, build_discrete_time_to_event
from .models import CoxModelReport, SurvivalByGroupReport, fit_cox_model, kaplan_meier_by_group

__all__ = [
    "DiscreteTimeToEventDataset",
    "build_discrete_time_to_event",
    "SurvivalByGroupReport",
    "kaplan_meier_by_group",
    "CoxModelReport",
    "fit_cox_model",
]
