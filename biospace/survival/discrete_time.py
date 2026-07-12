"""
biospace.survival.discrete_time
==================================

Análise de sobrevivência em tempo discreto, sobre a MESMA
infraestrutura de Cohort/Trajectory do resto do projeto — não
específico de nenhuma doença ou fonte de dado.

"Tempo" aqui é sempre um ÍNDICE ORDINAL (posição na sequência de
observações de um paciente), nunca inferido como intervalo real de
calendário — a menos que o Observation.timestamp de fato represente
isso, o que esta função não assume nem verifica. Deliberado: o UCI
Diabetes 130-US Hospitals (o caso de uso motivador) não tem datas
reais, só `encounter_id` como proxy de ordem cronológica.

Desenho: a PRIMEIRA observação de cada paciente é usada como fonte de
covariáveis de baseline e NUNCA conta como tempo em risco — consistente
com o Contrato de Temporalidade (5.7): nenhuma característica usada
para estratificar/fenotipar um paciente pode ser calculada olhando
para observações futuras do mesmo paciente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from ..core import Cohort, Observation

__all__ = ["DiscreteTimeToEventDataset", "build_discrete_time_to_event"]


@dataclass
class DiscreteTimeToEventDataset:
    df: pd.DataFrame
    time_is_ordinal: bool = True
    n_excluded_single_observation: int = 0
    n_included: int = 0


def build_discrete_time_to_event(
    cohort: Cohort,
    event_fn: Callable[[Observation], bool],
    covariates: Optional[dict[str, Callable[[Observation], object]]] = None,
) -> DiscreteTimeToEventDataset:
    """
    Para cada paciente com >=2 observações: a 1ª observação é baseline
    (fonte de covariáveis via `covariates`, NUNCA de tempo em risco);
    a partir da 2ª, caminha em ordem procurando a primeira observação
    em que `event_fn(observation)` é True — essa é a `duration`
    (índice, 1-based, contando a partir da 2ª observação) com `event=1`.

    Se nenhuma observação subsequente satisfizer `event_fn`, o paciente
    é CENSURADO em `duration=len(observations)-1`, `event=0` — nunca
    observamos o evento dentro da janela disponível, o que não é
    evidência de que o paciente nunca teria o evento (censura à
    direita genuína, não "paciente confirmado saudável").

    `covariates`: dict nome -> função(observation_baseline) -> valor,
    aplicada SÓ à 1ª observação de cada paciente — nunca à observação
    onde o evento ocorreu nem a qualquer uma posterior à baseline.

    Pacientes com só 1 observação são excluídos (não há risco
    subsequente a observar) — contados em `n_excluded_single_observation`,
    nunca descartados silenciosamente sem registro.
    """
    linhas = []
    n_excluidos = 0
    covariates = covariates or {}

    for sid, system in cohort.systems.items():
        obs = system.observations
        if len(obs) < 2:
            n_excluidos += 1
            continue

        baseline = obs[0]
        linha: dict = {"system_id": sid}
        for nome, fn in covariates.items():
            linha[nome] = fn(baseline)

        duration = None
        event = 0
        for i, o in enumerate(obs[1:], start=1):
            if event_fn(o):
                duration = i
                event = 1
                break
        if duration is None:
            duration = len(obs) - 1
            event = 0

        linha["duration"] = duration
        linha["event"] = event
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    return DiscreteTimeToEventDataset(df=df, n_excluded_single_observation=n_excluidos, n_included=len(df))
