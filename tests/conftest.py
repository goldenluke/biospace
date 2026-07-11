"""
tests.conftest
================

Fixtures compartilhados. Deliberadamente NÃO dependem do Excel real
(que não está disponível em CI/execução automatizada) — tudo aqui é
sintético, mas usa os MESMOS nomes de campo e faixas de valor plausíveis
usadas em `demo_sleep.py`, para que os testes reflitam o uso real do
plugin sleep.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `pytest` puro (o comando de console, sem `-m`) NÃO adiciona o diretório
# atual ao sys.path automaticamente — só `python3 -m pytest` faz isso.
# Sem esta linha, `pytest` (o jeito padrão de invocar, inclusive em CI)
# falha com "ModuleNotFoundError: No module named 'biospace'" na coleta
# de TODO teste, mesmo que `python3 -m pytest` funcione perfeitamente no
# mesmo diretório — achado real ao montar o workflow de CI.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


def make_exam_values(**overrides) -> dict:
    """Valores de exame padrão (perfil 'moderado' plausível), sobrescrevíveis via kwargs."""
    base = {
        "idade": 50.0,
        "peso_kg": 85.0,
        "altura_cm": 170.0,
        "imc": 29.0,
        "ido": 20.0,
        "ido_sono": 18.0,
        "no_de_dessaturacoes": 120.0,
        "tempo_total_de_ronco_min": 45.0,
        "tempo_em_ronco_baixo": 15.0,
        "tempo_em_ronco_medio": 20.0,
        "tempo_em_ronco_alto": 10.0,
        "spo2_minima": 85.0,
        "spo2_media": 93.0,
        "spo2_maxima": 97.0,
        "tempo_spo2_90": 10.0,
        "carga_hipoxica_min_h": 25.0,
        "no_de_eventos_de_hipoxemia": 90.0,
        "tempo_total_em_hipoxemia_min": 30.0,
        "tempo_para_dormir_min": 20.0,
        "tempo_total_de_sono_min": 380.0,
        "tempo_acordado_pos_sono_min": 25.0,
        "eficiencia_do_sono": 85.0,
        "fc_minima_bpm": 58.0,
        "fc_media_bpm": 72.0,
        "fc_maxima_bpm": 105.0,
        "doencas": "",
        "sintomas": "",
        "tratamentos": "",
    }
    base.update(overrides)
    return base


@pytest.fixture
def exam_values_factory():
    return make_exam_values


@pytest.fixture
def sleep_representation():
    from biospace.plugins.sleep import SleepRepresentation

    return SleepRepresentation()


@pytest.fixture
def sleep_system_factory():
    """Factory: cria um SleepSystem novo e vazio (útil para check_temporality, que exige uma NOVA instância por chamada)."""
    from biospace.plugins.sleep import SleepSystem

    def _factory():
        return SleepSystem()

    return _factory


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    """
    Um DataFrame 'Excel-like' com 6 pacientes, 3 dos quais têm MÚLTIPLOS
    exames (para testar agrupamento por paciente e trajetórias) —
    replica em miniatura a estrutura real (355 pacientes, 296 com
    múltiplos exames) que expôs o bug original do loader.
    """
    t0 = datetime(2020, 1, 1)
    rows = []

    # 3 pacientes com um único exame
    for i, ido in enumerate([8.0, 22.0, 40.0]):
        row = make_exam_values(ido=ido, ido_sono=ido)
        row["paciente"] = f"PAC_{i:03d}"
        row["inicio"] = t0 + timedelta(days=i)
        rows.append(row)

    # 3 pacientes com múltiplos exames (progressão temporal real)
    for i in range(3, 6):
        n_exames = i  # 3, 4, 5 exames
        for j in range(n_exames):
            row = make_exam_values(ido=15.0 + j * 3, ido_sono=15.0 + j * 3)
            row["paciente"] = f"PAC_{i:03d}"
            row["inicio"] = t0 + timedelta(days=j * 60)
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def small_cohort(synthetic_dataframe):
    """Cohort + Representation construídos via load_from_dataframe (o mecanismo real de produção)."""
    from biospace.plugins.sleep import load_from_dataframe

    cohort, representation = load_from_dataframe(synthetic_dataframe)
    return cohort, representation
