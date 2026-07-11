"""
biospace.plugins.sleep.loader
================================

Constrói uma Cohort de SleepSystem a partir de uma planilha real de exames,
migrando a lógica de 01_exploracao.py -> 02_descobrir_cabecalho.py ->
03_ingestao.py -> 04_preprocessamento.py, mas produzindo objetos do
meta-modelo (BiologicalSystem / Observation / Cohort) em vez de um
DataFrame monolítico com features derivadas soltas.

Nota: `pandas` é usado aqui deliberadamente — é um plugin de I/O
específico de doença, não o núcleo (`biospace.core` nunca importa pandas).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from biospace.core import Cohort, Observation

from .domains import Reference, fit_reference
from .representation import SleepRepresentation
from .system import SleepSystem

__all__ = ["normalize_column", "load_from_dataframe", "load_from_excel"]


def normalize_column(col: Any) -> str:
    """
    Idêntico a `normalize_column()` de 04_preprocessamento.py: remove
    acentos, força minúsculas e colapsa qualquer sequência não
    alfanumérica em um único underscore.
    """
    col = str(col)
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove valores nulos/NaN, preservando o restante intacto (inclusive strings vazias -> descartadas)."""
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        clean[key] = value
    return clean


def load_from_dataframe(
    df: pd.DataFrame,
    representation: Optional[SleepRepresentation] = None,
    timestamp: Optional[datetime] = None,
    id_column: str = "paciente",
    order_column: str = "inicio",
) -> tuple[Cohort, SleepRepresentation]:
    """
    Constrói uma Cohort a partir de um DataFrame onde cada linha é UM EXAME
    (não necessariamente um paciente distinto — a mesma planilha real pode
    ter dezenas de exames repetidos para o mesmo paciente ao longo do
    tempo). As linhas são agrupadas por `id_column` e ordenadas por
    `order_column` (data de início do exame), produzindo UM SleepSystem
    por paciente com sua trajetória completa — nunca um sistema por linha
    (Seção 9.3 da teoria: cada nova observação atualiza a trajetória,
    jamais cria um novo paciente).

    Se `representation` não for informada, ajusta (fit) as estatísticas de
    referência sobre todas as linhas (todos os exames, de todos os
    pacientes) — equivalente ao `StandardScaler.fit()` de
    07_clusterizacao.py, mas produzindo uma referência explícita,
    reutilizável e auditável (Contrato 5.8). Note que isso pondera cada
    EXAME igualmente, não cada PACIENTE — um paciente com muitos exames de
    acompanhamento contribui mais para a referência populacional do que um
    paciente com um único exame. Isso é uma escolha de projeto, não um bug;
    se preferir ponderar por paciente, ajuste `reference` externamente
    (ex.: usando apenas a última linha de cada paciente) antes de chamar
    esta função.
    """
    df = df.copy()
    df.columns = [normalize_column(c) for c in df.columns]
    records = [_clean_record(r) for r in df.to_dict(orient="records")]

    # Descarta linhas sem identificação de paciente (comum em planilhas
    # Excel reais: linha em branco residual no final do arquivo).
    n_exams_before = len(records)
    records = [r for r in records if r.get(id_column)]
    n_discarded = n_exams_before - len(records)

    if representation is None:
        reference: Reference = fit_reference(records)
        representation = SleepRepresentation(reference=reference)

    # Agrupa por paciente, preservando a ordem de primeira aparição.
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, record in enumerate(records):
        patient_label = str(record[id_column])
        groups.setdefault(patient_label, []).append((i, record))

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, Any]:
        row_index, record = item
        order_value = record.get(order_column)
        if order_value is None:
            return (1, row_index)
        return (0, order_value)

    cohort = Cohort()
    default_ts = timestamp or datetime.now()
    n_multi_exam_patients = 0

    for patient_label, indexed_records in groups.items():
        indexed_records.sort(key=sort_key)
        if len(indexed_records) > 1:
            n_multi_exam_patients += 1

        system = SleepSystem(identifier=f"sleep_{normalize_column(patient_label)}")
        system.metadata = {"paciente_original": patient_label, "n_exames": len(indexed_records)}

        for _, record in indexed_records:
            order_value = record.get(order_column)
            if isinstance(order_value, datetime):
                exam_ts = order_value
            elif order_value is not None:
                try:
                    exam_ts = pd.Timestamp(order_value).to_pydatetime()
                except (ValueError, TypeError):
                    exam_ts = default_ts
            else:
                exam_ts = default_ts

            system.observe(Observation(timestamp=exam_ts, source="exame_completo", values=record))
            cohort.update(system, representation, timestamp=exam_ts)

    # Relatório de qualidade de carga, anexado à Cohort para auditoria
    # (não impresso automaticamente — a decisão de reportar é do chamador).
    cohort.loader_report = {
        "n_exams_loaded": len(records),
        "n_rows_discarded": n_discarded,
        "n_patients": len(groups),
        "n_patients_with_multiple_exams": n_multi_exam_patients,
    }

    return cohort, representation


def load_from_excel(
    path: str,
    header: int = 1,
    representation: Optional[SleepRepresentation] = None,
) -> tuple[Cohort, SleepRepresentation]:
    """
    Equivalente direto de 03_ingestao.py + 04_preprocessamento.py
    (mesmo `header=1`, pois a planilha original tem uma linha de título
    acima do cabeçalho real — ver 02_descobrir_cabecalho.py), produzindo
    uma Cohort em vez de um parquet solto.
    """
    df = pd.read_excel(path, header=header)
    return load_from_dataframe(df, representation=representation)
