"""
biospace.plugins.metabolic.loader
====================================

load_from_dataframe: agrupa linhas por paciente (`id_column`) e ordena
por data (`order_column`), produzindo UM MetabolicSystem por paciente
com trajetória completa — nunca um sistema por linha (mesma lógica de
`biospace.plugins.sleep.loader`, reimplementada aqui de forma
independente).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from biospace.core import Cohort, Observation

from .reference import fit_reference
from .representation import MetabolicRepresentation
from .system import MetabolicSystem

__all__ = ["load_from_dataframe"]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not _is_missing(v)}


def load_from_dataframe(
    df: pd.DataFrame,
    representation: Optional[MetabolicRepresentation] = None,
    timestamp: Optional[datetime] = None,
    id_column: str = "paciente",
    order_column: str = "data_exame",
) -> tuple[Cohort, MetabolicRepresentation]:
    """Ver docstring do módulo. Espelha `sleep.loader.load_from_dataframe`."""
    records = [_clean_record(r) for r in df.to_dict(orient="records")]
    records = [r for r in records if r.get(id_column)]

    if representation is None:
        reference = fit_reference(records)
        representation = MetabolicRepresentation(reference=reference)

    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, record in enumerate(records):
        groups.setdefault(str(record[id_column]), []).append((i, record))

    def sort_key(item: tuple[int, dict]) -> tuple[int, Any]:
        row_index, record = item
        order_value = record.get(order_column)
        return (1, row_index) if order_value is None else (0, order_value)

    cohort = Cohort()
    default_ts = timestamp or datetime.now()

    for patient_label, indexed_records in groups.items():
        indexed_records.sort(key=sort_key)
        system = MetabolicSystem(identifier=f"metabolic_{patient_label}")
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

            system.observe(Observation(timestamp=exam_ts, source="exame_metabolico", values=record))
            cohort.update(system, representation, timestamp=exam_ts)

    cohort.loader_report = {"n_rows_discarded": len(df) - len(records), "n_patients": len(groups)}
    return cohort, representation
