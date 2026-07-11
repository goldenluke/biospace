"""
biospace.plugins.metabolic.reference
======================================

Mesma disciplina do plugin sleep (`biospace/plugins/sleep/domains.py`),
reimplementada aqui de forma independente — não compartilhada por
código entre plugins de propósito (cada plugin de doença é
independente; o núcleo não impõe nenhum padrão de normalização, cada
domínio decide o seu). O PADRÃO é o mesmo (z-score contra referência
populacional, ponderado por completude), a implementação é própria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from biospace.core import Feature, Measurement

__all__ = ["FieldStats", "Reference", "fit_reference", "zscore_features"]


@dataclass(frozen=True)
class FieldStats:
    mean: float
    std: float
    completeness: float  # fração de registros em que o campo estava presente


Reference = dict[str, FieldStats]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        import math

        return math.isnan(value)
    return False


def fit_reference(raw_records: list[dict[str, Any]], keys: Optional[list[str]] = None) -> Reference:
    """Ajusta (mean, std, completude) por campo, sobre uma população de registros brutos (dicts)."""
    if not raw_records:
        raise ValueError("fit_reference() precisa de pelo menos 1 registro.")

    all_keys = keys or sorted({k for r in raw_records for k in r.keys()})
    reference: Reference = {}
    n_total = len(raw_records)

    for key in all_keys:
        values = []
        for r in raw_records:
            if key not in r or _is_missing(r.get(key)):
                continue
            raw_value = r[key]
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                values.append(float(raw_value))
            # valores não numéricos (str, Timestamp, ...) são ignorados aqui — este domínio
            # não tenta normalizar campos de texto/data; ComorbidityDomain/TreatmentDomain
            # tratam seus próprios campos binários sem passar por fit_reference().
        completeness = len(values) / n_total if n_total else 0.0
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance**0.5 if variance > 1e-12 else 1.0
        elif len(values) == 1:
            mean, std = values[0], 1.0
        else:
            mean, std = 0.0, 1.0
        reference[key] = FieldStats(mean=mean, std=std, completeness=completeness)

    return reference


def _completeness_weight(completeness: float, exclude_below: float) -> float:
    return 0.0 if completeness < exclude_below else completeness


def zscore_features(
    measurements: dict[str, Measurement],
    keys: list[str],
    reference: Reference,
    invert: Optional[set[str]] = None,
    exclude_below: float = 0.05,
) -> list[Feature]:
    """
    Codifica `keys` como z-score contra `reference`, ponderado pela
    completude do campo na população de ajuste — campos ausentes viram
    z=0 (peso da completude), nunca uma falha. `invert`: campos onde um
    valor MAIOR é clinicamente MELHOR (ex.: eGFR) — o sinal do z-score é
    invertido para manter "maior = pior" consistente em toda a Feature.
    """
    invert = invert or set()
    features: list[Feature] = []
    for key in keys:
        stats = reference.get(key, FieldStats(0.0, 1.0, 1.0))
        weight = _completeness_weight(stats.completeness, exclude_below)
        measurement = measurements.get(key)
        is_missing = measurement is None or measurement.is_missing()

        if is_missing:
            features.append(
                Feature(name=key, value=0.0, raw_value=None, z_score=0.0, weight=weight, is_missing=True, is_excluded=(weight == 0.0), provenance=(key,))
            )
        else:
            raw = float(measurement.value)
            z = (raw - stats.mean) / stats.std
            if key in invert:
                z = -z
            uncertainty = (measurement.uncertainty / stats.std) * weight if measurement.uncertainty > 0 else None
            features.append(
                Feature(name=key, value=z * weight, raw_value=raw, z_score=z, weight=weight, is_missing=False, is_excluded=(weight == 0.0), provenance=(key,), uncertainty=uncertainty)
            )
    return features
