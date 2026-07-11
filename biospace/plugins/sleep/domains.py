"""
biospace.plugins.sleep.domains
=================================

Domínios semânticos reais de SAOS, migrados da pipeline legada em pandas
(04_preprocessamento.py, 06_features_texto.py, 07_clusterizacao.py).

Cada domínio reproduz exatamente a mesma regra clínica que existia
espalhada em scripts de pré-processamento — thresholds, mapas de texto,
sinais invertidos — só que agora encapsulada em um φ_i determinístico que
consome `Measurement`s (com proveniência) e produz `Feature`s (com valor
bruto, z-score, peso e proveniência), satisfazendo os contratos formais da
Seção 5 da teoria de ponta a ponta — não apenas na composição dos
domínios, mas em cada coordenada individual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from biospace.core import Feature, Measurement, SemanticDomain

from .clinical_maps import MAPA_DOENCAS, MAPA_SINTOMAS, MAPA_TRATAMENTOS
from .observables import (
    AlturaCm,
    CargaHipoxicaMinH,
    DoencasTexto,
    EficienciaDoSono,
    FcMaximaBpm,
    FcMediaBpm,
    FcMinimaBpm,
    Idade,
    Ido,
    IdoSono,
    Imc,
    NoDeDessaturacoes,
    NoDeEventosDeHipoxemia,
    PesoKg,
    SintomasTexto,
    Spo2Maxima,
    Spo2Media,
    Spo2Minima,
    TempoAcordadoPosSonoMin,
    TempoEmRoncoAlto,
    TempoEmRoncoBaixo,
    TempoEmRoncoMedio,
    TempoParaDormirMin,
    TempoSpo290,
    TempoTotalDeRoncoMin,
    TempoTotalDeSonoMin,
    TempoTotalEmHipoxemiaMin,
    TratamentosTexto,
)

__all__ = [
    "classificar_apneia",
    "classificar_hipoxemia",
    "fit_reference",
    "FieldStats",
    "Reference",
    "AnthropometricDomain",
    "ApneaDomain",
    "HypoxiaDomain",
    "SleepArchitectureDomain",
    "CardiovascularDomain",
    "ComorbidityDomain",
    "SymptomsDomain",
    "TreatmentDomain",
]

Reference = dict[str, "FieldStats"]


@dataclass(frozen=True)
class FieldStats:
    """Estatísticas de um campo numérico: média/desvio para z-score + completude populacional."""

    mean: float
    std: float
    completeness: float = 1.0  # fração da população com valor não-ausente, em [0, 1]


# =============================================================================
# Regras clínicas migradas literalmente de 04_preprocessamento.py
# =============================================================================
def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def classificar_apneia(ido: Optional[float]) -> Optional[str]:
    """
    Migrado literalmente de `classificar_apneia()` (04_preprocessamento.py).
    Classifica a severidade pelo IDO (Índice de Dessaturação de Oxigênio),
    usado neste dataset como proxy do IAH (não há canal de fluxo aéreo).
    """
    if _is_missing(ido):
        return None
    if ido < 5:
        return "Normal"
    if ido < 15:
        return "Leve"
    if ido < 30:
        return "Moderada"
    return "Grave"


def classificar_hipoxemia(spo2_minima: Optional[float]) -> Optional[str]:
    """Classificação de hipoxemia por SpO2 mínima (usada nas features 'hipoxemia_grave/moderada')."""
    if _is_missing(spo2_minima):
        return None
    if spo2_minima < 80:
        return "Grave"
    if spo2_minima < 90:
        return "Moderada"
    return "Normal"


# =============================================================================
# Estatísticas de referência para z-score
# =============================================================================
# Valores de fallback ilustrativos (completeness=1.0), usados apenas se
# fit_reference() não for chamado sobre os dados reais (o loader sempre
# ajusta a referência real, incluindo a completude por campo).
_RAW_DEFAULTS: dict[str, tuple[float, float]] = {
    "idade": (50.0, 14.0), "peso_kg": (80.0, 18.0), "altura_cm": (168.0, 9.0), "imc": (28.0, 5.0),
    "ido": (18.0, 15.0), "ido_sono": (18.0, 15.0), "no_de_dessaturacoes": (120.0, 100.0),
    "tempo_total_de_ronco_min": (60.0, 50.0),
    "tempo_em_ronco_baixo": (20.0, 20.0), "tempo_em_ronco_medio": (20.0, 20.0), "tempo_em_ronco_alto": (10.0, 15.0),
    "spo2_minima": (85.0, 8.0), "spo2_media": (94.0, 3.0), "spo2_maxima": (97.0, 2.0),
    "tempo_spo2_90": (30.0, 40.0), "carga_hipoxica_min_h": (30.0, 25.0),
    "no_de_eventos_de_hipoxemia": (100.0, 90.0), "tempo_total_em_hipoxemia_min": (40.0, 40.0),
    "tempo_para_dormir_min": (20.0, 15.0), "tempo_total_de_sono_min": (360.0, 60.0),
    "tempo_acordado_pos_sono_min": (30.0, 25.0), "eficiencia_do_sono": (85.0, 8.0),
    "fc_minima_bpm": (55.0, 8.0), "fc_media_bpm": (70.0, 10.0), "fc_maxima_bpm": (100.0, 15.0),
    "amplitude_fc": (40.0, 15.0),
}
_DEFAULT_REFERENCE: Reference = {k: FieldStats(m, s, 1.0) for k, (m, s) in _RAW_DEFAULTS.items()}


def fit_reference(raw_records: list[dict[str, Any]], keys: Optional[list[str]] = None) -> Reference:
    """
    Ajusta (média, desvio, completude) diretamente sobre uma população de
    registros brutos — equivalente ao `StandardScaler.fit()` de
    07_clusterizacao.py, mas produzindo uma referência explícita e
    reutilizável (Contrato 5.8) que TAMBÉM registra, por campo, a fração
    da população que de fato tinha aquele valor observado.

    Essa completude é usada por `_zscore_features()` para ponderar
    automaticamente a contribuição de cada eixo: campos raramente
    presentes pesam menos na geometria do espaço, em vez de serem
    tratados como se tivessem a mesma confiabilidade de um campo presente
    em 99% dos casos.
    """
    keys = keys or [k for k in _DEFAULT_REFERENCE if k != "amplitude_fc"]
    n_total = len(raw_records) or 1
    reference: Reference = {}
    for key in keys:
        values = np.array(
            [r[key] for r in raw_records if key in r and not _is_missing(r[key])],
            dtype=float,
        )
        completeness = len(values) / n_total
        if len(values) < 2:
            base = _DEFAULT_REFERENCE.get(key, FieldStats(0.0, 1.0, 1.0))
            reference[key] = FieldStats(base.mean, base.std, completeness)
            continue
        std = float(values.std())
        reference[key] = FieldStats(float(values.mean()), std if std > 1e-9 else 1.0, completeness)

    if "fc_maxima_bpm" in reference and "fc_minima_bpm" in reference:
        amplitude_pairs = [(r.get("fc_maxima_bpm"), r.get("fc_minima_bpm")) for r in raw_records]
        amplitudes = np.array(
            [a - b for a, b in amplitude_pairs if not _is_missing(a) and not _is_missing(b)],
            dtype=float,
        )
        completeness = len(amplitudes) / n_total
        if len(amplitudes) >= 2:
            std = float(amplitudes.std())
            reference["amplitude_fc"] = FieldStats(float(amplitudes.mean()), std if std > 1e-9 else 1.0, completeness)
        else:
            base = _DEFAULT_REFERENCE["amplitude_fc"]
            reference["amplitude_fc"] = FieldStats(base.mean, base.std, completeness)

    return reference


def _completeness_weight(completeness: float, exclude_below: float) -> float:
    """
    Peso aplicado a um eixo, proporcional à completude do campo na
    população. Campos com completude abaixo de `exclude_below` são
    excluídos por completo (peso 0) — funcionalmente equivalentes a não
    fazerem parte da Representation, mas sem quebrar a consistência
    dimensional do vetor (Princípio da Composicionalidade).
    """
    if completeness < exclude_below:
        return 0.0
    return completeness


def _zscore_features(
    measurements: dict[str, Measurement],
    keys: list[str],
    reference: Reference,
    missing_counts: Optional[dict[str, int]] = None,
    exclude_below: float = 0.05,
) -> list[Feature]:
    """
    Constrói uma Feature por chave em `keys`, com z-score contra a
    referência populacional, ponderado pela completude do campo.

    Dados clínicos reais têm ausência estrutural (nem todo exame calcula
    todos os índices — ex.: 'no_de_eventos_de_hipoxemia' está ausente em
    ~70% desta planilha real). Em vez de falhar (violando o Contrato 5.1
    de rastreabilidade só para dizer "não sei"), valores ausentes são
    imputados por z=0 ("nem acima, nem abaixo do esperado"), e a
    coordenada inteira é multiplicada pelo peso de completude do campo.
    Cada Feature carrega seu próprio `raw_value`, `z_score`, `weight`,
    `is_missing`/`is_excluded` e `provenance` — auditável individualmente,
    sem depender de um contador solto por fora.

    PROPAGAÇÃO DE INCERTEZA (observações probabilísticas — ver
    `core.distribution`): se a Measurement carregar uma Distribution
    (`measurement.uncertainty > 0`), a incerteza se propaga
    algebricamente através da transformação linear z = (raw-mean)/std:
    como z-score e peso são ambos lineares, sigma_final =
    (sigma_bruto/std_referencia) * peso — exato, não uma aproximação,
    porque a transformação é linear.
    """
    features: list[Feature] = []
    for key in keys:
        stats = reference.get(key, _DEFAULT_REFERENCE.get(key, FieldStats(0.0, 1.0, 1.0)))
        weight = _completeness_weight(stats.completeness, exclude_below)
        measurement = measurements.get(key)
        is_missing = measurement is None or measurement.is_missing()

        if is_missing:
            if missing_counts is not None:
                missing_counts[key] = missing_counts.get(key, 0) + 1
            features.append(
                Feature(
                    name=key, value=0.0, raw_value=None, z_score=0.0,
                    weight=weight, is_missing=True, is_excluded=(weight == 0.0),
                    provenance=(key,),
                )
            )
        else:
            raw = float(measurement.value)
            z = (raw - stats.mean) / stats.std
            uncertainty = (measurement.uncertainty / stats.std) * weight if measurement.uncertainty > 0 else None
            features.append(
                Feature(
                    name=key, value=z * weight, raw_value=raw, z_score=z,
                    weight=weight, is_missing=False, is_excluded=(weight == 0.0),
                    provenance=(key,), uncertainty=uncertainty,
                )
            )
    return features


class _ReferenceDomain(SemanticDomain):
    """Base comum para domínios codificados por z-score contra uma referência populacional (Contrato 5.8)."""

    _keys: list[str] = []

    def __init__(self, observables, reference: Optional[Reference] = None, exclude_below: float = 0.05):
        super().__init__(observables)
        self.reference: Reference = {**_DEFAULT_REFERENCE, **(reference or {})}
        self.missing_counts: dict[str, int] = {}
        self.exclude_below = exclude_below

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        return _zscore_features(measurements, self._keys, self.reference, self.missing_counts, self.exclude_below)

    def feature_weights(self) -> dict[str, float]:
        """Peso efetivo (por completude) aplicado a cada campo deste domínio — para auditoria/relatório."""
        return {
            key: _completeness_weight(self.reference.get(key, FieldStats(0.0, 1.0, 1.0)).completeness, self.exclude_below)
            for key in self._keys
        }


# =============================================================================
# Domínios
# =============================================================================
class AnthropometricDomain(_ReferenceDomain):
    name = "anthropometric"
    description = "Características antropométricas e demográficas"
    _keys = ["idade", "imc"]

    def __init__(self, reference: Optional[Reference] = None, exclude_below: float = 0.05):
        super().__init__([Idade(), Imc(), PesoKg(), AlturaCm()], reference, exclude_below)


class ApneaDomain(_ReferenceDomain):
    """
    Domínio de apneia. Migra a Seção '02_Apneia' do dashboard e a
    classificação `classificar_apneia()` — porém como eixos numéricos
    contínuos (z-score), não como categoria fixa. `classificar_apneia()`
    continua disponível separadamente para validação/relatório.
    """

    name = "apnea"
    description = "Índice de dessaturação (proxy de apneia) e ronco associado"
    _keys = ["ido", "ido_sono", "no_de_dessaturacoes", "tempo_total_de_ronco_min", "tempo_em_ronco_alto"]

    def __init__(self, reference: Optional[Reference] = None, exclude_below: float = 0.05):
        super().__init__(
            [Ido(), IdoSono(), NoDeDessaturacoes(),
             TempoTotalDeRoncoMin(), TempoEmRoncoBaixo(), TempoEmRoncoMedio(), TempoEmRoncoAlto()],
            reference, exclude_below,
        )


class HypoxiaDomain(_ReferenceDomain):
    """Domínio de hipoxemia — migra a Seção '03_Hipoxemia' do dashboard."""

    name = "hypoxia"
    description = "Carga de dessaturação de oxigênio"
    _keys = [
        "spo2_minima", "spo2_media", "spo2_maxima", "tempo_spo2_90",
        "carga_hipoxica_min_h", "no_de_eventos_de_hipoxemia", "tempo_total_em_hipoxemia_min",
    ]
    _inverted_keys = {"spo2_minima", "spo2_media", "spo2_maxima"}

    def __init__(self, reference: Optional[Reference] = None, exclude_below: float = 0.05):
        super().__init__(
            [Spo2Minima(), Spo2Media(), Spo2Maxima(), TempoSpo290(),
             CargaHipoxicaMinH(), NoDeEventosDeHipoxemia(), TempoTotalEmHipoxemiaMin()],
            reference, exclude_below,
        )

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = super().encode(measurements)
        # Inverte SpO2 (Contrato de orientação de eixo): assim, em todos os
        # eixos do domínio, "maior valor" passa a significar "mais grave" —
        # a mesma inversão manual feita em 07_clusterizacao.py
        # (`perfil_score["spo2_minima"] = 100 - perfil_score["spo2_minima"]`).
        for f in features:
            if f.name in self._inverted_keys:
                f.value = -f.value
                if f.z_score is not None:
                    f.z_score = -f.z_score
        return features


class SleepArchitectureDomain(_ReferenceDomain):
    name = "sleep_architecture"
    description = "Latência, duração e eficiência do sono"
    _keys = ["tempo_para_dormir_min", "tempo_total_de_sono_min", "tempo_acordado_pos_sono_min", "eficiencia_do_sono"]
    _inverted_keys = {"eficiencia_do_sono"}

    def __init__(self, reference: Optional[Reference] = None, exclude_below: float = 0.05):
        super().__init__(
            [TempoParaDormirMin(), TempoTotalDeSonoMin(), TempoAcordadoPosSonoMin(), EficienciaDoSono()],
            reference, exclude_below,
        )

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = super().encode(measurements)
        # eficiencia_do_sono: maior é melhor -> inverte para manter a
        # convenção "maior = mais grave" também neste domínio.
        for f in features:
            if f.name in self._inverted_keys:
                f.value = -f.value
                if f.z_score is not None:
                    f.z_score = -f.z_score
        return features


class CardiovascularDomain(_ReferenceDomain):
    """Migra a Seção '09_Frequencia_Cardiaca' do dashboard, incluindo `amplitude_fc`."""

    name = "cardiovascular"
    description = "Frequência cardíaca durante o sono"
    _keys = ["fc_minima_bpm", "fc_media_bpm", "fc_maxima_bpm"]

    def __init__(self, reference: Optional[Reference] = None, exclude_below: float = 0.05):
        super().__init__([FcMinimaBpm(), FcMediaBpm(), FcMaximaBpm()], reference, exclude_below)

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = super().encode(measurements)

        fc_max_m = measurements.get("fc_maxima_bpm")
        fc_min_m = measurements.get("fc_minima_bpm")
        stats = self.reference.get("amplitude_fc", _DEFAULT_REFERENCE["amplitude_fc"])
        weight = _completeness_weight(stats.completeness, self.exclude_below)

        is_missing = (
            fc_max_m is None or fc_min_m is None or fc_max_m.is_missing() or fc_min_m.is_missing()
        )
        if is_missing:
            self.missing_counts["amplitude_fc"] = self.missing_counts.get("amplitude_fc", 0) + 1
            features.append(
                Feature(name="amplitude_fc", value=0.0, raw_value=None, z_score=0.0,
                        weight=weight, is_missing=True, is_excluded=(weight == 0.0),
                        provenance=("fc_maxima_bpm", "fc_minima_bpm"))
            )
        else:
            amplitude = float(fc_max_m.value) - float(fc_min_m.value)
            z = (amplitude - stats.mean) / stats.std
            # Incerteza de uma DIFERENÇA de duas medições independentes:
            # sigma_diferenca = sqrt(sigma_max^2 + sigma_min^2) — soma em
            # quadratura, propagada depois pela mesma escala linear (std_ref, peso).
            unc_max, unc_min = fc_max_m.uncertainty, fc_min_m.uncertainty
            uncertainty = None
            if unc_max > 0 or unc_min > 0:
                sigma_amplitude = (unc_max**2 + unc_min**2) ** 0.5
                uncertainty = (sigma_amplitude / stats.std) * weight
            features.append(
                Feature(name="amplitude_fc", value=z * weight, raw_value=amplitude, z_score=z,
                        weight=weight, is_missing=False, is_excluded=(weight == 0.0),
                        provenance=("fc_maxima_bpm", "fc_minima_bpm"), uncertainty=uncertainty)
            )
        return features

    def feature_weights(self) -> dict[str, float]:
        weights = super().feature_weights()
        stats = self.reference.get("amplitude_fc", _DEFAULT_REFERENCE["amplitude_fc"])
        weights["amplitude_fc"] = _completeness_weight(stats.completeness, self.exclude_below)
        return weights


def _text_domain_labels(mapping: dict[str, str]) -> list[str]:
    """Códigos únicos preservando a ordem de primeira ocorrência (várias descrições -> mesmo código)."""
    seen: list[str] = []
    for code in mapping.values():
        if code not in seen:
            seen.append(code)
    return seen


def _text_domain_encode(
    source_key: str, measurements: dict[str, Measurement], mapping: dict[str, str], codes: list[str]
) -> list[Feature]:
    """
    Converte um texto livre separado por vírgulas (ex.: "Diabetes, Asma")
    em uma Feature binária por código clínico. Diferente dos domínios
    numéricos, aqui a "ausência" de texto é tratada como "nenhum item
    reportado" (0 em todos os códigos) — não como dado faltante — pois é
    exatamente essa a semântica de uma planilha clínica com campo vazio.
    """
    measurement = measurements.get(source_key)
    raw_text = measurement.value if measurement is not None else ""
    items = {item.strip() for item in str(raw_text or "").split(",") if item.strip()}

    features = []
    for code in codes:
        descriptions = [desc for desc, c in mapping.items() if c == code]
        present = any(desc in items for desc in descriptions)
        features.append(
            Feature(
                name=code, value=1.0 if present else 0.0, raw_value=1.0 if present else 0.0,
                z_score=None, weight=1.0, is_missing=False, is_excluded=False,
                provenance=(source_key,),
            )
        )
    return features


class ComorbidityDomain(SemanticDomain):
    """
    Domínio de comorbidades. Migra literalmente o parsing de texto livre de
    06_features_texto.py (coluna 'doencas' separada por vírgulas), trocando
    o loop `df.at[idx, coluna] = 1` por um φ_i determinístico.
    """

    name = "comorbidity"
    description = "Comorbidades reportadas (texto livre estruturado)"

    def __init__(self):
        super().__init__([DoencasTexto()])
        self._codes = _text_domain_labels(MAPA_DOENCAS)

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        return _text_domain_encode("doencas", measurements, MAPA_DOENCAS, self._codes)

    def labels(self) -> list[str]:
        return list(self._codes)


class SymptomsDomain(SemanticDomain):
    name = "symptoms"
    description = "Sintomas clínicos reportados (texto livre estruturado)"

    def __init__(self):
        super().__init__([SintomasTexto()])
        self._codes = _text_domain_labels(MAPA_SINTOMAS)

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        return _text_domain_encode("sintomas", measurements, MAPA_SINTOMAS, self._codes)

    def labels(self) -> list[str]:
        return list(self._codes)


class TreatmentDomain(SemanticDomain):
    name = "treatment"
    description = "Tratamentos em uso (texto livre estruturado)"

    def __init__(self):
        super().__init__([TratamentosTexto()])
        self._codes = _text_domain_labels(MAPA_TRATAMENTOS)

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        return _text_domain_encode("tratamentos", measurements, MAPA_TRATAMENTOS, self._codes)

    def labels(self) -> list[str]:
        return list(self._codes)
