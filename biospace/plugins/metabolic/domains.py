"""
biospace.plugins.metabolic.domains
=====================================

6 domínios semânticos do sistema metabólico — agrupados por significado
fisiológico (Contrato: SemanticDomain agrupa por semântica, não por
estrutura matemática), mesma disciplina do plugin sleep.

Nenhum destes domínios conhece "diabetes". GlycemicDomain mede
homeostase glicêmica; RenalDomain mede função renal; nenhum deles
pergunta ou assume qual doença o paciente tem. Interpretações clínicas
específicas (diabetes, síndrome metabólica, ...) vivem em
`interpretations.py`, nunca aqui.
"""

from __future__ import annotations

from typing import Optional

from biospace.core import Feature, Measurement, Observable, SemanticDomain

from .observables import (
    CircunferenciaAbdominalObservable,
    ColesterolTotalObservable,
    CreatininaObservable,
    FrequenciaCardiacaObservable,
    GlicemiaJejumObservable,
    HbA1cObservable,
    HdlObservable,
    IdadeObservable,
    ImcObservable,
    PressaoDiastolicaObservable,
    PressaoSistolicaObservable,
    SexoObservable,
    TaxaFiltracaoGlomerularObservable,
    TrigliceridiosObservable,
)
from .reference import Reference, zscore_features

__all__ = [
    "GlycemicDomain",
    "AnthropometricDomain",
    "CardiovascularDomain",
    "RenalDomain",
    "LipidDomain",
    "ComorbidityDomain",
    "TreatmentDomain",
]

_COMORBIDADES = ["hipertensao", "retinopatia", "neuropatia", "doenca_cardiovascular"]
_TRATAMENTOS = ["metformina", "insulina"]


class _FlagObservable(Observable):
    def __init__(self, key: str):
        self.key = key


class _ReferenceDomain(SemanticDomain):
    """Base comum: domínios codificados por z-score contra uma Reference populacional."""

    _keys: list[str] = []
    _invert: set[str] = set()

    def __init__(self, reference: Optional[Reference] = None):
        super().__init__(self._make_observables())
        self.reference: Reference = reference or {}

    def _make_observables(self):
        raise NotImplementedError

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        return zscore_features(measurements, self._keys, self.reference, invert=self._invert)


class GlycemicDomain(_ReferenceDomain):
    """Homeostase glicêmica — glicemia de jejum e HbA1c. Não é 'o domínio do diabetes': mede um processo fisiológico presente em qualquer paciente, diabético ou não."""

    name = "glycemic"
    description = "Homeostase glicêmica — glicemia de jejum e HbA1c"
    _keys = ["glicemia_jejum_mg_dl", "hba1c_pct"]

    def _make_observables(self):
        return [GlicemiaJejumObservable(), HbA1cObservable()]


class AnthropometricDomain(_ReferenceDomain):
    name = "anthropometric"
    description = "Idade, sexo, IMC, circunferência abdominal (adiposidade central)"
    _keys = ["idade", "sexo", "imc", "circunferencia_abdominal_cm"]

    def _make_observables(self):
        return [IdadeObservable(), SexoObservable(), ImcObservable(), CircunferenciaAbdominalObservable()]


class CardiovascularDomain(_ReferenceDomain):
    name = "cardiovascular"
    description = "Pressão arterial e frequência cardíaca de repouso"
    _keys = ["pressao_sistolica_mmhg", "pressao_diastolica_mmhg", "fc_repouso_bpm"]

    def _make_observables(self):
        return [PressaoSistolicaObservable(), PressaoDiastolicaObservable(), FrequenciaCardiacaObservable()]


class RenalDomain(_ReferenceDomain):
    name = "renal"
    description = "Função renal — creatinina (maior=pior) e eGFR (menor=pior, sinal invertido)"
    _keys = ["creatinina_mg_dl", "taxa_filtracao_glomerular"]
    _invert = {"taxa_filtracao_glomerular"}

    def _make_observables(self):
        return [CreatininaObservable(), TaxaFiltracaoGlomerularObservable()]


class LipidDomain(_ReferenceDomain):
    """Perfil lipídico — colesterol total, HDL (maior=melhor, invertido) e triglicerídeos. Triglicerídeos esparso por desenho (só subamostra em jejum no NHANES), tratado como qualquer outra ausência via peso de completude."""

    name = "lipid"
    description = "Perfil lipídico — colesterol total, HDL (invertido) e triglicerídeos"
    _keys = ["colesterol_total_mg_dl", "hdl_mg_dl", "trigliceridios_mg_dl"]
    _invert = {"hdl_mg_dl"}

    def _make_observables(self):
        return [ColesterolTotalObservable(), HdlObservable(), TrigliceridiosObservable()]


class ComorbidityDomain(SemanticDomain):
    """Flags binários (0/1) — presente/ausente, não normalizado por z-score (não faz sentido para binários). Comorbidades tipicamente associadas à disfunção metabólica crônica, não exclusivas de nenhuma doença específica."""

    name = "comorbidity"
    description = "Comorbidades associadas à disfunção metabólica crônica (hipertensão, retinopatia, neuropatia, DCV)"

    def __init__(self):
        super().__init__([_FlagObservable(k) for k in _COMORBIDADES])

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key in _COMORBIDADES:
            m = measurements.get(key)
            v = float(m.value) if (m is not None and not m.is_missing()) else 0.0
            features.append(Feature(name=key, value=v, raw_value=v, provenance=(key,)))
        return features


class TreatmentDomain(SemanticDomain):
    """Flags binários de tratamento farmacológico em curso (metformina, insulina)."""

    name = "treatment"
    description = "Tratamento farmacológico em curso"

    def __init__(self):
        super().__init__([_FlagObservable(k) for k in _TRATAMENTOS])

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key in _TRATAMENTOS:
            m = measurements.get(key)
            v = float(m.value) if (m is not None and not m.is_missing()) else 0.0
            features.append(Feature(name=key, value=v, raw_value=v, provenance=(key,)))
        return features
