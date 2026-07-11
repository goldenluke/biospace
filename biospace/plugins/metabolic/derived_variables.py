"""
biospace.plugins.metabolic.derived_variables
===============================================

3 DerivedVariable concretas, todas sobre a série de HbA1c ao longo da
trajetória (a mesma Feature, 3 formas diferentes de resumir sua
história):

- `HbA1cSlopeVariable`: tendência linear (%/ano) — regressão linear
  simples sobre (dias, valor), não apenas "primeiro menos último".
- `HbA1cVariabilityVariable`: desvio padrão da série — instabilidade de
  controle, não sua tendência.
- `GlycemicBurdenVariable`: carga hiperglicêmica acumulada — reaproveita
  EXATAMENTE o mecanismo já usado (e validado por
  `test_renal_decline_correlates_with_chronic_glycemic_exposure`) no
  gerador sintético de `plugins/diabetes/synthetic.py` (soma do excesso
  de HbA1c acima de 7.0% a cada ponto da trajetória) — não um novo
  mecanismo inventado aqui, o mesmo já testado em outro contexto.

Todas declaram `process="glucose_homeostasis"` (mesmo processo da
Feature de origem, `hba1c_pct` — ver `processes.py`), demonstrando que
uma variável derivada herda naturalmente o processo fisiológico da
Feature que resume.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from biospace.core import DerivedVariable, Feature, Trajectory

__all__ = ["HbA1cSlopeVariable", "HbA1cVariabilityVariable", "GlycemicBurdenVariable"]


class HbA1cSlopeVariable(DerivedVariable):
    name = "hba1c_slope_per_year"
    description = "Tendência linear de HbA1c ao longo da trajetória (%/ano) — regressão linear simples sobre (dias, valor)."
    process = "glucose_homeostasis"
    domain_name = "glycemic"
    feature_name = "hba1c_pct"
    min_points = 2

    def compute(self, trajectory: Trajectory) -> Optional[Feature]:
        pontos = self.series(trajectory)
        if len(pontos) < self.min_points:
            return None
        dias = np.array([p[0] for p in pontos])
        valores = np.array([p[1] for p in pontos])
        if np.ptp(dias) == 0:
            return None  # todos os pontos no mesmo instante -- slope indefinido, nao zero
        slope_por_dia, _ = np.polyfit(dias, valores, 1)
        slope_por_ano = float(slope_por_dia * 365.25)
        return Feature(name=self.name, value=slope_por_ano, raw_value=slope_por_ano, provenance=("hba1c_pct",))


class HbA1cVariabilityVariable(DerivedVariable):
    name = "hba1c_variability"
    description = "Desvio padrão de HbA1c ao longo da trajetória -- instabilidade de controle, distinta de tendência."
    process = "glucose_homeostasis"
    domain_name = "glycemic"
    feature_name = "hba1c_pct"
    min_points = 2

    def compute(self, trajectory: Trajectory) -> Optional[Feature]:
        pontos = self.series(trajectory)
        if len(pontos) < self.min_points:
            return None
        valores = np.array([p[1] for p in pontos])
        desvio = float(np.std(valores, ddof=1))
        return Feature(name=self.name, value=desvio, raw_value=desvio, provenance=("hba1c_pct",))


class GlycemicBurdenVariable(DerivedVariable):
    """
    Reaproveita EXATAMENTE o mecanismo de `plugins/diabetes/synthetic.py`
    (soma do excesso de HbA1c acima de 7.0% a cada ponto) -- o mesmo
    mecanismo cuja correlação com declínio renal já foi validada em
    `tests/test_diabetes_plugin.py::test_renal_decline_correlates_with_chronic_glycemic_exposure`.
    """

    name = "glycemic_burden"
    description = "Carga hiperglicêmica acumulada -- soma do excesso de HbA1c acima de 7,0% em cada ponto da trajetória."
    process = "glucose_homeostasis"
    domain_name = "glycemic"
    feature_name = "hba1c_pct"
    min_points = 1  # burden faz sentido ate' com 1 ponto (a diferenca e' que nao cresce mais)
    _LIMIAR = 7.0

    def compute(self, trajectory: Trajectory) -> Optional[Feature]:
        pontos = self.series(trajectory)
        if len(pontos) < self.min_points:
            return None
        carga = sum(max(0.0, valor - self._LIMIAR) for _, valor in pontos)
        return Feature(name=self.name, value=carga, raw_value=carga, provenance=("hba1c_pct",))
