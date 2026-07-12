"""
biospace.survival.models
===========================

Kaplan-Meier e Cox proportional hazards sobre o `DiscreteTimeToEventDataset`
de `discrete_time.py`. Envelope fino sobre `lifelines` — a lógica de
extração de tempo-até-evento (a parte específica deste projeto) já
aconteceu em `discrete_time.py`; aqui só ajustamos os modelos clássicos
e devolvemos relatórios auditáveis, no mesmo espírito dos outros
relatórios do projeto (`BalanceReport`, `StabilityReport`, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

__all__ = ["SurvivalByGroupReport", "kaplan_meier_by_group", "CoxModelReport", "fit_cox_model"]


@dataclass
class SurvivalByGroupReport:
    fitters: dict
    median_survival: dict
    logrank_p: float
    n_por_grupo: dict

    def summary(self) -> str:
        linhas = [f"Log-rank p={self.logrank_p:.2e}"]
        for grupo, n in self.n_por_grupo.items():
            mediana = self.median_survival.get(grupo)
            mediana_str = f"{mediana:.1f}" if mediana == mediana else "não atingida (>50% sobrevive além do observado)"
            linhas.append(f"  {grupo}: n={n}, mediana de sobrevivência={mediana_str}")
        return "\n".join(linhas)


def kaplan_meier_by_group(df: pd.DataFrame, group_col: str, duration_col: str = "duration", event_col: str = "event") -> SurvivalByGroupReport:
    """Ajusta uma curva de Kaplan-Meier por grupo (ex.: fenótipo de baseline) e testa diferença via log-rank multivariado (>=2 grupos)."""
    fitters = {}
    medianas = {}
    n_por_grupo = {}
    for grupo, sub in df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], event_observed=sub[event_col], label=str(grupo))
        fitters[grupo] = kmf
        medianas[grupo] = kmf.median_survival_time_
        n_por_grupo[grupo] = len(sub)

    resultado_logrank = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col])
    return SurvivalByGroupReport(fitters=fitters, median_survival=medianas, logrank_p=resultado_logrank.p_value, n_por_grupo=n_por_grupo)


@dataclass
class CoxModelReport:
    hazard_ratios: dict
    p_values: dict
    concordance_index: float
    summary_df: pd.DataFrame

    def summary(self) -> str:
        linhas = [f"Índice de concordância: {self.concordance_index:.3f}"]
        for cov, hr in self.hazard_ratios.items():
            linhas.append(f"  {cov}: HR={hr:.3f}, p={self.p_values[cov]:.2e}")
        return "\n".join(linhas)


def fit_cox_model(df: pd.DataFrame, covariate_cols: list[str], duration_col: str = "duration", event_col: str = "event") -> CoxModelReport:
    """Cox proportional hazards. `covariate_cols` devem já estar numéricas (one-hot para categóricas, feito pelo chamador -- este módulo não infere tipo)."""
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariate_cols], duration_col=duration_col, event_col=event_col)
    summary = cph.summary
    return CoxModelReport(
        hazard_ratios=summary["exp(coef)"].to_dict(),
        p_values=summary["p"].to_dict(),
        concordance_index=cph.concordance_index_,
        summary_df=summary,
    )
