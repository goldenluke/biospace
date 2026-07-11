"""
components.charts
====================

Helpers de gráficos reutilizáveis entre páginas.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CLASSE_CONTROLE_CORES = {
    "Controlado": "#00C853",
    "Moderado": "#FFD600",
    "Descompensado": "#FF0000",
}

ORDEM_CLASSE_CONTROLE = ["Controlado", "Moderado", "Descompensado"]


def histograma_faixas(
    df: pd.DataFrame,
    coluna: str,
    bins: int,
    titulo_x: str,
    faixas: list[tuple[float | None, float | None, str, str]],
) -> go.Figure:
    """Histograma colorido por faixas clínicas."""
    serie = df[coluna].dropna()
    fig = go.Figure()
    for lo, hi, color, label in faixas:
        mask = pd.Series(True, index=serie.index)
        if lo is not None:
            mask &= serie >= lo
        if hi is not None:
            mask &= serie < hi
        fig.add_trace(go.Histogram(x=serie[mask], nbinsx=bins, name=label, marker_color=color, opacity=0.85))
    fig.update_layout(barmode="overlay", xaxis_title=titulo_x, yaxis_title="Nº de pacientes", legend_title="Faixa", margin=dict(t=30, b=10))
    return fig


def apply_default_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=10, l=10, r=10))
    return fig
