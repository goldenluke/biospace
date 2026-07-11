"""
components.charts
====================

Helpers de gráficos reutilizáveis.
"""

from __future__ import annotations

import plotly.graph_objects as go

STATUS_CORES = {"normal": "#00C853", "pre_diabetes": "#FFD600", "diabetes": "#FF0000", "indeterminado": "#9E9E9E"}
ORDEM_STATUS = ["normal", "pre_diabetes", "diabetes", "indeterminado"]


def apply_default_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig
