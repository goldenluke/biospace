"""
components.charts
====================
"""

from __future__ import annotations

import plotly.graph_objects as go

READMIT_CORES = {"<30": "#FF0000", ">30": "#FFD600", "NO": "#00C853"}
ORDEM_READMIT = ["<30", ">30", "NO"]


def apply_default_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig
