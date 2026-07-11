"""
components.state
===================

Acesso ao Pipeline (Cohort/RepresentationSpace/Phenotypes) guardado em
st.session_state, compartilhado entre a página inicial e as páginas de
pages/. Mesmo padrão do dashboard de sleep.
"""

from __future__ import annotations

import streamlit as st

from .pipeline import Pipeline

STATE_KEY = "biospace_diabetes_pipeline"

_KNOWN_ANALYSIS_KEYS = [
    "_evo", "_stability_report", "_balance", "_effect", "_tratamento_label",
    "_prop_model", "_match_result", "_insulin_proxy",
]


def set_pipeline(pipeline: Pipeline, source_label: str) -> None:
    st.session_state[STATE_KEY] = pipeline
    st.session_state[f"{STATE_KEY}_source"] = source_label


def get_pipeline() -> Pipeline | None:
    return st.session_state.get(STATE_KEY)


def get_source_label() -> str | None:
    return st.session_state.get(f"{STATE_KEY}_source")


def clear_pipeline() -> None:
    """Remove o pipeline carregado e qualquer análise já rodada nas outras páginas."""
    for key in [STATE_KEY, f"{STATE_KEY}_source", *_KNOWN_ANALYSIS_KEYS]:
        st.session_state.pop(key, None)


def require_pipeline() -> Pipeline:
    pipeline = get_pipeline()
    if pipeline is None:
        st.warning("Nenhum dado carregado ainda. Volte à página **Início** e gere uma coorte sintética.")
        st.stop()
    return pipeline
