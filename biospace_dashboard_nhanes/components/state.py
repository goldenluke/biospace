"""
components.state
===================

Acesso ao Pipeline guardado em st.session_state, mesmo padrão dos
outros dois dashboards.
"""

from __future__ import annotations

import streamlit as st

from .pipeline import Pipeline

STATE_KEY = "biospace_nhanes_pipeline"


def set_pipeline(pipeline: Pipeline, source_label: str) -> None:
    st.session_state[STATE_KEY] = pipeline
    st.session_state[f"{STATE_KEY}_source"] = source_label


def get_pipeline() -> Pipeline | None:
    return st.session_state.get(STATE_KEY)


def get_source_label() -> str | None:
    return st.session_state.get(f"{STATE_KEY}_source")


def clear_pipeline() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(STATE_KEY) or key.startswith("_nhanes"):
            st.session_state.pop(key, None)


def require_pipeline() -> Pipeline:
    pipeline = get_pipeline()
    if pipeline is None:
        st.warning("Nenhum dado carregado ainda. Volte à página **Início** e carregue a coorte NHANES.")
        st.stop()
    return pipeline
