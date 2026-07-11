"""
components.state
===================

Acesso ao Pipeline (Cohort/RepresentationSpace/Phenotypes) guardado em
st.session_state, compartilhado entre a página inicial e as páginas de
pages/. Cada página chama `require_pipeline()` no topo.
"""

from __future__ import annotations

import streamlit as st

from .pipeline import Pipeline

STATE_KEY = "biospace_pipeline"


def set_pipeline(pipeline: Pipeline, source_label: str) -> None:
    st.session_state[STATE_KEY] = pipeline
    st.session_state[f"{STATE_KEY}_source"] = source_label


def get_pipeline() -> Pipeline | None:
    return st.session_state.get(STATE_KEY)


def get_source_label() -> str | None:
    return st.session_state.get(f"{STATE_KEY}_source")


def get_source_kind() -> str | None:
    """'real', 'synthetic', ou None (nada carregado ainda) — a partir do source_label bruto."""
    label = get_source_label()
    if label is None:
        return None
    return "synthetic" if label.startswith("synthetic::") else "real"


def clear_pipeline() -> None:
    """Remove o pipeline carregado e qualquer estado dependente de páginas específicas — para trocar de dataset sem misturar resultados de análises antigas (ex.: um EvolutionOperator ajustado sobre a coorte anterior)."""
    known_analysis_keys = [
        "_evo", "_stability_report", "_balance", "_effect", "_tratamento_label",
        "_prop_model", "_match_result", "_scenario_results", "_ews_results", "_ews_indicador",
        "_infl_dom", "_frailty_dom", "_auto_dom", "_riem_geo", "_dyn_geo",
    ]
    for key in [STATE_KEY, f"{STATE_KEY}_source", *known_analysis_keys]:
        st.session_state.pop(key, None)


def require_pipeline() -> Pipeline:
    pipeline = get_pipeline()
    if pipeline is None:
        st.warning(
            "Nenhum dado carregado ainda. Volte à página **Início**, envie uma "
            "planilha (.xlsx) ou gere dados de demonstração."
        )
        st.stop()
    return pipeline
