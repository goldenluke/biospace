"""
components.filters
=====================

Filtros de sidebar aplicados sobre `display_df` (visualização apenas — a
Representation/Cohort subjacentes nunca são alteradas por um filtro).
Usa as mesmas `key=` em todas as páginas para que a seleção persista ao
navegar entre elas.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

_ORDEM_CLASSES = ["Normal", "Leve", "Moderada", "Grave"]


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    fenotipos = sorted(df["fenotipo"].dropna().unique().tolist())
    selected_fenotipos = st.sidebar.multiselect("Fenótipo", fenotipos, default=fenotipos, key="filtro_fenotipo")

    classes_presentes = [c for c in _ORDEM_CLASSES if c in df["classe_apneia"].dropna().unique()]
    selected_classes = st.sidebar.multiselect(
        "Classe de apneia (IDO)", classes_presentes, default=classes_presentes, key="filtro_classe_apneia"
    )

    generos = sorted(df["genero"].dropna().unique().tolist())
    selected_generos = st.sidebar.multiselect("Gênero", generos, default=generos, key="filtro_genero")

    idade_min, idade_max = int(df["idade"].min()), int(df["idade"].max())
    if idade_min < idade_max:
        idade_range = st.sidebar.slider(
            "Faixa etária", idade_min, idade_max, (idade_min, idade_max), key="filtro_idade"
        )
    else:
        idade_range = (idade_min, idade_max)

    filtered = df[
        df["fenotipo"].isin(selected_fenotipos)
        & df["classe_apneia"].isin(selected_classes)
        & df["genero"].isin(selected_generos)
        & df["idade"].between(idade_range[0], idade_range[1])
    ]

    st.sidebar.caption(f"{len(filtered)} de {len(df)} pacientes selecionados")
    return filtered
