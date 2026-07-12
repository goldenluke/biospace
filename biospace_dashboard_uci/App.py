"""
App.py
======

Página inicial do dashboard BioSpace UCI Diabetes 130-US Hospitals —
dados REAIS (Strack et al., 2014), enviados pelo usuário.
"""

import components._bootstrap  # noqa: F401

import streamlit as st

from components.pipeline import run_pipeline
from components.state import clear_pipeline, get_pipeline, get_source_label, set_pipeline

st.set_page_config(page_title="BioSpace - UCI Diabetes", page_icon="🏥", layout="wide")

st.title("🏥 BioSpace — UCI Diabetes 130-US Hospitals (dados reais)")
st.caption(
    "Strack et al. (2014) — 101.766 encontros hospitalares, 71.518 pacientes, 130 hospitais dos EUA "
    "(1999-2008). Estrutura DIFERENTE do NHANES: sem lab contínuo (HbA1c/glicemia só categóricas, "
    "83-95% ausentes), sem antropometria/pressão/creatinina — mas utilização hospitalar 100% completa, "
    "e 23,4% dos pacientes com múltiplos encontros (trajetória real)."
)
st.info("🌐 Dado público, enviado manualmente pelo usuário. Arquivo esperado: `diabetic_data.csv`.")
st.warning(
    "⚠️ A base completa tem ~100 mil linhas — o carregamento inicial pode levar ~1-2 minutos. Use a "
    "amostra reduzida para exploração rápida, e a base completa para os números que aparecem no artigo."
)

st.divider()

st.subheader("Carregar coorte")
caminho = st.text_input("Caminho do arquivo CSV", value="/mnt/user-data/uploads/diabetic_data.csv")
modo = st.radio("Tamanho", ["Amostra rápida (5.000 encontros)", "Base completa (101.766 encontros, ~1-2 min)"], horizontal=False)
max_rows = 5000 if modo.startswith("Amostra") else None
n_clusters = st.slider("Nº de fenótipos (K-Means)", 2, 6, value=4)
incluir_diagnostico = st.radio(
    "Representação",
    ["Com diagnóstico ICD-9 (4 domínios, o default atual)", "Sem diagnóstico (3 domínios, o achado original de readmissão ~2x)"],
    horizontal=False,
)
include_diagnosis_category = incluir_diagnostico.startswith("Com")
st.caption(
    "🔬 **Achado real**: incluir o domínio de diagnóstico muda o que K-Means encontra como fenótipo dominante — "
    "de um organizado por utilização prévia (associação forte com readmissão) para um organizado mais por tempo "
    "de internação extremo (associação mais fraca). Nenhuma das duas está errada — capturam estruturas diferentes "
    "do mesmo dado real. Ver a página 'Fenótipos e Readmissão'."
)

if st.button("Carregar UCI Diabetes", type="primary"):
    cache_key = f"uci::{modo}::{n_clusters}::{include_diagnosis_category}"
    with st.spinner("Carregando CSV, construindo trajetórias por paciente, fenotipando... (pode levar minutos na base completa)"):
        try:
            pipeline = run_pipeline(caminho, max_rows=max_rows, n_clusters=n_clusters, include_diagnosis_category=include_diagnosis_category)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
    set_pipeline(pipeline, cache_key)
    st.success(f"Carregado: {pipeline.n_encontros} encontros, {pipeline.n_pacientes} pacientes.")
    st.rerun()

st.divider()

pipeline = get_pipeline()
if pipeline is None:
    st.info("Nenhum dado carregado ainda. Clique em 'Carregar UCI Diabetes' acima.")
else:
    col_badge, col_clear = st.columns([4, 1])
    with col_badge:
        st.subheader(f"Resumo da carga atual ({get_source_label()})")
    with col_clear:
        if st.button("🗑️ Limpar dados"):
            clear_pipeline()
            st.rerun()

    df = pipeline.display_df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Encontros", pipeline.n_encontros)
    c2.metric("Pacientes", pipeline.n_pacientes)
    c3.metric("Pacientes com múltiplos encontros", int((df["n_encontros"] >= 2).sum()))
    c4.metric("Maior trajetória (nº de encontros)", int(df["n_encontros"].max()))

    st.write("**Distribuição de fenótipos:**")
    st.write(df["fenotipo"].value_counts())

    st.divider()
    st.subheader("Páginas disponíveis")
    st.markdown(
        """
- **Visão Geral** — distribuições de utilização hospitalar
- **Fenótipos e Readmissão** — a associação mais forte encontrada no projeto
- **Trajetórias** — pacientes com múltiplos encontros, variável derivada
- **Dinâmica** — `MeanRevertingEvolutionOperator` na trajetória real, diagnóstico de robustez
- **Paciente** — busca individual
        """
    )
