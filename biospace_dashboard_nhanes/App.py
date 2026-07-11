"""
App.py
======

Página inicial do dashboard BioSpace NHANES — dados REAIS (ciclo
Pre-pandemic, ago/2017-mar/2020), enviados pelo usuário.
"""

import components._bootstrap  # noqa: F401

import streamlit as st

from components.pipeline import run_pipeline
from components.state import clear_pipeline, get_pipeline, get_source_label, set_pipeline

st.set_page_config(page_title="BioSpace - NHANES", page_icon="🧪", layout="wide")

st.title("🧪 BioSpace — NHANES (dados reais)")
st.caption(
    "National Health and Nutrition Examination Survey, ciclo Pre-pandemic (agosto/2017-março/2020). "
    "Inquérito de saúde populacional real do CDC — não dado sintético. Representação computacional "
    "via `biospace.plugins.metabolic` (MetabolicRepresentation), interpretações clínicas via "
    "`classify_diabetes_status` e `classify_metabolic_syndrome_risk`."
)
st.info(
    "🌐 Dados públicos, de-identificados pelo CDC. Enviados manualmente pelo usuário (o ambiente não "
    "tem acesso de rede a cdc.gov). Arquivos esperados em `/mnt/user-data/uploads`: P_DEMO.xpt, "
    "P_GHB.xpt, P_GLU.xpt, P_BMX.xpt, P_BPXO.xpt, P_DIQ.xpt."
)

st.divider()

st.subheader("Carregar coorte")
caminho = st.text_input("Pasta com os 6 arquivos .XPT", value="/mnt/user-data/uploads")
idade_minima = st.slider("Idade mínima (filtra para adultos — critério ADA de diabetes não se aplica a crianças)", 0, 40, value=20)

if st.button("Carregar NHANES", type="primary"):
    cache_key = f"nhanes::{caminho}::{idade_minima}"
    with st.spinner("Carregando arquivos .XPT, construindo representação, classificando..."):
        try:
            pipeline = run_pipeline(caminho, idade_minima=idade_minima)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
    set_pipeline(pipeline, cache_key)
    st.success(f"Carregado: {pipeline.n_total} participantes totais, {pipeline.n_adultos} com idade ≥{idade_minima}.")
    st.rerun()

st.divider()

pipeline = get_pipeline()
if pipeline is None:
    st.info("Nenhum dado carregado ainda. Clique em 'Carregar NHANES' acima.")
else:
    col_badge, col_clear = st.columns([4, 1])
    with col_badge:
        st.subheader(f"Resumo da carga atual ({get_source_label()})")
    with col_clear:
        if st.button("🗑️ Limpar dados"):
            clear_pipeline()
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participantes totais (todas as idades)", pipeline.n_total)
    c2.metric("Adultos analisados", pipeline.n_adultos)
    c3.metric("Com HbA1c registrado", int(pipeline.display_df["hba1c_pct"].notna().sum()))
    c4.metric("Domínios", len(pipeline.representation.domain_names()))

    st.write("**Distribuição de status glicêmico (critério ADA, laboratorial):**")
    st.write(pipeline.display_df["status_diabetes_laboratorial"].value_counts())

    st.divider()
    st.subheader("Páginas disponíveis")
    st.markdown(
        """
- **Visão Geral** — distribuições demográficas e laboratoriais
- **Diagnóstico** — classificação laboratorial x autorrelato (sensibilidade/especificidade)
- **Síndrome Metabólica** — critérios adaptados NCEP ATP III
- **Coerência de Processo** — `check_process_coherence` em população real
- **Estabilidade e Curvatura** — varredura de estabilidade fenotípica (com/sem idade) e curvatura estrutural, mesma metodologia de SAOS
- **Paciente** — busca individual
        """
    )
