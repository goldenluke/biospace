import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import check_process_coherence

st.set_page_config(page_title="Coerência de Processo - NHANES", page_icon="🔗", layout="wide")

pipeline = require_pipeline()

st.title("🔗 Coerência de Processo Fisiológico")
st.caption(
    "Testa se Features do mesmo PhysiologicalProcess (ex.: HbA1c e glicemia, ambas "
    "'glucose_homeostasis') correlacionam mais entre si, através da população, do que Features de "
    "processos diferentes. Validado antes em cenários sintéticos com verdade conhecida — aqui, testado "
    "pela primeira vez em população real."
)
st.success(
    "Achado documentado: em população NHANES real, a coerência se CONFIRMA (p=0,0022) — o oposto do "
    "achado no gerador sintético de diabetes usado em outras partes deste projeto (não confirmado lá). "
    "Mesma ferramenta, duas fontes com propriedades diferentes, duas respostas corretas."
)

if st.button("Rodar check_process_coherence", type="primary"):
    with st.spinner("Calculando correlações par a par..."):
        relatorio = check_process_coherence(pipeline.representation, pipeline.space)
    st.session_state["_nhanes_coherence"] = relatorio

if "_nhanes_coherence" in st.session_state:
    relatorio = st.session_state["_nhanes_coherence"]
    st.text(relatorio.summary())

    c1, c2 = st.columns(2)
    c1.metric("|r| médio, mesmo processo", f"{relatorio.mean_same_process:.3f}", help=f"{len(relatorio.same_process_pairs)} pares")
    c2.metric("|r| médio, processos diferentes", f"{relatorio.mean_different_process:.3f}", help=f"{len(relatorio.different_process_pairs)} pares")

    if relatorio.mannwhitney_p is not None:
        st.metric("p-valor (Mann-Whitney)", f"{relatorio.mannwhitney_p:.2e}")
        if relatorio.is_coherent:
            st.success("Coerência CONFIRMADA.")
        else:
            st.warning("Coerência NÃO confirmada nesta execução.")

    if relatorio.same_process_pairs:
        import pandas as pd

        df_pares = pd.DataFrame(relatorio.same_process_pairs, columns=["feature_a", "feature_b", "abs_r"])
        fig = px.bar(df_pares, x=df_pares.index, y="abs_r", hover_data=["feature_a", "feature_b"], title="Pares do mesmo processo")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
