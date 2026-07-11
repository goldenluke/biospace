import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import ORDEM_CLASSE_CONTROLE, apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Perfis - Diabetes", page_icon="🧬", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🧬 Perfis (Fenotipagem)")
st.caption(
    "KMeansPhenotyper genérico (biospace.phenotyping) — nenhum código específico de diabetes na "
    "fenotipagem em si; os fenótipos emergem da estrutura numérica da Representation."
)

c1, c2 = st.columns(2)
c1.metric("Fenótipos encontrados", len(pipeline.phenotypes))
c2.metric("Pacientes classificados", df["fenotipo"].notna().sum())

st.divider()

fig = px.histogram(df, x="fenotipo", color="classe_controle", category_orders={"classe_controle": ORDEM_CLASSE_CONTROLE}, title="Fenótipos x classe clínica de controle (comparação, não usada no ajuste)")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.subheader("Perfil médio por fenótipo")
cols_numericas = ["idade", "imc", "circunferencia_abdominal_cm", "glicemia_jejum_mg_dl", "hba1c_pct", "pressao_sistolica_mmhg", "creatinina_mg_dl", "taxa_filtracao_glomerular"]
tabela = df.groupby("fenotipo")[cols_numericas].mean().round(2)
st.dataframe(tabela, use_container_width=True)

st.subheader("Dispersão por fenótipo")
col1, col2 = st.columns(2)
with col1:
    fig2 = px.scatter(df, x="hba1c_pct", y="taxa_filtracao_glomerular", color="fenotipo", title="HbA1c x eGFR")
    st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
with col2:
    fig3 = px.scatter(df, x="imc", y="pressao_sistolica_mmhg", color="fenotipo", title="IMC x Pressão sistólica")
    st.plotly_chart(apply_default_layout(fig3), use_container_width=True)
