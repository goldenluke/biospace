import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import ORDEM_CLASSE_CONTROLE, apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Visão Geral - Diabetes", page_icon="📊", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("📊 Visão Geral")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pacientes", len(df))
c2.metric("Idade média", f"{df['idade'].mean():.1f} anos")
c3.metric("HbA1c média", f"{df['hba1c_pct'].mean():.1f}%")
c4.metric("IMC médio", f"{df['imc'].mean():.1f}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="classe_controle", category_orders={"classe_controle": ORDEM_CLASSE_CONTROLE}, title="Distribuição de controle glicêmico")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col2:
    fig = px.pie(df, names="classe_controle", category_orders={"classe_controle": ORDEM_CLASSE_CONTROLE}, title="Proporção por classe")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig = px.histogram(df, x="hba1c_pct", color="classe_controle", category_orders={"classe_controle": ORDEM_CLASSE_CONTROLE}, title="HbA1c por classe")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col4:
    fig = px.scatter(df, x="imc", y="glicemia_jejum_mg_dl", color="classe_controle", category_orders={"classe_controle": ORDEM_CLASSE_CONTROLE}, title="IMC x Glicemia de jejum")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()
st.subheader("Tabela de pacientes")
st.dataframe(df, use_container_width=True, hide_index=True)
