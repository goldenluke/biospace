import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Visão Geral - NHANES", page_icon="📊", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("📊 Visão Geral")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Adultos analisados", len(df))
c2.metric("Idade média", f"{df['idade'].mean():.1f} anos")
c3.metric("HbA1c médio", f"{df['hba1c_pct'].mean():.1f}%" if df["hba1c_pct"].notna().any() else "—")
c4.metric("IMC médio", f"{df['imc'].mean():.1f}" if df["imc"].notna().any() else "—")

st.divider()

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="idade", nbins=30, title="Distribuição de idade")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col2:
    fig = px.histogram(df.dropna(subset=["hba1c_pct"]), x="hba1c_pct", nbins=40, title="Distribuição de HbA1c")
    fig.add_vline(x=6.5, line_dash="dash", line_color="red", annotation_text="limiar diabetes")
    fig.add_vline(x=5.7, line_dash="dash", line_color="orange", annotation_text="limiar pré-diabetes")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig2 = px.scatter(df.dropna(subset=["imc", "hba1c_pct"]), x="imc", y="hba1c_pct", color="status_diabetes_laboratorial", title="IMC x HbA1c")
    st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
with col4:
    fig3 = px.scatter(
        df.dropna(subset=["pressao_sistolica_mmhg", "circunferencia_abdominal_cm"]),
        x="circunferencia_abdominal_cm", y="pressao_sistolica_mmhg", color="sindrome_metabolica_risco",
        title="Circunferência abdominal x Pressão sistólica",
    )
    st.plotly_chart(apply_default_layout(fig3), use_container_width=True)

st.divider()
st.subheader("Tabela de participantes (amostra)")
st.dataframe(df.head(200), use_container_width=True, hide_index=True)
