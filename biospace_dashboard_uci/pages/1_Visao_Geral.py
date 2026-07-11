import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Visão Geral - UCI", page_icon="📊", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("📊 Visão Geral")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tempo de internação médio", f"{df['time_in_hospital'].mean():.1f} dias")
c2.metric("Medicações médias", f"{df['num_medications'].mean():.1f}")
c3.metric("Procedimentos de laboratório médios", f"{df['num_lab_procedures'].mean():.1f}")
c4.metric("Encontros por paciente (mediana)", f"{df['n_encontros'].median():.0f}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="n_encontros", title="Distribuição de nº de encontros por paciente", log_y=True)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col2:
    fig = px.histogram(df, x="time_in_hospital", nbins=14, title="Distribuição de tempo de internação")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig2 = px.histogram(df, x="readmitted_ultimo_encontro", title="Readmissão (último encontro)")
    st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
with col4:
    fig3 = px.scatter(df, x="number_inpatient", y="num_medications", color="readmitted_ultimo_encontro", title="Internações prévias x Nº de medicações")
    st.plotly_chart(apply_default_layout(fig3), use_container_width=True)

st.divider()
st.subheader("Tabela de pacientes (amostra)")
st.dataframe(df.head(200), use_container_width=True, hide_index=True)
