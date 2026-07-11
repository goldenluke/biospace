import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import CLASSE_APNEIA_CORES, ORDEM_CLASSE_APNEIA, apply_default_layout
from components.filters import render_filters
from components.state import require_pipeline

st.set_page_config(page_title="Visão Geral · BioSpace", page_icon="📊", layout="wide")

pipeline = require_pipeline()
df = render_filters(pipeline.display_df)

st.title("📊 Visão Geral")
st.caption("Resumo populacional sobre o RepresentationSpace já construído — nenhum recálculo acontece aqui.")

if len(df) == 0:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pacientes", len(df))
c2.metric("Idade média", f"{df['idade'].mean():.1f}")
c3.metric("IMC médio", f"{df['imc'].mean():.1f}")
c4.metric("IDO médio", f"{df['ido'].mean():.1f}")
c5.metric("% Grave (IDO)", f"{100 * (df['classe_apneia'] == 'Grave').mean():.1f}%")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição por classe de apneia (IDO)")
    counts = df["classe_apneia"].value_counts().reindex(ORDEM_CLASSE_APNEIA).fillna(0)
    fig = px.bar(
        x=counts.index, y=counts.values,
        color=counts.index, color_discrete_map=CLASSE_APNEIA_CORES,
        labels={"x": "Classe", "y": "Nº de pacientes"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

with col2:
    st.subheader("Distribuição por fenótipo")
    counts = df["fenotipo"].value_counts()
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Fenótipo", "y": "Nº de pacientes"})
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Idade × IDO")
    fig = px.scatter(
        df, x="idade", y="ido", color="classe_apneia", hover_name="paciente",
        color_discrete_map=CLASSE_APNEIA_CORES,
        category_orders={"classe_apneia": ORDEM_CLASSE_APNEIA},
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

with col4:
    st.subheader("IMC × IDO, por fenótipo")
    fig = px.scatter(df, x="imc", y="ido", color="fenotipo", hover_name="paciente")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()
st.subheader("Resumo por classe de apneia")
resumo = (
    df.groupby("classe_apneia")[["idade", "imc", "ido", "spo2_minima"]]
    .mean()
    .reindex(ORDEM_CLASSE_APNEIA)
    .round(1)
)
st.dataframe(resumo, use_container_width=True)
