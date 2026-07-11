import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import CLASSE_APNEIA_CORES, ORDEM_CLASSE_APNEIA, apply_default_layout, histograma_faixas
from components.filters import render_filters
from components.state import require_pipeline

st.set_page_config(page_title="Frequência Cardíaca · BioSpace", page_icon="❤️", layout="wide")

pipeline = require_pipeline()
df = render_filters(pipeline.display_df)

st.title("❤️ Frequência Cardíaca")
st.caption("CardiovascularDomain do biospace, incluindo amplitude_fc (fc_maxima - fc_minima).")

if len(df) == 0:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("FC mínima média", f"{df['fc_minima_bpm'].mean():.1f}")
c2.metric("FC média", f"{df['fc_media_bpm'].mean():.1f}")
c3.metric("FC máxima média", f"{df['fc_maxima_bpm'].mean():.1f}")
c4.metric("Amplitude média", f"{df['amplitude_fc'].mean():.1f}")

st.divider()

st.subheader("Distribuição da Frequência Cardíaca Média")
fig = histograma_faixas(
    df, "fc_media_bpm", bins=30, titulo_x="Frequência Cardíaca Média (bpm)",
    faixas=[(None, 60, "#00AAFF", "Bradicardia"), (60, 100, "#00C853", "Normal"), (100, 120, "#FFD600", "Taquicardia leve"), (120, None, "#FF0000", "Taquicardia")],
)
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("FC média por classe de apneia")
    fig = px.box(
        df, x="classe_apneia", y="fc_media_bpm", color="classe_apneia",
        color_discrete_map=CLASSE_APNEIA_CORES, category_orders={"classe_apneia": ORDEM_CLASSE_APNEIA},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

with col2:
    st.subheader("Amplitude cardíaca")
    fig = histograma_faixas(
        df, "amplitude_fc", bins=30, titulo_x="Amplitude cardíaca (bpm)",
        faixas=[(None, 30, "#00AAFF", "Baixa"), (30, 50, "#00C853", "Normal"), (50, 70, "#FFD600", "Elevada"), (70, None, "#FF0000", "Muito elevada")],
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.subheader("IDO × FC média, por fenótipo")
fig = px.scatter(df, x="ido", y="fc_media_bpm", color="fenotipo", hover_name="paciente")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.subheader("Resumo por fenótipo")
resumo = df.groupby("fenotipo")[["fc_minima_bpm", "fc_media_bpm", "fc_maxima_bpm", "amplitude_fc"]].mean().round(1)
st.dataframe(resumo, use_container_width=True)
