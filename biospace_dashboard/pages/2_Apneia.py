import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import CLASSE_APNEIA_CORES, ORDEM_CLASSE_APNEIA, apply_default_layout, histograma_faixas
from components.filters import render_filters
from components.state import require_pipeline

st.set_page_config(page_title="Apneia · BioSpace", page_icon="😴", layout="wide")

pipeline = require_pipeline()
df = render_filters(pipeline.display_df)

st.title("😴 Apneia (via IDO)")
st.caption(
    "Este dataset não tem canal de fluxo aéreo — o IDO (Índice de Dessaturação de "
    "Oxigênio) é usado como proxy de apneia, exatamente como no ApneaDomain do biospace."
)

if len(df) == 0:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("IDO médio", f"{df['ido'].mean():.1f}")
c2.metric("IDO máximo", f"{df['ido'].max():.1f}")
c3.metric("Pacientes graves", int((df["classe_apneia"] == "Grave").sum()))
c4.metric("% Grave", f"{100 * (df['classe_apneia'] == 'Grave').mean():.1f}%")

st.divider()

st.subheader("Distribuição do IDO")
fig = histograma_faixas(
    df, "ido", bins=40, titulo_x="IDO",
    faixas=[(None, 5, "#00C853", "Normal"), (5, 15, "#FFD600", "Leve"), (15, 30, "#FF7043", "Moderada"), (30, None, "#FF0000", "Grave")],
)
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("IDO por classe de apneia")
    fig = px.box(
        df, x="classe_apneia", y="ido", color="classe_apneia",
        color_discrete_map=CLASSE_APNEIA_CORES, category_orders={"classe_apneia": ORDEM_CLASSE_APNEIA},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

with col2:
    st.subheader("IDO por gênero")
    fig = px.box(df, x="genero", y="ido", color="genero")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.subheader("Top 20 por IDO")
top = (
    df.sort_values("ido", ascending=False)
    [["paciente", "idade", "imc", "ido", "spo2_minima", "classe_apneia", "fenotipo"]]
    .head(20)
)
st.dataframe(top, use_container_width=True, hide_index=True)
