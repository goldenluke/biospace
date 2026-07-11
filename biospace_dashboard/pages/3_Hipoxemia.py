import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import CLASSE_APNEIA_CORES, ORDEM_CLASSE_APNEIA, apply_default_layout, histograma_faixas
from components.filters import render_filters
from components.state import require_pipeline

st.set_page_config(page_title="Hipoxemia · BioSpace", page_icon="🫁", layout="wide")

pipeline = require_pipeline()
df = render_filters(pipeline.display_df)

st.title("🫁 Hipoxemia")
st.caption(
    "HypoxiaDomain do biospace — SpO2 é invertida internamente na representação "
    "(maior valor = mais grave), mas aqui exibimos a escala clínica original."
)

if len(df) == 0:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

domain = next((d for d in pipeline.representation.domains if d.name == "hypoxia"), None)
if domain is not None and getattr(domain, "missing_counts", None):
    n = len(pipeline.cohort)
    weights = domain.feature_weights()
    with st.expander("⚠️ Qualidade de dados deste domínio (ver também a página 'Qualidade de Dados')"):
        for key, count in sorted(domain.missing_counts.items(), key=lambda kv: -kv[1]):
            w = weights.get(key, 1.0)
            st.write(f"- `{key}`: {count}/{n} ausentes ({100*count/n:.1f}%), peso aplicado = {w:.2f}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("SpO2 mínima média", f"{df['spo2_minima'].mean():.1f}%")
c2.metric("SpO2 média", f"{df['spo2_media'].mean():.1f}%")
c3.metric("Carga hipóxica média", f"{df['carga_hipoxica_min_h'].mean():.1f}")
c4.metric("Nº dessaturações (médio)", f"{df['no_de_dessaturacoes'].mean():.1f}")

st.divider()

st.subheader("Distribuição de SpO2 mínima")
fig = histograma_faixas(
    df, "spo2_minima", bins=40, titulo_x="SpO2 mínima (%)",
    faixas=[(None, 80, "#FF0000", "Grave"), (80, 90, "#FF7043", "Moderada"), (90, None, "#00C853", "Normal")],
)
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("SpO2 mínima por classe de apneia")
    fig = px.box(
        df, x="classe_apneia", y="spo2_minima", color="classe_apneia",
        color_discrete_map=CLASSE_APNEIA_CORES, category_orders={"classe_apneia": ORDEM_CLASSE_APNEIA},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

with col2:
    st.subheader("IDO × SpO2 mínima")
    fig = px.scatter(
        df, x="ido", y="spo2_minima", color="classe_apneia", hover_name="paciente",
        color_discrete_map=CLASSE_APNEIA_CORES, category_orders={"classe_apneia": ORDEM_CLASSE_APNEIA},
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.subheader("Carga hipóxica × Nº de dessaturações")
fig = px.scatter(df, x="no_de_dessaturacoes", y="carga_hipoxica_min_h", color="fenotipo", hover_name="paciente")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)
