import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout, histograma_faixas
from components.state import require_pipeline

st.set_page_config(page_title="Controle Glicêmico - Diabetes", page_icon="🍬", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🍬 Controle Glicêmico")

col1, col2 = st.columns(2)
with col1:
    fig = histograma_faixas(
        df, "hba1c_pct", 20, "HbA1c (%)",
        [(None, 7.0, "#00C853", "Controlado"), (7.0, 9.0, "#FFD600", "Moderado"), (9.0, None, "#FF0000", "Descompensado")],
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col2:
    fig2 = px.histogram(df, x="glicemia_jejum_mg_dl", nbins=25, title="Glicemia de jejum (mg/dL)")
    st.plotly_chart(apply_default_layout(fig2), use_container_width=True)

st.subheader("Glicemia × HbA1c")
st.caption("Devem correlacionar — ambas medem controle glicêmico, mas em janelas de tempo diferentes (glicemia = agora; HbA1c = média de ~3 meses).")
fig3 = px.scatter(df, x="glicemia_jejum_mg_dl", y="hba1c_pct", color="classe_controle")
st.plotly_chart(apply_default_layout(fig3), use_container_width=True)

st.subheader("Controle glicêmico por uso de tratamento")
col3, col4 = st.columns(2)
with col3:
    fig4 = px.box(df, x="metformina", y="hba1c_pct", title="HbA1c por uso de metformina (0/1)")
    st.plotly_chart(apply_default_layout(fig4), use_container_width=True)
with col4:
    fig5 = px.box(df, x="insulina", y="hba1c_pct", title="HbA1c por uso de insulina (0/1)")
    st.caption("Pacientes em insulina tendem a HbA1c MAIOR — não porque insulina piora o controle, mas porque é prescrita para os casos mais graves (confundimento por indicação — ver página Inferência Causal).")
    st.plotly_chart(apply_default_layout(fig5), use_container_width=True)
