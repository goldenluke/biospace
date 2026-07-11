import components._bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Função Renal - Diabetes", page_icon="🫘", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🫘 Função Renal")
st.caption(
    "eGFR: valor MENOR = pior função renal. Creatinina: valor MAIOR = pior. "
    "O gerador sintético liga o declínio renal à EXPOSIÇÃO GLICÊMICA ACUMULADA ao longo do tempo "
    "(hiperglicemia crônica danifica os rins — mecanismo real), não só à severidade do instante."
)

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="taxa_filtracao_glomerular", nbins=25, title="eGFR (mL/min/1.73m²)")
    fig.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Limiar de doença renal crônica (60)")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col2:
    fig2 = px.histogram(df, x="creatinina_mg_dl", nbins=25, title="Creatinina (mg/dL)")
    st.plotly_chart(apply_default_layout(fig2), use_container_width=True)

st.divider()
st.subheader("Declínio renal × exposição glicêmica crônica (achado de rigor do gerador)")
st.caption(
    "Para cada paciente com ≥3 exames: HbA1c médio ao longo da trajetória × queda de eGFR "
    "(1º exame - último exame). Correlação esperada POSITIVA — pior controle crônico, mais dano renal."
)

pares = []
for sid, traj in pipeline.cohort.trajectories.items():
    if len(traj) < 3:
        continue
    hba1c_medios = [
        f.raw_value for i in range(len(traj)) for f in traj.at(i).components["glycemic"] if f.name == "hba1c_pct" and f.raw_value is not None
    ]
    egfr_i = next((f.raw_value for f in traj.at(0).components["renal"] if f.name == "taxa_filtracao_glomerular"), None)
    egfr_f = next((f.raw_value for f in traj.at(-1).components["renal"] if f.name == "taxa_filtracao_glomerular"), None)
    if egfr_i is not None and egfr_f is not None and hba1c_medios:
        pares.append({"hba1c_medio": float(np.mean(hba1c_medios)), "queda_egfr": egfr_i - egfr_f, "paciente": sid})

if len(pares) >= 5:
    pares_df = pd.DataFrame(pares)
    corr = np.corrcoef(pares_df["hba1c_medio"], pares_df["queda_egfr"])[0, 1]
    st.metric("Correlação HbA1c médio × queda de eGFR", f"ρ={corr:.3f}", "positiva = esperado" if corr > 0 else "⚠️ negativa, inesperado")
    fig3 = px.scatter(pares_df, x="hba1c_medio", y="queda_egfr", hover_data=["paciente"], title="Cada ponto = 1 paciente com ≥3 exames")
    fig3.update_layout(xaxis_title="HbA1c médio na trajetória (%)", yaxis_title="Queda de eGFR (1º exame - último)")
    st.plotly_chart(apply_default_layout(fig3), use_container_width=True)
else:
    st.info("Poucos pacientes com ≥3 exames no filtro atual para calcular a correlação — gere mais dados ou amplie os filtros.")

st.divider()
st.subheader("Trajetória renal de um paciente")
elegiveis = [sid for sid, t in pipeline.cohort.trajectories.items() if len(t) >= 2]
if elegiveis:
    labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in elegiveis}
    escolhido = st.selectbox("Paciente", elegiveis, format_func=lambda s: labels_map[s])
    traj = pipeline.cohort.trajectories[escolhido]
    dias = [(traj.at(i).timestamp - traj.at(0).timestamp).days for i in range(len(traj))]
    egfr_serie = [next((f.raw_value for f in traj.at(i).components["renal"] if f.name == "taxa_filtracao_glomerular"), None) for i in range(len(traj))]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=dias, y=egfr_serie, mode="lines+markers"))
    fig4.update_layout(title=f"eGFR ao longo do tempo — {labels_map[escolhido]}", xaxis_title="Dias desde o 1º exame", yaxis_title="eGFR")
    st.plotly_chart(apply_default_layout(fig4), use_container_width=True)
else:
    st.info("Nenhum paciente com ≥2 exames no filtro atual.")
