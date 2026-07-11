import components._bootstrap  # noqa: F401

import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Paciente - Diabetes", page_icon="🔍", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🔍 Paciente")

ids = pipeline.space.ids()
labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in ids}
escolhido = st.selectbox("Paciente", ids, format_func=lambda s: labels_map[s])

linha = df[df["id"] == escolhido].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Idade", f"{linha['idade']:.0f}")
c2.metric("IMC", f"{linha['imc']:.1f}")
c3.metric("HbA1c", f"{linha['hba1c_pct']:.1f}%")
c4.metric("Classe", linha["classe_controle"])

c5, c6, c7 = st.columns(3)
c5.metric("Fenótipo", linha["fenotipo"])
c6.metric("N. exames", linha["n_exames"])
c7.metric("eGFR", f"{linha['taxa_filtracao_glomerular']:.0f}" if linha["taxa_filtracao_glomerular"] == linha["taxa_filtracao_glomerular"] else "ausente")

comorbidades = [c for c in ["hipertensao", "retinopatia", "neuropatia", "doenca_cardiovascular"] if linha[c] == 1]
tratamentos = [t for t in ["metformina", "insulina"] if linha[t] == 1]
st.write("**Comorbidades:**", ", ".join(comorbidades) if comorbidades else "nenhuma")
st.write("**Tratamentos:**", ", ".join(tratamentos) if tratamentos else "nenhum")

st.divider()

traj = pipeline.cohort.trajectories[escolhido]
if len(traj) >= 2:
    st.subheader(f"Trajetória ({len(traj)} exames)")
    dias = [(traj.at(i).timestamp - traj.at(0).timestamp).days for i in range(len(traj))]

    col1, col2 = st.columns(2)
    with col1:
        hba1c_serie = [next((f.raw_value for f in traj.at(i).components["glycemic"] if f.name == "hba1c_pct"), None) for i in range(len(traj))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dias, y=hba1c_serie, mode="lines+markers"))
        fig.update_layout(title="HbA1c ao longo do tempo", xaxis_title="Dias", yaxis_title="HbA1c (%)")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    with col2:
        egfr_serie = [next((f.raw_value for f in traj.at(i).components["renal"] if f.name == "taxa_filtracao_glomerular"), None) for i in range(len(traj))]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dias, y=egfr_serie, mode="lines+markers", line_color="orange"))
        fig2.update_layout(title="eGFR ao longo do tempo", xaxis_title="Dias", yaxis_title="eGFR")
        st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
else:
    st.info("Este paciente tem apenas 1 exame — sem trajetória para mostrar.")
