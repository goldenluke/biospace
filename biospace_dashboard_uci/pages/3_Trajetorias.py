import components._bootstrap  # noqa: F401

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import DerivedVariable, Feature

st.set_page_config(page_title="Trajetórias - UCI", page_icon="📈", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("📈 Trajetórias Multi-Encontro")
st.caption(
    "23,4% dos pacientes têm múltiplos encontros — a primeira trajetória REAL (não sintética, não "
    "transversal) testada no projeto. `encounter_id` não é data real, mas cresce monotonicamente — "
    "usado como proxy de ORDEM cronológica, não intervalo real."
)

multi = df[df["n_encontros"] >= 2].sort_values("n_encontros", ascending=False)
st.metric("Pacientes com múltiplos encontros", len(multi))

if multi.empty:
    st.info("Nenhum paciente com múltiplos encontros nesta amostra — use a base completa na página inicial.")
    st.stop()

labels_map = {row["id"]: f"{row['paciente']} ({row['n_encontros']} encontros)" for _, row in multi.head(100).iterrows()}
escolhido = st.selectbox("Paciente (ordenado por nº de encontros)", list(labels_map.keys()), format_func=lambda s: labels_map[s])

if escolhido:
    traj = pipeline.cohort.trajectories[escolhido]
    order = pipeline.representation.domain_names()

    st.subheader(f"Trajetória: {len(traj)} encontros")

    nomes_features = []
    for dom in order:
        vec0 = traj.at(0)
        for f in vec0.components.get(dom, []):
            nomes_features.append(f"{dom}.{f.name}")
    idx_default = nomes_features.index("utilization.num_medications") if "utilization.num_medications" in nomes_features else 0
    feature_escolhida = st.selectbox("Feature para visualizar ao longo dos encontros", nomes_features, index=idx_default)
    dom_escolhido, nome_feature = feature_escolhida.split(".", 1)

    valores = []
    for i in range(len(traj)):
        vec = traj.at(i)
        f = next((f for f in vec.components.get(dom_escolhido, []) if f.name == nome_feature), None)
        valores.append(f.raw_value if f and not f.is_missing else None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(traj) + 1)), y=valores, mode="lines+markers"))
    fig.update_layout(title=f"{feature_escolhida} ao longo dos encontros (ordem, não data real)", xaxis_title="Nº do encontro (ordem)", yaxis_title="Valor bruto")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    st.divider()
    st.subheader("Variável derivada: tendência (slope) de num_medications")

    class _NumMedicationsSlope(DerivedVariable):
        name = "num_medications_slope"
        domain_name = "utilization"
        feature_name = "num_medications"
        min_points = 2

        def compute(self, trajectory):
            pontos = self.series(trajectory)
            if len(pontos) < self.min_points:
                return None
            dias = np.array([p[0] for p in pontos])
            vals = np.array([p[1] for p in pontos])
            if np.ptp(dias) == 0:
                return None
            slope, _ = np.polyfit(dias, vals, 1)
            return Feature(name=self.name, value=float(slope), raw_value=float(slope))

    resultado = _NumMedicationsSlope().compute(traj)
    if resultado:
        st.metric("Slope de num_medications (por encontro, ordem sintética)", f"{resultado.value:.3f}")
    else:
        st.info("Trajetória curta demais para computar slope.")
