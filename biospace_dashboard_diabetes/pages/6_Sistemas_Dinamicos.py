import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.dynamics import DynamicSystem, MeanRevertingEvolutionOperator, StabilityOperator
from biospace.geometry import Euclidean

st.set_page_config(page_title="Sistemas Dinâmicos - Diabetes", page_icon="🌀", layout="wide")

pipeline = require_pipeline()
order = pipeline.representation.domain_names()

st.title("🌀 Sistemas Dinâmicos")
st.caption(
    "Trajectory -> EvolutionOperator -> Future State. Processo de Ornstein-Uhlenbeck discreto "
    "ajustado por Feature sobre pares consecutivos de toda a coorte — a dinâmica ESPONTÂNEA, "
    "sem nenhuma intervenção."
)

if st.button("Ajustar dinâmica sobre a coorte", type="primary"):
    with st.spinner("Ajustando modelo de reversão à média..."):
        try:
            evo = MeanRevertingEvolutionOperator(order=order)
            evo.fit(pipeline.cohort)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        stability_report = StabilityOperator(evolution_operator=evo, n_worst=15).analyze(pipeline.cohort)
    st.session_state["_evo"] = evo
    st.session_state["_stability_report"] = stability_report

if "_evo" not in st.session_state:
    st.info("Clique para ajustar (precisa de pacientes com >=2 exames).")
    st.stop()

evo = st.session_state["_evo"]
stability_report = st.session_state["_stability_report"]

st.divider()
st.subheader("Estabilidade da dinâmica ajustada")
c1, c2, c3 = st.columns(3)
c1.metric("Features estáveis", f"{stability_report.n_stable}/{stability_report.n_features}")
c2.metric("Features instáveis (|phi|>=1)", stability_report.n_unstable)
c3.metric("Globalmente estável?", "Sim" if stability_report.is_globally_stable else "Não")

rows = [{"feature": name, "phi_dia": fd.phi_per_day, "n_pares": fd.n_pairs, "estavel": fd.is_stable} for name, fd in stability_report.dynamics.items()]
dyn_df = pd.DataFrame(rows).sort_values("phi_dia", ascending=False)
fig = px.bar(dyn_df.head(20), x="feature", y="phi_dia", color="estavel", color_discrete_map={True: "#00C853", False: "#FF7043"}, title="Phi por Feature")
fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig.update_xaxes(tickangle=-45)
st.plotly_chart(apply_default_layout(fig), use_container_width=True)
st.dataframe(dyn_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Previsão/simulação por paciente")
elegiveis = [sid for sid, t in pipeline.cohort.trajectories.items() if len(t) >= 1]
labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in elegiveis}
escolhido = st.selectbox("Paciente", elegiveis, format_func=lambda s: labels_map[s])
horizon_days = st.slider("Horizonte (dias)", 30, 720, 180, step=30)
step_days = st.slider("Passo (dias)", 15, 180, 60, step=15)

traj = pipeline.cohort.trajectories[escolhido]
ds = DynamicSystem(trajectory=traj, evolution_operator=evo, order=order)
caminho = ds.simulate(horizon_days=horizon_days, step_days=step_days)

euclid = Euclidean()
estado_base = caminho[0][1]
distancias = [euclid.distance(estado_base, x) for _, x in caminho]
dias = [t for t, _ in caminho]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dias, y=distancias, mode="lines+markers"))
fig2.update_layout(title=f"Trajetória simulada (sem intervenção) - {labels_map[escolhido]}", xaxis_title="Dias no futuro", yaxis_title="Distância ao estado atual")
st.plotly_chart(apply_default_layout(fig2), use_container_width=True)

st.divider()
st.subheader("Simulação em conjunto: múltiplos futuros possíveis (Fase 9)")
st.caption(
    "twin.simulate() acima dá um único futuro determinístico (a média). Um gêmeo digital de "
    "verdade relata incerteza — twin.simulate_ensemble() roda centenas de trajetórias "
    "estocásticas independentes, com a variância teoricamente correta do processo ajustado."
)

n_samples = st.slider("Número de trajetórias simuladas", 50, 500, value=200, step=50)

if st.button("Simular em conjunto"):
    from biospace.causal import DigitalTwin

    with st.spinner(f"Simulando {n_samples} futuros possíveis..."):
        twin = DigitalTwin.clone_from(traj, order=order)
        try:
            resultado_ensemble = twin.simulate_ensemble(evo, horizon_days=horizon_days, step_days=step_days, n_samples=n_samples, seed=0)
        except TypeError as e:
            st.error(str(e))
            st.stop()
    st.session_state["_ensemble"] = resultado_ensemble
    st.session_state["_ensemble_patient"] = escolhido

if "_ensemble" in st.session_state and st.session_state.get("_ensemble_patient") == escolhido:
    resultado_ensemble = st.session_state["_ensemble"]
    tempos = resultado_ensemble["times"]

    nomes_features = []
    for dom in order:
        vec0 = traj.at(0)
        for f in vec0.components.get(dom, []):
            nomes_features.append(f"{dom}.{f.name}")
    feature_escolhida = st.selectbox("Feature para visualizar a faixa de incerteza", nomes_features, key="ensemble_feature")
    idx_feature = nomes_features.index(feature_escolhida)

    media = resultado_ensemble["mean"][:, idx_feature]
    desvio = resultado_ensemble["std"][:, idx_feature]

    fig_ens = go.Figure()
    fig_ens.add_trace(go.Scatter(
        x=list(tempos) + list(tempos[::-1]),
        y=list(media + 1.96 * desvio) + list((media - 1.96 * desvio)[::-1]),
        fill="toself", fillcolor="rgba(0,150,255,0.15)", line=dict(width=0), name="95% dos futuros simulados",
    ))
    fig_ens.add_trace(go.Scatter(x=tempos, y=media, mode="lines+markers", name="Média", line=dict(color="rgb(0,100,200)")))
    fig_ens.update_layout(title=f"{feature_escolhida} - {n_samples} futuros simulados", xaxis_title="Dias no futuro", yaxis_title="Valor (z-score)")
    st.plotly_chart(apply_default_layout(fig_ens), use_container_width=True)
