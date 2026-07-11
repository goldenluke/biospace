import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.dynamics import DynamicSystem, MeanRevertingEvolutionOperator, StabilityOperator
from biospace.geometry import Euclidean

st.set_page_config(page_title="Sistemas Dinâmicos · BioSpace", page_icon="🌀", layout="wide")

pipeline = require_pipeline()

st.title("🌀 Sistemas Dinâmicos")
st.caption(
    "Trajectory → EvolutionOperator → Future State. Um processo de Ornstein-Uhlenbeck discreto "
    "ajustado POR FEATURE sobre todos os pares consecutivos (x_t, x_t+Δt) de toda a coorte — "
    "não uma intervenção hipotética, a dinâmica ESPONTÂNEA observada nos dados."
)

with st.expander("⚠️ Como interpretar φ (taxa de contração diária)", expanded=False):
    st.markdown(
        """
- **|φ| < 1**: a Feature reverte à média — perturbações se dissipam (estável).
- **|φ| ≥ 1**: a Feature diverge — não é necessariamente "piora patológica"; Features como
  **idade** (sempre cresce) ou **comorbidades** (diagnósticos persistentes, não revertem)
  aparecem corretamente como "instáveis" neste sentido técnico, sem que isso seja preocupante.
- φ perto de 1 mas com **poucos pares** (`n_pairs` baixo) é uma estimativa pouco confiável —
  julgue com essa informação ao lado, não só o valor de φ isolado.
        """
    )

order = pipeline.representation.domain_names()

if st.button("Ajustar dinâmica sobre a coorte inteira", type="primary"):
    with st.spinner("Ajustando um modelo de reversão à média por Feature (mínimos quadrados não lineares)..."):
        evo = MeanRevertingEvolutionOperator(order=order)
        try:
            evo.fit(pipeline.cohort)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        stability_op = StabilityOperator(evolution_operator=evo, n_worst=15)
        stability_report = stability_op.analyze(pipeline.cohort)
    st.session_state["_evo"] = evo
    st.session_state["_stability_report"] = stability_report

if "_evo" not in st.session_state:
    st.info("Clique no botão acima para ajustar a dinâmica (precisa de pacientes com ≥ 2 exames).")
    st.stop()

evo = st.session_state["_evo"]
stability_report = st.session_state["_stability_report"]

st.divider()
st.subheader("Estabilidade da dinâmica ajustada")

c1, c2, c3 = st.columns(3)
c1.metric("Features estáveis", f"{stability_report.n_stable}/{stability_report.n_features}")
c2.metric("Features instáveis (|φ|≥1)", stability_report.n_unstable)
c3.metric("Globalmente estável?", "Sim" if stability_report.is_globally_stable else "Não")

rows = [
    {"feature": name, "phi_dia": fd.phi_per_day, "n_pares": fd.n_pairs, "estavel": fd.is_stable, "residual_std": fd.residual_std}
    for name, fd in stability_report.dynamics.items()
]
dyn_df = pd.DataFrame(rows).sort_values("phi_dia", ascending=False)

fig = px.bar(
    dyn_df.head(20), x="feature", y="phi_dia", color="estavel",
    color_discrete_map={True: "#00C853", False: "#FF7043"},
    title="φ por Feature (20 maiores) — linha em φ=1 separa estável de instável",
)
fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig.update_xaxes(tickangle=-45)
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.dataframe(
    dyn_df.style.format({"phi_dia": "{:.4f}", "residual_std": "{:.3f}"}),
    use_container_width=True, hide_index=True,
)

st.divider()

# -----------------------------------------------------------------------------
# Previsão / simulação por paciente
# -----------------------------------------------------------------------------
st.subheader("Previsão e simulação por paciente")

elegiveis = [sid for sid, traj in pipeline.cohort.trajectories.items() if len(traj) >= 1]
labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in elegiveis}
paciente_escolhido = st.selectbox("Paciente", elegiveis, format_func=lambda s: labels_map[s])

horizon_days = st.slider("Horizonte de simulação (dias)", 30, 720, value=180, step=30)
step_days = st.slider("Passo da simulação (dias)", 15, 180, value=60, step=15)

traj = pipeline.cohort.trajectories[paciente_escolhido]
ds = DynamicSystem(trajectory=traj, evolution_operator=evo, order=order)

caminho = ds.simulate(horizon_days=horizon_days, step_days=step_days)

euclid = Euclidean()
estado_base = caminho[0][1]
distancias = [euclid.distance(estado_base, x) for _, x in caminho]
dias = [t for t, _ in caminho]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dias, y=distancias, mode="lines+markers", name="Distância ao estado atual (simulada)"))
fig2.update_layout(
    title=f"Trajetória simulada para {labels_map[paciente_escolhido]} (dinâmica espontânea, sem intervenção)",
    xaxis_title="Dias no futuro", yaxis_title="Distância Euclidiana ao estado atual",
)
st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
st.caption(
    "Isto projeta a evolução ESPONTÂNEA (sem nenhuma intervenção) — para simular o efeito de um "
    "tratamento hipotético ou observacional, use a página 'Inferência Causal' (do() + gêmeo digital)."
)

st.divider()

# -----------------------------------------------------------------------------
# Fase 9 — Simulação em conjunto (múltiplos futuros, com incerteza real)
# -----------------------------------------------------------------------------
st.subheader("Simulação em conjunto: múltiplos futuros possíveis (Fase 9)")
st.caption(
    "`twin.simulate()` acima dá UM futuro determinístico (a média). Um gêmeo digital de verdade "
    "relata incerteza — `twin.simulate_ensemble()` roda centenas de trajetórias estocásticas "
    "independentes, usando a variância teoricamente correta do processo de Ornstein-Uhlenbeck "
    "ajustado. Achado real ao validar isto (ver README do biospace): a escala de ruído inicial "
    "estava errada e inflava a incerteza em ~7x — corrigido e testado contra a variância "
    "estacionária teórica conhecida."
)

n_samples = st.slider("Nº de trajetórias simuladas", 50, 500, value=200, step=50)

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
    st.session_state["_ensemble_patient"] = paciente_escolhido

if "_ensemble" in st.session_state and st.session_state.get("_ensemble_patient") == paciente_escolhido:
    resultado_ensemble = st.session_state["_ensemble"]
    tempos = resultado_ensemble["times"]

    # Escolher qual Feature mostrar a faixa de incerteza (a primeira por padrao, ou a que o usuario escolher)
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
    fig_ens.update_layout(
        title=f"{feature_escolhida} — {n_samples} futuros simulados, {labels_map[paciente_escolhido]}",
        xaxis_title="Dias no futuro", yaxis_title="Valor (z-score)",
    )
    st.plotly_chart(apply_default_layout(fig_ens), use_container_width=True)
    st.caption(
        f"Desvio padrão no horizonte final: {desvio[-1]:.3f}. Se a Feature tiver dinâmica estável "
        "(|φ|<1), esse valor deveria convergir para a variância estacionária do processo — não crescer sem limite."
    )
