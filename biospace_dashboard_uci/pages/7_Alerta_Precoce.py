import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import Cohort
from biospace.early_warning import CriticalSlowingDownDetector

st.set_page_config(page_title="Alerta Precoce - UCI", page_icon="⏱️", layout="wide")

pipeline = require_pipeline()

st.title("⏱️ Sinais de Alerta Precoce (Critical Slowing Down)")
st.caption(
    "Teoria de Scheffer et al. (2009) / Dakos et al. (2012): perto de uma transição crítica, autocorrelação, "
    "variância e assimetria de uma série sobem antes da transição — testado via tendência de Kendall sobre "
    "janela móvel, com significância por dados substitutos AR(1). Requer ≥8 encontros por paciente."
)
st.warning(
    "**Achado documentado**: só 320 de 71.518 pacientes (0,45%) têm ≥8 encontros. Restringindo a um holdout "
    "real (detecção nos 8 primeiros, checagem de evento nos seguintes), sobram 139 (0,19%) — poder "
    "estatístico insuficiente para testar associação com desfecho futuro. Não é falha do método (este é o "
    "detector completo, validado contra simulação de bifurcação real) — é limite genuíno do acompanhamento "
    "disponível nesta base."
)

df = pipeline.display_df
multi = df[df["n_encontros"] >= 8]
st.metric("Pacientes com ≥8 encontros nesta amostra", len(multi))

if len(multi) < 5:
    st.error("Poucos pacientes com ≥8 encontros nesta amostra — use a base completa na página inicial.")
    st.stop()

feature_escolhida = st.selectbox("Feature (domínio utilization)", ["num_medications", "num_lab_procedures", "time_in_hospital", "number_diagnoses", "num_procedures"])

if st.button("Rodar detector de critical slowing down", type="primary"):
    cohort_longo = Cohort()
    for sid in multi["id"]:
        if len(pipeline.cohort.trajectories[sid]) >= 8:
            cohort_longo.systems[sid] = pipeline.cohort.systems[sid]
            cohort_longo.trajectories[sid] = pipeline.cohort.trajectories[sid]

    with st.spinner("Ajustando detrend + janela móvel + Kendall + 100 substitutos AR(1) por paciente..."):
        detector = CriticalSlowingDownDetector.for_feature("utilization", feature_escolhida, min_points=8, window_size=4, n_surrogates=100)
        resultados = detector.fit(cohort_longo)

    st.session_state["_uci_csd"] = resultados

if "_uci_csd" in st.session_state:
    resultados = st.session_state["_uci_csd"]
    suficientes = {sid: r for sid, r in resultados.items() if r.sufficient_data}
    n_warning = sum(1 for r in suficientes.values() if r.warning)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes analisados", len(suficientes))
    c2.metric("Com warning=True", n_warning)
    c3.metric("Taxa de warning", f"{100*n_warning/len(suficientes):.1f}%" if suficientes else "—")

    linhas = [{"system_id": sid, "tau_variancia": r.tau_variance, "tau_autocorrelacao": r.tau_autocorrelation, "tau_assimetria": r.tau_skewness, "warning": r.warning} for sid, r in suficientes.items()]
    dfc = pd.DataFrame(linhas)
    fig = px.histogram(dfc, x="tau_autocorrelacao", color="warning", barmode="overlay", opacity=0.6, title="Distribuição de τ de autocorrelação (Kendall), por status de alerta")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    st.divider()
    st.subheader("Detalhe por paciente")
    paciente_escolhido = st.selectbox("Paciente", list(suficientes.keys()))
    if paciente_escolhido:
        st.text(suficientes[paciente_escolhido].summary())
