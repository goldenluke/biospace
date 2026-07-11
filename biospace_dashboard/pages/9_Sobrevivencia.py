import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.longitudinal import SurvivalAnalyzer

st.set_page_config(page_title="Sobrevivência · BioSpace", page_icon="⏳", layout="wide")

pipeline = require_pipeline()

st.title("⏳ Análise de Sobrevivência (Tempo-até-Evento)")
st.caption(
    "SurvivalAnalyzer: tempo desde o primeiro exame de cada paciente até que uma condição de "
    "interesse ocorra pela primeira vez. Pacientes cuja trajetória termina sem o evento são "
    "CENSURADOS — tratamento estatístico padrão, não dado descartado (estimador de Kaplan-Meier)."
)


def plot_km(km, titulo: str):
    if not km.time_days:
        st.warning("Nenhum evento observado nesta população — não é possível estimar a curva.")
        return
    x = [0.0] + km.time_days
    y = [1.0] + km.survival
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line_shape="hv", name="S(t)"))
    mediana = km.median_survival_time()
    if mediana is not None:
        fig.add_vline(x=mediana, line_dash="dash", line_color="red", annotation_text=f"mediana: {mediana:.0f}d")
    fig.update_layout(
        title=titulo, xaxis_title="Dias desde o primeiro exame", yaxis_title="S(t) — probabilidade de não-evento",
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes analisados", km.n_total)
    c2.metric("Eventos observados", km.n_events_total)
    c3.metric("Censurados", km.n_censored_total)


tab1, tab2 = st.tabs(["Tempo até fenótipo grave", "Tempo até início de tratamento"])

with tab1:
    st.subheader("Tempo até entrar no fenótipo mais severo")
    st.caption(
        "Usa os fenótipos já estimados na página 'Perfis' (ClinicalKMeansPhenotyper). O fenótipo "
        "'mais severo' é escolhido pela interpretação (maior severidade_relativa)."
    )

    def _severity_from_interpretation(ph) -> float:
        try:
            return float(ph.interpretation.split("severidade_relativa=")[1].split(",")[0])
        except (IndexError, ValueError):
            return float("-inf")

    if pipeline.phenotypes:
        fenotipo_grave = max(pipeline.phenotypes, key=_severity_from_interpretation)
        st.info(f"Fenótipo mais severo identificado: **{fenotipo_grave.name}**")

        analyzer = SurvivalAnalyzer.for_phenotype(fenotipo_grave, order=pipeline.space.order())
        km = analyzer.kaplan_meier(pipeline.cohort)
        plot_km(km, f"Tempo até '{fenotipo_grave.name}'")
    else:
        st.warning("Nenhum fenótipo estimado ainda.")

with tab2:
    st.subheader("Tempo até início de tratamento (AAM)")
    st.caption("Evento = a Feature binária 'aam' do TreatmentDomain passa a valer 1.0 pela primeira vez.")

    analyzer_treat = SurvivalAnalyzer.for_feature_threshold("treatment", "aam")
    km_treat = analyzer_treat.kaplan_meier(pipeline.cohort)
    plot_km(km_treat, "Tempo até início de AAM (aparelho de avanço mandibular)")

    st.caption("Trocar para CPAP:")
    if st.button("Ver tempo até início de CPAP"):
        analyzer_cpap = SurvivalAnalyzer.for_feature_threshold("treatment", "cpap")
        km_cpap = analyzer_cpap.kaplan_meier(pipeline.cohort)
        plot_km(km_cpap, "Tempo até início de CPAP")
