import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.filters import render_filters
from components.state import require_pipeline
from biospace.plugins.sleep import MAPA_DOENCAS, MAPA_SINTOMAS, MAPA_TRATAMENTOS

st.set_page_config(page_title="Sintomas e Comorbidades · BioSpace", page_icon="🩺", layout="wide")

pipeline = require_pipeline()
df = render_filters(pipeline.display_df)

st.title("🩺 Sintomas, Comorbidades e Tratamentos")
st.caption(
    "Prevalência calculada com os MESMOS mapas de texto usados pelo ComorbidityDomain / "
    "SymptomsDomain / TreatmentDomain do biospace (clinical_maps.py) — não uma lógica paralela."
)

if len(df) == 0:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()


def _prevalence(df: pd.DataFrame, column: str, mapping: dict[str, str]) -> pd.DataFrame:
    codes: list[str] = []
    for code in mapping.values():
        if code not in codes:
            codes.append(code)

    counts = {code: 0 for code in codes}
    for raw in df[column].fillna(""):
        items = {item.strip() for item in str(raw).split(",") if item.strip()}
        for code in codes:
            descriptions = [desc for desc, c in mapping.items() if c == code]
            if any(desc in items for desc in descriptions):
                counts[code] += 1

    result = pd.DataFrame({"codigo": list(counts.keys()), "n": list(counts.values())})
    result["pct"] = 100 * result["n"] / max(len(df), 1)
    return result.sort_values("n", ascending=False)


tab1, tab2, tab3 = st.tabs(["Comorbidades", "Sintomas", "Tratamentos"])

with tab1:
    prevalencia = _prevalence(df, "doencas", MAPA_DOENCAS)
    fig = px.bar(prevalencia, x="codigo", y="pct", labels={"codigo": "Comorbidade", "pct": "% dos pacientes"})
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    st.dataframe(prevalencia, use_container_width=True, hide_index=True)

with tab2:
    prevalencia = _prevalence(df, "sintomas", MAPA_SINTOMAS)
    fig = px.bar(prevalencia, x="codigo", y="pct", labels={"codigo": "Sintoma", "pct": "% dos pacientes"})
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    st.dataframe(prevalencia, use_container_width=True, hide_index=True)

with tab3:
    prevalencia = _prevalence(df, "tratamentos", MAPA_TRATAMENTOS)
    fig = px.bar(prevalencia, x="codigo", y="pct", labels={"codigo": "Tratamento", "pct": "% dos pacientes"})
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    st.dataframe(prevalencia, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Sonolência/sintomas × classe de apneia")
st.caption("Proporção de pacientes com pelo menos 2 sintomas reportados, por classe de apneia.")
df = df.copy()
df["n_sintomas"] = df["sintomas"].fillna("").apply(lambda s: len([i for i in str(s).split(",") if i.strip()]))
resumo = df.groupby("classe_apneia")["n_sintomas"].mean().round(2)
st.bar_chart(resumo)
