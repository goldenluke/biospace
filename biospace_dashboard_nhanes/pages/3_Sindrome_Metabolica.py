import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.plugins.metabolic import classify_metabolic_syndrome_risk, classify_metabolic_syndrome_risk_full

st.set_page_config(page_title="Síndrome Metabólica - NHANES", page_icon="⚠️", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("⚠️ Síndrome Metabólica")
st.success(
    "Critério NCEP ATP III **completo** (Grundy et al., 2005) agora disponível — 5 critérios sexo-específicos "
    "(cintura, triglicerídeos, HDL, pressão, glicemia), habilitado pelos arquivos P_BIOPRO/P_TCHOL/P_HDL/P_TRIGLY. "
    "A versão adaptada de 4 critérios (abaixo) fica mantida para referência histórica — era a única disponível "
    "antes desses arquivos existirem."
)

avaliaveis_completo = df[df["sindrome_metabolica_completa_risco"].notna()]
c1, c2, c3 = st.columns(3)
c1.metric("Risco elevado (≥3 de 5 critérios, completo)", f"{100*avaliaveis_completo['sindrome_metabolica_completa_risco'].mean():.1f}%" if len(avaliaveis_completo) else "—")
c2.metric("Avaliáveis com os 5 critérios completos", int((df['sindrome_metabolica_completa_n_avaliaveis']==5).sum()))
c3.metric("Participantes totais", len(df))

st.divider()
fig_completo = px.histogram(df, x="sindrome_metabolica_completa_n_avaliaveis", title="Distribuição de nº de critérios AVALIÁVEIS (5 = completo, <3 = sem risco classificável)")
st.plotly_chart(apply_default_layout(fig_completo), use_container_width=True)

st.divider()
with st.expander("Versão adaptada de 4 critérios (referência histórica, antes de creatinina/lipídios existirem)"):
    st.warning(
        "Versão ADAPTADA dos critérios NCEP ATP III (2 de 4 critérios disponíveis: circunferência abdominal, "
        "IMC, pressão arterial, glicemia de jejum) — não sexo-específica, usa IMC como proxy. Mantida só para "
        "comparação com o achado publicado originalmente."
    )
    c1, c2 = st.columns(2)
    c1.metric("Risco elevado (≥2 de 4 critérios)", f"{100*df['sindrome_metabolica_risco'].mean():.1f}%")
    c2.metric("Participantes avaliados", len(df))
    fig = px.histogram(df, x="sindrome_metabolica_n_criterios", title="Distribuição do número de critérios presentes (versão adaptada)")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()
st.subheader("Testar num paciente específico")
labels_map = {row["id"]: row["paciente"] for _, row in df.iterrows()}
escolhido = st.selectbox("Paciente", list(labels_map.keys()), format_func=lambda s: labels_map[s])

if escolhido:
    vetor = pipeline.space.get(escolhido)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Critério completo (5, sexo-específico)**")
        st.json(classify_metabolic_syndrome_risk_full(vetor))
    with col2:
        st.write("**Critério adaptado (4, referência histórica)**")
        st.json(classify_metabolic_syndrome_risk(vetor))
