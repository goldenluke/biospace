import components._bootstrap  # noqa: F401

import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.plugins.metabolic import classify_metabolic_syndrome_risk

st.set_page_config(page_title="Síndrome Metabólica - NHANES", page_icon="⚠️", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("⚠️ Síndrome Metabólica (critério adaptado)")
st.warning(
    "Versão ADAPTADA dos critérios NCEP ATP III (2 de 4 critérios disponíveis: circunferência abdominal, "
    "IMC, pressão arterial, glicemia de jejum) — não o critério clínico completo de 5 critérios (que exige "
    "triglicerídeos/HDL, não coletados neste subconjunto de arquivos). Prevalência aqui NÃO é diretamente "
    "comparável a taxas publicadas com o critério original."
)

c1, c2 = st.columns(2)
c1.metric("Risco elevado (≥2 de 4 critérios)", f"{100*df['sindrome_metabolica_risco'].mean():.1f}%")
c2.metric("Participantes avaliados", len(df))

st.divider()

fig = px.histogram(df, x="sindrome_metabolica_n_criterios", title="Distribuição do número de critérios presentes")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.subheader("Testar num paciente específico")
labels_map = {row["id"]: row["paciente"] for _, row in df.iterrows()}
escolhido = st.selectbox("Paciente", list(labels_map.keys()), format_func=lambda s: labels_map[s])

if escolhido:
    vetor = pipeline.space.get(escolhido)
    resultado = classify_metabolic_syndrome_risk(vetor)
    st.json(resultado)
