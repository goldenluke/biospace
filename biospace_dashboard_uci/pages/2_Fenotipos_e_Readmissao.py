import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Fenótipos e Readmissão - UCI", page_icon="🎯", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df
com_diagnostico = "diagnosis_category" in pipeline.representation.domain_names()

st.title("🎯 Fenótipos e Readmissão")

if com_diagnostico:
    st.info(
        "📊 Representação ATUAL: **com diagnóstico ICD-9** (4 domínios — o default). Achado real: incluir esse "
        "domínio muda o que K-Means encontra — a associação com readmissão fica mais fraca (~1,5x) que na "
        "representação sem diagnóstico (~2,2x, ver abaixo). Volte à página inicial para trocar de representação."
    )
else:
    st.success(
        "📊 Representação ATUAL: **sem diagnóstico** (3 domínios — o achado original deste projeto). Fenotipagem "
        "sem idade nem diagnóstico — só utilização hospitalar, testagem glicêmica esparsa e intensidade de "
        "medicação — associa com readmissão real em 30 dias. O grupo de maior risco não é o de mais medicação; "
        "é o de maior utilização PRÉVIA."
    )

st.warning(
    "🔬 **Nenhuma das duas representações está errada** — capturam estruturas diferentes do mesmo dado real. "
    "É a mesma tese do artigo \"Representation Before Inference\": adicionar uma variável à representação não é "
    "neutro. Troque a representação na página inicial e volte aqui para comparar os dois resultados você mesmo."
)

df_validos = df.dropna(subset=["fenotipo", "readmitted_ultimo_encontro"])
tabela_taxa = df_validos.groupby("fenotipo")["readmitted_ultimo_encontro"].apply(lambda s: (s == "<30").mean()).sort_values(ascending=False)

st.subheader("Taxa de readmissão em 30 dias, por fenótipo")
fig = px.bar(tabela_taxa.reset_index(), x="fenotipo", y="readmitted_ultimo_encontro", title="Taxa de readmissão <30 dias por fenótipo")
fig.update_yaxes(tickformat=".1%")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

razao = tabela_taxa.max() / tabela_taxa.min() if tabela_taxa.min() > 0 else float("nan")
st.metric("Razão entre fenótipo de maior e menor risco", f"{razao:.2f}x")

st.divider()
st.subheader("Caracterização de cada fenótipo")
cols_base = ["time_in_hospital", "num_medications", "num_lab_procedures", "number_outpatient", "number_emergency", "number_inpatient", "insulin_ordinal"]
cols_caracterizacao = [c for c in cols_base if c in df.columns]
tabela_caract = df.groupby("fenotipo")[cols_caracterizacao].mean().round(3)
st.dataframe(tabela_caract, use_container_width=True)

if com_diagnostico:
    st.caption(
        "Com diagnóstico incluído, procure o grupo com 'time_in_hospital' muito mais alto que os demais — é o "
        "achado documentado (~13 dias vs. ~4-5 dos outros), não mais o de maior utilização prévia isoladamente."
    )
else:
    st.caption(
        "Compare a coluna 'insulin_ordinal' (intensidade de medicação) com 'number_outpatient'/'number_emergency'/"
        "'number_inpatient' (utilização prévia) entre o fenótipo de maior e menor risco acima — o achado "
        "documentado é que utilização prévia, não intensidade de medicação, distingue o grupo de maior risco."
    )

st.divider()
st.subheader("Distribuição de readmissão por fenótipo (contagem completa)")
tabela_completa = pd.crosstab(df_validos["fenotipo"], df_validos["readmitted_ultimo_encontro"], normalize="index")
st.dataframe(tabela_completa, use_container_width=True)
