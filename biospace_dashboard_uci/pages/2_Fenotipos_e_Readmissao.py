import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Fenótipos e Readmissão - UCI", page_icon="🎯", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🎯 Fenótipos e Readmissão")
st.success(
    "O achado mais forte deste projeto: fenotipagem sem idade nem diagnóstico — só utilização "
    "hospitalar, testagem glicêmica esparsa e intensidade de medicação — associa com readmissão real "
    "em 30 dias. O grupo de maior risco não é o de mais medicação; é o de maior utilização PRÉVIA."
)

df_validos = df.dropna(subset=["fenotipo", "readmitted_ultimo_encontro"])
tabela_taxa = df_validos.groupby("fenotipo")["readmitted_ultimo_encontro"].apply(lambda s: (s == "<30").mean()).sort_values(ascending=False)

st.subheader("Taxa de readmissão em 30 dias, por fenótipo")
fig = px.bar(tabela_taxa.reset_index(), x="fenotipo", y="readmitted_ultimo_encontro", title="Taxa de readmissão <30 dias por fenótipo")
fig.update_yaxes(tickformat=".1%")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()
st.subheader("Caracterização de cada fenótipo")
cols_caracterizacao = ["time_in_hospital", "num_medications", "num_lab_procedures", "number_outpatient", "number_emergency", "number_inpatient", "insulin_ordinal"]
tabela_caract = df.groupby("fenotipo")[cols_caracterizacao].mean().round(3)
st.dataframe(tabela_caract, use_container_width=True)

st.caption(
    "Compare a coluna 'insulin_ordinal' (intensidade de medicação) com 'number_outpatient'/'number_emergency'/"
    "'number_inpatient' (utilização prévia) entre o fenótipo de maior e menor risco acima — o achado "
    "documentado é que utilização prévia, não intensidade de medicação, distingue o grupo de maior risco."
)

st.divider()
st.subheader("Distribuição de readmissão por fenótipo (contagem completa)")
tabela_completa = pd.crosstab(df_validos["fenotipo"], df_validos["readmitted_ultimo_encontro"], normalize="index")
st.dataframe(tabela_completa, use_container_width=True)
