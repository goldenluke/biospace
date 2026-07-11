import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Diagnóstico - NHANES", page_icon="🩺", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🩺 Diagnóstico Laboratorial x Autorrelato")
st.caption(
    "Critério glicêmico da American Diabetes Association (HbA1c ≥6,5% ou glicemia de jejum ≥126 mg/dL → "
    "diabetes; ≥5,7%/≥100 mg/dL → pré-diabetes) comparado à resposta autorreferida no questionário de diabetes."
)

comparaveis = df[df["diabetes_autorreferido"].isin([1.0, 2.0]) & (df["status_diabetes_laboratorial"] != "indeterminado")].copy()
comparaveis["autorrelato"] = comparaveis["diabetes_autorreferido"].map({1.0: "Sim", 2.0: "Não"})

st.metric("Participantes comparáveis (têm HbA1c/glicemia E resposta binária de autorrelato)", len(comparaveis))

tabela = pd.crosstab(comparaveis["status_diabetes_laboratorial"], comparaveis["autorrelato"])
tabela = tabela.reindex(["diabetes", "pre_diabetes", "normal"])
st.subheader("Tabela de confusão")
st.dataframe(tabela, use_container_width=True)

tp = tabela.loc["diabetes", "Sim"] if "Sim" in tabela.columns else 0
fn = (tabela.loc["normal", "Sim"] if "normal" in tabela.index else 0) + (tabela.loc["pre_diabetes", "Sim"] if "pre_diabetes" in tabela.index else 0)
fp = tabela.loc["diabetes", "Não"] if "Não" in tabela.columns else 0
tn = (tabela.loc["normal", "Não"] if "normal" in tabela.index else 0) + (tabela.loc["pre_diabetes", "Não"] if "pre_diabetes" in tabela.index else 0)

sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
especificidade = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
acuracia = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")

c1, c2, c3 = st.columns(3)
c1.metric("Sensibilidade", f"{sensibilidade:.1%}")
c2.metric("Especificidade", f"{especificidade:.1%}")
c3.metric("Acurácia", f"{acuracia:.1%}")

st.info(
    "Sensibilidade abaixo de 100% é consistente com subdiagnóstico de diabetes — fenômeno documentado na "
    "literatura epidemiológica, não um erro do classificador."
)

st.divider()
n_pre = tabela.loc["pre_diabetes"].sum() if "pre_diabetes" in tabela.index else 0
n_pre_sem_relato = tabela.loc["pre_diabetes", "Não"] if "pre_diabetes" in tabela.index and "Não" in tabela.columns else 0
if n_pre > 0:
    st.metric("Pré-diabetes (laboratorial) SEM autorrelato de diabetes", f"{100*n_pre_sem_relato/n_pre:.1f}%", help=f"{n_pre_sem_relato} de {n_pre} casos")

fig = px.bar(tabela.reset_index().melt(id_vars="status_diabetes_laboratorial"), x="status_diabetes_laboratorial", y="value", color="autorrelato", barmode="group", title="Classificação laboratorial x autorrelato")
st.plotly_chart(apply_default_layout(fig), use_container_width=True)
