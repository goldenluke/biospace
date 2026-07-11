import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Qualidade de Dados · BioSpace", page_icon="🔍", layout="wide")

pipeline = require_pipeline()

st.title("🔍 Qualidade de Dados")
st.caption(
    "Ausência estrutural por domínio e o peso de completude aplicado por _ReferenceDomain "
    "(Contrato 5.1 — Rastreabilidade: nada é imputado silenciosamente)."
)

n_exams = sum(len(traj) for traj in pipeline.cohort.trajectories.values())
rows = []
for domain in pipeline.representation.domains:
    if not hasattr(domain, "missing_counts"):
        continue
    weights = domain.feature_weights() if hasattr(domain, "feature_weights") else {}
    for key, count in domain.missing_counts.items():
        rows.append(
            {
                "dominio": domain.name,
                "campo": key,
                "n_ausente": count,
                "pct_ausente": 100 * count / n_exams if n_exams else 0.0,
                "peso_aplicado": weights.get(key, 1.0),
                "excluido": weights.get(key, 1.0) == 0.0,
            }
        )

if not rows:
    st.success("Nenhuma ausência detectada nos campos numéricos desta população.")
else:
    quality_df = pd.DataFrame(rows).sort_values("pct_ausente", ascending=False)

    excluidos = quality_df[quality_df["excluido"]]
    if len(excluidos) > 0:
        st.warning(
            f"{len(excluidos)} campo(s) excluído(s) automaticamente (completude abaixo do limiar): "
            + ", ".join(excluidos["campo"].tolist())
        )

    fig = px.bar(
        quality_df, x="campo", y="pct_ausente", color="dominio",
        labels={"pct_ausente": "% ausente (sobre exames)", "campo": "Campo"},
        title="Ausência por campo, agrupado por domínio",
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    st.subheader("Tabela completa (peso aplicado à geometria por campo)")
    st.dataframe(
        quality_df[["dominio", "campo", "n_ausente", "pct_ausente", "peso_aplicado", "excluido"]]
        .style.format({"pct_ausente": "{:.1f}%", "peso_aplicado": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Relatório de carga")
report = getattr(pipeline.cohort, "loader_report", {})
st.json(report)
