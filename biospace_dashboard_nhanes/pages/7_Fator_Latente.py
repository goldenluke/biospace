import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.latent.factor_analysis import FactorAnalysisLatentDomain

st.set_page_config(page_title="Fator Latente - NHANES", page_icon="🧬", layout="wide")

pipeline = require_pipeline()

st.title("🧬 Fator Latente: Carga Metabólica")
st.caption(
    "Análise Fatorial de verdade (não uma combinação linear arbitrária) sobre os domínios "
    "glycemic+cardiovascular+anthropometric — extrai o eixo de variação COMPARTILHADA entre eles."
)
st.warning(
    "**Achado documentado**: o fator extraído fica dominado por adiposidade (circunferência abdominal e "
    "IMC, cargas muito maiores que HbA1c ou pressão diastólica) — não distribuído igualmente entre os 3 "
    "domínios como se poderia assumir. Mesmo assim, mostra gradiente forte com status de diabetes (não "
    "usado na construção do fator): normal < pré-diabetes < diabetes."
)

if st.button("Ajustar Análise Fatorial e testar contra status de diabetes", type="primary"):
    with st.spinner("Ajustando Análise Fatorial sobre a coorte..."):
        glycemic = next(d for d in pipeline.representation.domains if d.name == "glycemic")
        cardio = next(d for d in pipeline.representation.domains if d.name == "cardiovascular")
        anthro = next(d for d in pipeline.representation.domains if d.name == "anthropometric")

        class _CargaMetabolica(FactorAnalysisLatentDomain):
            name = "carga_metabolica"
            hypothesis = "Fator compartilhado entre controle glicêmico, pressão arterial e adiposidade -- hipótese de um eixo geral de risco cardiometabólico."
            n_factors = 1

        fator = _CargaMetabolica([glycemic, cardio, anthro])
        fator.fit(pipeline.cohort)

        scores = {}
        for sid in pipeline.cohort.systems:
            features = fator.transform(pipeline.cohort.systems[sid])
            scores[sid] = features[0].value

        top_cargas = fator.top_loadings(n=6)

    st.session_state["_nhanes_fator"] = (scores, top_cargas)

if "_nhanes_fator" in st.session_state:
    scores, top_cargas = st.session_state["_nhanes_fator"]

    st.divider()
    st.subheader("Cargas do fator (quais Features mais contribuem)")
    df_cargas = pd.DataFrame(top_cargas, columns=["feature", "carga"]).sort_values("carga", key=abs, ascending=True)
    fig_cargas = px.bar(df_cargas, x="carga", y="feature", orientation="h", title="Cargas do fator 'carga_metabolica' (Análise Fatorial)")
    st.plotly_chart(apply_default_layout(fig_cargas), use_container_width=True)

    st.divider()
    st.subheader("O fator distingue status de diabetes? (validação externa, não usada no ajuste)")
    df = pipeline.display_df.copy()
    df["score_fator"] = df["id"].map(scores)
    df_valido = df[df["status_diabetes_laboratorial"] != "indeterminado"]

    fig_box = px.box(df_valido, x="status_diabetes_laboratorial", y="score_fator",
                      category_orders={"status_diabetes_laboratorial": ["normal", "pre_diabetes", "diabetes"]},
                      title="Score do fator 'carga metabólica', por status de diabetes laboratorial")
    st.plotly_chart(apply_default_layout(fig_box), use_container_width=True)

    medias = df_valido.groupby("status_diabetes_laboratorial")["score_fator"].mean()
    st.write("**Score médio por grupo:**")
    st.write(medias.reindex(["normal", "pre_diabetes", "diabetes"]))
