import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import RepresentationSpace
from biospace.phenotyping import KMeansPhenotyper
from biospace.survival import build_discrete_time_to_event, fit_cox_model, kaplan_meier_by_group

st.set_page_config(page_title="Sobrevivência - UCI", page_icon="📈", layout="wide")

pipeline = require_pipeline()

st.title("📈 Análise de Sobrevivência (Kaplan-Meier / Cox)")
st.caption(
    "Pergunta diferente do achado publicado (fenótipo → ~2x readmissão, caracterização TRANSVERSAL sobre o "
    "estado mais recente). Aqui: fenotipando só pelo PRIMEIRO encontro de cada paciente (sem olhar pro futuro, "
    "Contrato de Temporalidade 5.7), a predição PROSPECTIVA de sobrevivência livre de readmissão precoce "
    "ainda é significativa, mas mais fraca — a tarefa é genuinamente mais difícil."
)
st.info("'Tempo' aqui é índice de encontro (ordem), não calendário real — a UCI não tem datas reais.")

df = pipeline.display_df
multi = df[df["n_encontros"] >= 2]
st.metric("Pacientes multi-encontro nesta amostra", len(multi))

if len(multi) < 30:
    st.warning("Poucos pacientes multi-encontro nesta amostra — use a base completa na página inicial para resultado robusto.")

if st.button("Fenotipar por baseline e ajustar sobrevivência", type="primary"):
    order = pipeline.representation.domain_names()

    with st.spinner("Extraindo tempo-até-evento, fenotipando pelo 1º encontro, ajustando Kaplan-Meier e Cox..."):
        resultado = build_discrete_time_to_event(pipeline.cohort, event_fn=lambda o: o.metadata.get("readmitted") == "<30")

        ids_multi = set(resultado.df["system_id"])
        space_baseline = RepresentationSpace(domain_order=order)
        for sid in ids_multi:
            if sid in pipeline.cohort.trajectories:
                space_baseline.add(pipeline.cohort.trajectories[sid].at(0))

        if len(space_baseline.ids()) < 20:
            st.error("Poucos pacientes multi-encontro nesta amostra para fenotipar com confiança.")
            st.stop()

        phenotyper = KMeansPhenotyper(n_clusters=min(4, len(space_baseline.ids()) // 5))
        phenotypes = phenotyper.fit(space_baseline)
        labels = {}
        for sid in space_baseline.ids():
            vec = space_baseline.get(sid).as_vector(order)
            labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

        df_surv = resultado.df.copy()
        df_surv["fenotipo_baseline"] = df_surv["system_id"].map(labels)
        df_surv = df_surv.dropna(subset=["fenotipo_baseline"])

        relatorio_km = kaplan_meier_by_group(df_surv, group_col="fenotipo_baseline")

        df_cox = pd.get_dummies(df_surv, columns=["fenotipo_baseline"], drop_first=True)
        cols_dummy = [c for c in df_cox.columns if c.startswith("fenotipo_baseline_")]
        relatorio_cox = None
        erro_cox = None
        if cols_dummy:
            try:
                relatorio_cox = fit_cox_model(df_cox, covariate_cols=cols_dummy)
            except Exception as e:
                erro_cox = str(e)

    st.session_state["_uci_survival"] = (df_surv, relatorio_km, relatorio_cox, erro_cox, resultado)

if "_uci_survival" in st.session_state:
    df_surv, relatorio_km, relatorio_cox, erro_cox, resultado = st.session_state["_uci_survival"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes incluídos (≥2 encontros)", resultado.n_included)
    c2.metric("Taxa de evento (readmissão <30 em algum momento)", f"{100*df_surv['event'].mean():.1f}%")
    c3.metric("Log-rank p-valor", f"{relatorio_km.logrank_p:.2e}")

    st.divider()
    st.subheader("Curvas de Kaplan-Meier por fenótipo de baseline")
    fig = go.Figure()
    for grupo, kmf in relatorio_km.fitters.items():
        sf = kmf.survival_function_
        fig.add_trace(go.Scatter(x=sf.index, y=sf.iloc[:, 0], mode="lines", name=f"{grupo} (n={relatorio_km.n_por_grupo[grupo]})"))
    fig.update_layout(title="Probabilidade de continuar sem readmissão precoce, por fenótipo de baseline", xaxis_title="Nº de encontros desde o baseline", yaxis_title="S(t)")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    st.write("**Mediana de sobrevivência por fenótipo:**")
    st.write({g: (f"{m:.1f} encontros" if m == m else "não atingida") for g, m in relatorio_km.median_survival.items()})

    if relatorio_cox:
        st.divider()
        st.subheader("Modelo de Cox")
        st.caption("Índice de concordância perto de 0,5 = quase aleatório. Achado documentado: predição prospectiva por baseline é bem mais fraca que a caracterização transversal publicada.")
        st.text(relatorio_cox.summary())
        st.dataframe(relatorio_cox.summary_df, use_container_width=True)
    elif erro_cox:
        st.divider()
        st.warning(
            "O modelo de Cox não convergiu nesta amostra — comum com amostras pequenas, onde um fenótipo pode "
            "ficar com poucos pacientes e 'separação completa' (o grupo prediz o evento perfeitamente, o que "
            "quebra a otimização numérica). Use a base completa na página inicial para um ajuste estável. "
            "As curvas de Kaplan-Meier acima continuam válidas independente disso."
        )
