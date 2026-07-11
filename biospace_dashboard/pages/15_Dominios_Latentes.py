import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.plugins.sleep.latent import AutonomicBalanceProxyDomain, FrailtyProxyDomain, InflammationProxyDomain

st.set_page_config(page_title="Domínios Latentes · BioSpace", page_icon="🧬", layout="wide")

pipeline = require_pipeline()

st.title("🧬 Domínios Latentes")
st.warning(
    "⚠️ Nenhum destes domínios observa uma variável diretamente — todos são INFERIDOS por Análise "
    "Fatorial a partir de outros domínios. `is_validated = False` em todos: não há biomarcador "
    "independente nesta planilha (PCR, IL-6, HRV real, educação/QI) contra o qual confirmar que o "
    "fator mede de fato o que o nome sugere. Trate como hipótese estatística, não fato clínico."
)

domains = {d.name: d for d in pipeline.representation.domains}


def _plot_loadings(domain, titulo: str, factor_index: int = 0):
    top = domain.top_loadings(factor_index=factor_index, n=10)
    df = pd.DataFrame(top, columns=["feature", "carga"])
    fig = px.bar(df, x="feature", y="carga", title=titulo)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)


tab1, tab2, tab3 = st.tabs(["Inflamação (proxy)", "Fragilidade (proxy)", "Balanço Autonômico (proxy)"])

with tab1:
    st.caption(
        "Fontes: hipoxemia + cardiovascular + antropometria. **Achado real deste projeto**: com "
        "1 fator, o resultado é dominado por hipoxemia (a contribuição cardiovascular fica perto "
        "de zero) — por isso o padrão aqui é comparar 1 vs. 2 fatores."
    )
    n_factors_infl = st.radio("Nº de fatores", [1, 2], horizontal=True, key="infl_n")
    if st.button("Ajustar InflammationProxyDomain"):
        with st.spinner("Ajustando Análise Fatorial..."):
            dom = InflammationProxyDomain(domains["hypoxia"], domains["cardiovascular"], domains["anthropometric"], n_factors=n_factors_infl)
            dom.fit(pipeline.cohort)
        st.session_state["_infl_dom"] = dom

    if "_infl_dom" in st.session_state:
        dom = st.session_state["_infl_dom"]
        if dom.n_factors == 1:
            _plot_loadings(dom, "Fator único — cargas por Feature")
        else:
            col1, col2 = st.columns(2)
            with col1:
                _plot_loadings(dom, "Fator 1", factor_index=0)
            with col2:
                _plot_loadings(dom, "Fator 2", factor_index=1)

with tab2:
    st.caption(
        "Fontes: antropometria + comorbidade + arquitetura do sono + sintomas. Cobre só o "
        "componente de EXAUSTÃO do fenótipo de fragilidade clássico (Fried et al., 2001) — marcha, "
        "preensão e perda de peso não existem nesta planilha."
    )
    if st.button("Ajustar FrailtyProxyDomain"):
        with st.spinner("Ajustando Análise Fatorial..."):
            dom = FrailtyProxyDomain(domains["anthropometric"], domains["comorbidity"], domains["sleep_architecture"], domains["symptoms"])
            dom.fit(pipeline.cohort)
        st.session_state["_frailty_dom"] = dom

    if "_frailty_dom" in st.session_state:
        _plot_loadings(st.session_state["_frailty_dom"], "Fator único — cargas por Feature")

with tab3:
    st.caption(
        "Fontes: cardiovascular + hipoxemia. Sem HRV real (SDNN/RMSSD) nesta planilha — só FC "
        "mínima/média/máxima como proxy grosseiro. Padrão com 2 fatores: Fator 1 = eixo "
        "cardiovascular genuíno; Fator 2 = redundante com HypoxiaDomain."
    )
    if st.button("Ajustar AutonomicBalanceProxyDomain"):
        with st.spinner("Ajustando Análise Fatorial..."):
            dom = AutonomicBalanceProxyDomain(domains["cardiovascular"], domains["hypoxia"])
            dom.fit(pipeline.cohort)
        st.session_state["_auto_dom"] = dom

    if "_auto_dom" in st.session_state:
        dom = st.session_state["_auto_dom"]
        col1, col2 = st.columns(2)
        with col1:
            _plot_loadings(dom, "Fator 1 (eixo cardiovascular)", factor_index=0)
        with col2:
            _plot_loadings(dom, "Fator 2 (redundante com hipoxemia)", factor_index=1)
