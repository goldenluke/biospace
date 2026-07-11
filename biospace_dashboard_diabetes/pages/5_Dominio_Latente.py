import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.plugins.diabetes import InsulinResistanceProxyDomain

st.set_page_config(page_title="Domínio Latente - Diabetes", page_icon="🔬", layout="wide")

pipeline = require_pipeline()

st.title("🔬 Domínio Latente: Proxy de Resistência à Insulina")
st.warning(
    "⚠️ Este domínio NÃO observa nenhuma variável diretamente — é INFERIDO por Análise Fatorial a "
    "partir de GlycemicDomain + AnthropometricDomain. `is_validated = False`: sem HOMA-IR real ou "
    "clamp euglicêmico (o padrão-ouro) nesta base sintética para confirmar que o fator extraído "
    "mede resistência à insulina de fato. Trate como hipótese estatística, não fato clínico."
)

glycemic_domain = next(d for d in pipeline.representation.domains if d.name == "glycemic")
anthro_domain = next(d for d in pipeline.representation.domains if d.name == "anthropometric")

n_factors = st.radio("Número de fatores", [1, 2], horizontal=True)

if st.button("Ajustar InsulinResistanceProxyDomain", type="primary"):
    with st.spinner("Ajustando Análise Fatorial..."):
        proxy = InsulinResistanceProxyDomain(glycemic_domain, anthro_domain, n_factors=n_factors)
        proxy.fit(pipeline.cohort)
    st.session_state["_insulin_proxy"] = proxy

if "_insulin_proxy" in st.session_state:
    proxy = st.session_state["_insulin_proxy"]
    st.caption(f"Hipótese declarada: {proxy.hypothesis}")

    if proxy.n_factors == 1:
        top = proxy.top_loadings(n=10)
        df_load = pd.DataFrame(top, columns=["feature", "carga"])
        fig = px.bar(df_load, x="feature", y="carga", title="Cargas do fator único")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            top1 = proxy.top_loadings(factor_index=0, n=10)
            fig1 = px.bar(pd.DataFrame(top1, columns=["feature", "carga"]), x="feature", y="carga", title="Fator 1")
            st.plotly_chart(apply_default_layout(fig1), use_container_width=True)
        with col2:
            top2 = proxy.top_loadings(factor_index=1, n=10)
            fig2 = px.bar(pd.DataFrame(top2, columns=["feature", "carga"]), x="feature", y="carga", title="Fator 2")
            st.plotly_chart(apply_default_layout(fig2), use_container_width=True)

    st.divider()
    st.subheader("Distribuição do fator no paciente")
    ids = pipeline.space.ids()
    labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in ids}
    escolhido = st.selectbox("Paciente", ids, format_func=lambda s: labels_map[s])
    system = pipeline.cohort.systems[escolhido]
    valores = proxy.transform(system)
    for f in valores:
        st.metric(f.name, f"{f.value:.3f}")
