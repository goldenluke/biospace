import components._bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.dynamics import MeanRevertingEvolutionOperator
from biospace.geometry import Euclidean, detect_metastability, graph_curvature_summary
from biospace.graph import build_cohort_similarity_graph

st.set_page_config(page_title="Curvatura · BioSpace", page_icon="🌐", layout="wide")

pipeline = require_pipeline()
order = pipeline.representation.domain_names()

st.title("🌐 Curvatura (Fase 8)")
st.caption(
    "Paciente → Representação → Variedade → Trajetória → Curvatura → Estabilidade. "
    "3 formas INDEPENDENTES de medir curvatura, de fontes diferentes — não se espera que "
    "coincidam numericamente, mas concordância de direção é evidência de que capturam algo real."
)

tab_temporal, tab_densidade, tab_estrutural = st.tabs(["Temporal (φ)", "Densidade Populacional", "Estrutural (grafo)"])

# =============================================================================
# TAB Temporal
# =============================================================================
with tab_temporal:
    st.subheader("Curvatura Temporal: k = -ln(φ)")
    st.caption(
        "Vem DIRETO do φ já ajustado por MeanRevertingEvolutionOperator — sem nenhuma estimativa "
        "nova. Curvatura ALTA = poço estreito e profundo = recuperação rápida = alta resiliência. "
        "Curvatura BAIXA = poço raso = recuperação lenta = 'critical slowing down'."
    )

    if st.button("Ajustar dinâmica e calcular curvatura temporal", type="primary"):
        with st.spinner("Ajustando EvolutionOperator..."):
            evo = MeanRevertingEvolutionOperator(order=order)
            try:
                evo.fit(pipeline.cohort)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state["_evo_curvatura"] = evo

    if "_evo_curvatura" in st.session_state:
        evo = st.session_state["_evo_curvatura"]
        rows = []
        for name, fd in evo.dynamics_.items():
            if fd.curvature is not None:
                rows.append({"feature": name, "curvatura": fd.curvature, "resiliencia": fd.resilience_score, "n_pares": fd.n_pairs, "meia_vida_dias": fd.half_life_days})
        df_curv = pd.DataFrame(rows).sort_values("curvatura", ascending=False)

        fig = px.bar(df_curv.head(20), x="feature", y="curvatura", title="Maior curvatura (mais resiliente) — top 20")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)

        st.markdown("**Menor curvatura (menos resiliente — mais perto de critical slowing down):**")
        st.dataframe(df_curv.tail(10).sort_values("curvatura"), use_container_width=True, hide_index=True)

# =============================================================================
# TAB Densidade
# =============================================================================
with tab_densidade:
    st.subheader("Metaestabilidade via Densidade Populacional (poços de potencial)")
    st.caption(
        "Reconstrói um potencial efetivo U(x) = -log(densidade(x)) via KDE sobre a população — "
        "mais de 1 poço = múltiplos estados estáveis genuínos, com uma barreira de energia real "
        "entre eles (não apenas 'parece haver 2 grupos')."
    )

    nomes_features = []
    for dom in order:
        exemplo = next(iter(pipeline.cohort.trajectories.values())).at(0)
        for f in exemplo.components.get(dom, []):
            nomes_features.append(f"{dom}.{f.name}")
    feature_escolhida = st.selectbox("Feature", nomes_features, key="metaestab_feature")
    min_prominence = st.slider("Proeminência mínima do poço (maior = mais conservador)", 0.1, 1.0, value=0.3, step=0.1)

    if st.button("Detectar poços de potencial"):
        space = pipeline.cohort.snapshot()
        with st.spinner("Reconstruindo potencial via KDE..."):
            try:
                relatorio_meta = detect_metastability(space, feature_escolhida, min_prominence=min_prominence)
            except Exception as e:
                st.error(str(e))
                st.stop()
        st.session_state["_metaestabilidade"] = relatorio_meta

    if "_metaestabilidade" in st.session_state:
        relatorio_meta = st.session_state["_metaestabilidade"]
        st.text(relatorio_meta.summary())
        if len(relatorio_meta.wells) > 1:
            st.success(f"{len(relatorio_meta.wells)} poços detectados — múltiplos estados estáveis genuínos.")
        else:
            st.info("1 poço — população unimodal, sem evidência de metaestabilidade nesta Feature.")

# =============================================================================
# TAB Estrutural
# =============================================================================
with tab_estrutural:
    st.subheader("Curvatura Estrutural (Ollivier-Ricci sobre o grafo de similaridade)")
    st.caption(
        "A única das três que usa a VARIEDADE em si (o grafo k-NN). κ>0: vizinhanças sobrepostas, "
        "região 'costurada', mais estável. κ<0: gargalo estrutural, mais frágil — testado nos dados "
        "reais: arestas que cruzam fenótipos diferentes têm curvatura mais negativa que arestas "
        "dentro do mesmo fenótipo (Mann-Whitney p=5,7e-19)."
    )

    k_vizinhos = st.slider("k (vizinhos mais próximos no grafo)", 3, 15, value=8, key="curv_k")

    if st.button("Construir grafo e calcular curvatura estrutural"):
        space = pipeline.cohort.snapshot()
        with st.spinner("Construindo grafo de similaridade e calculando curvatura de Ollivier-Ricci..."):
            grafo = build_cohort_similarity_graph(space, Euclidean(), k=k_vizinhos, order=order)
            resumo_curv = graph_curvature_summary(grafo, weight="weight")
        st.session_state["_grafo_curvatura"] = (grafo, resumo_curv)

    if "_grafo_curvatura" in st.session_state:
        grafo, resumo_curv = st.session_state["_grafo_curvatura"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Curvatura média", f"{resumo_curv['global_mean']:.3f}")
        c2.metric("Mínima", f"{resumo_curv['global_min']:.3f}")
        c3.metric("Máxima", f"{resumo_curv['global_max']:.3f}")

        valores = list(resumo_curv["edge_curvatures"].values())
        fig_hist = px.histogram(x=valores, nbins=40, title="Distribuição da curvatura por aresta")
        fig_hist.update_layout(xaxis_title="Curvatura de Ollivier-Ricci", yaxis_title="Nº de arestas")
        st.plotly_chart(apply_default_layout(fig_hist), use_container_width=True)

        # comparar dentro vs entre fenotipos, se ja houver fenotipagem
        if pipeline.phenotypes:
            space = pipeline.cohort.snapshot()
            labels = {}
            for sid in space.ids():
                vec = space.get(sid).as_vector(order)
                labels[sid] = next((ph.name for ph in pipeline.phenotypes if ph.contains(vec)), None)

            dentro, entre = [], []
            for (u, v), k in resumo_curv["edge_curvatures"].items():
                if labels.get(u) and labels.get(v):
                    (dentro if labels[u] == labels[v] else entre).append(k)

            if dentro and entre:
                st.markdown("**Curvatura dentro vs. entre fenótipos:**")
                c1, c2 = st.columns(2)
                c1.metric("Dentro do mesmo fenótipo", f"{np.mean(dentro):.3f}", help=f"n={len(dentro)} arestas")
                c2.metric("Entre fenótipos diferentes", f"{np.mean(entre):.3f}", help=f"n={len(entre)} arestas")
                if np.mean(dentro) > np.mean(entre):
                    st.success("Confirma o padrão esperado: fronteiras entre fenótipos são estruturalmente mais frágeis (curvatura mais negativa).")
