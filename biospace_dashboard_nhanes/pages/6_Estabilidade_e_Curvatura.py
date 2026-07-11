import components._bootstrap  # noqa: F401

import random

import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from components.charts import apply_default_layout
from components.state import require_pipeline

st.set_page_config(page_title="Estabilidade e Curvatura - NHANES", page_icon="🧬", layout="wide")

pipeline = require_pipeline()

st.title("🧬 Estabilidade Fenotípica e Curvatura Estrutural")
st.caption(
    "A mesma metodologia aplicada em SAOS (varredura de estabilidade, curvatura de Ollivier-Ricci), "
    "testada aqui pela primeira vez. Achado central: o NHANES é o oposto de SAOS na estabilidade."
)

order = pipeline.representation.domain_names()
space = pipeline.space
ids = space.ids()

tab1, tab2 = st.tabs(["Estabilidade x Idade", "Curvatura Estrutural"])

with tab1:
    st.subheader("Estabilidade fenotípica: com e sem idade")
    st.info(
        "Em SAOS, nenhuma configuração cruzava ARI≥0,7. Aqui, K-Means chega a 0,957 em K=2. "
        "Testamos se isso é só idade disfarçada de estrutura metabólica removendo a Feature e repetindo."
    )

    if st.button("Rodar comparação (com/sem idade)", type="primary"):
        with st.spinner("Ajustando K-Means em metades independentes da coorte, com e sem idade..."):
            matriz_completa = np.stack([space.get(sid).as_vector(order) for sid in ids])
            nomes_features = []
            for dom in order:
                exemplo = space.get(ids[0])
                for f in exemplo.components.get(dom, []):
                    nomes_features.append(f"{dom}.{f.name}")
            idx_idade = nomes_features.index("anthropometric.idade")
            matriz_sem_idade = np.delete(matriz_completa, idx_idade, axis=1)

            def stability_ari(matriz, k, seed=42):
                n = matriz.shape[0]
                rng = np.random.default_rng(seed)
                shuffled = rng.permutation(n)
                split = n // 2
                idx1, idx2 = shuffled[:split], shuffled[split:]
                km1 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matriz[idx1])
                km2 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matriz[idx2])
                return adjusted_rand_score(km1.predict(matriz), km2.predict(matriz))

            resultados = []
            for k in range(2, 7):
                resultados.append({"K": k, "com_idade": stability_ari(matriz_completa, k), "sem_idade": stability_ari(matriz_sem_idade, k)})
        st.session_state["_nhanes_stability_age"] = resultados

    if "_nhanes_stability_age" in st.session_state:
        import pandas as pd

        df_resultado = pd.DataFrame(st.session_state["_nhanes_stability_age"])
        df_melt = df_resultado.melt(id_vars="K", var_name="condicao", value_name="ARI")
        fig = px.bar(df_melt, x="K", y="ARI", color="condicao", barmode="group", title="Estabilidade (ARI) por K, com e sem idade")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="limiar de estabilidade")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
        st.dataframe(df_resultado, use_container_width=True, hide_index=True)
        st.success(
            "Em K=2, a estabilidade é alta com OU sem idade — estrutura metabólica genuína além da idade. "
            "Em K=3, a estabilidade desaba sem idade — ali, idade sustentava a partição sozinha."
        )

with tab2:
    st.subheader("Curvatura estrutural (Ollivier-Ricci): dentro vs. entre fenótipos")
    st.warning(
        "Achado NEGATIVO documentado: diferente de SAOS (p=5,7e-19), aqui a diferença NÃO é "
        "significativa. Interpretação: a assinatura de curvatura parece específica de fronteiras "
        "frágeis em contínuos mal separados — ausente quando o fenótipo já é bem separado (como aqui)."
    )
    n_amostra = st.slider("Tamanho da amostra (grafo de similaridade é caro)", 500, 3000, value=1500, step=500)

    if st.button("Construir grafo e calcular curvatura"):
        from scipy import stats

        from biospace.core import RepresentationSpace
        from biospace.geometry import Euclidean, graph_curvature_summary
        from biospace.graph import build_cohort_similarity_graph
        from biospace.phenotyping import KMeansPhenotyper

        with st.spinner(f"Construindo grafo de similaridade (amostra de {n_amostra})..."):
            random.seed(0)
            ids_amostra = random.sample(ids, min(n_amostra, len(ids)))
            space_amostra = RepresentationSpace(domain_order=order)
            for sid in ids_amostra:
                space_amostra.add(space.get(sid))

            grafo = build_cohort_similarity_graph(space_amostra, Euclidean(), k=8, order=order)
            resumo = graph_curvature_summary(grafo, weight="weight")

            phenotyper = KMeansPhenotyper(n_clusters=2)
            phenotypes = phenotyper.fit(space_amostra)
            labels = {}
            for sid in ids_amostra:
                vec = space_amostra.get(sid).as_vector(order)
                labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

            dentro, entre = [], []
            for (u, v), k in resumo["edge_curvatures"].items():
                if labels.get(u) and labels.get(v):
                    (dentro if labels[u] == labels[v] else entre).append(k)

            _, p = stats.mannwhitneyu(dentro, entre, alternative="greater") if dentro and entre else (None, float("nan"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Dentro do mesmo fenótipo", f"{np.mean(dentro):.4f}", help=f"n={len(dentro)} arestas")
        c2.metric("Entre fenótipos diferentes", f"{np.mean(entre):.4f}", help=f"n={len(entre)} arestas")
        c3.metric("p-valor (Mann-Whitney)", f"{p:.2e}")

        if p > 0.05:
            st.info("Confirma o achado documentado: diferença NÃO significativa (diferente de SAOS).")
        else:
            st.warning("Esta execução deu diferença significativa — diferente do achado documentado anteriormente, vale investigar.")
