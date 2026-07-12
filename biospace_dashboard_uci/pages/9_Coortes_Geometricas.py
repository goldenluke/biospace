import components._bootstrap  # noqa: F401

import numpy as np
import streamlit as st

from components.state import require_pipeline
from biospace.geometry import Euclidean, cohort_around

st.set_page_config(page_title="Coortes Geométricas - UCI", page_icon="🧭", layout="wide")

pipeline = require_pipeline()

st.title("🧭 Coortes Automáticas por Proximidade Geométrica")
st.caption(
    "Coortes deixam de ser consultas SQL (limiares sobre uma Feature de cada vez) e passam a ser "
    "subconjuntos do espaço de representação — pacientes mais próximos de um PONTO de consulta, que pode "
    "ser um paciente real ou um ponto arbitrário (ex.: o centroide de um fenótipo)."
)

df = pipeline.display_df
fenotipos_disponiveis = sorted(df["fenotipo"].dropna().unique().tolist())

if not fenotipos_disponiveis:
    st.error("Nenhum fenótipo disponível nesta amostra.")
    st.stop()

fenotipo_escolhido = st.selectbox("Fenótipo de referência (o centroide dele vira o ponto de consulta)", fenotipos_disponiveis)
k = st.slider("k (nº de pacientes na coorte geométrica)", 10, min(2000, len(df) - 1), value=min(200, len(df) - 1))

if st.button("Construir coorte geométrica em torno do centroide", type="primary"):
    order = pipeline.representation.domain_names()
    ids_fenotipo = set(df[df["fenotipo"] == fenotipo_escolhido]["id"])

    with st.spinner("Calculando centroide e consultando k-mais-próximos..."):
        matriz_fenotipo = np.stack([pipeline.space.get(sid).as_vector(order) for sid in ids_fenotipo])
        centroide = matriz_fenotipo.mean(axis=0)

        geom = Euclidean()
        coorte_geometrica = cohort_around(pipeline.space, geom, query=centroide, order=order, k=k)
        comparacao = coorte_geometrica.overlap_with(ids_fenotipo)

        def taxa_readmissao(ids):
            validos = [sid for sid in ids if sid in pipeline.cohort.systems]
            if not validos:
                return None
            n_precoce = sum(1 for sid in validos if pipeline.cohort.systems[sid].observations[-1].metadata.get("readmitted") == "<30")
            return n_precoce / len(validos)

        taxa_geom = taxa_readmissao(set(coorte_geometrica.member_ids))
        taxa_original = taxa_readmissao(ids_fenotipo)
        taxa_geral = taxa_readmissao(set(df["id"]))

    st.session_state["_uci_geo_cohort"] = (comparacao, taxa_geom, taxa_original, taxa_geral, len(ids_fenotipo), k)

if "_uci_geo_cohort" in st.session_state:
    comparacao, taxa_geom, taxa_original, taxa_geral, n_original, k_usado = st.session_state["_uci_geo_cohort"]

    st.divider()
    st.subheader("Sobreposição: coorte geométrica vs. fenótipo original")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tamanho da coorte geométrica", k_usado)
    c2.metric("Tamanho do fenótipo original", n_original)
    c3.metric("Índice de Jaccard", f"{comparacao['jaccard']:.3f}")

    st.write(f"**{comparacao['n_intersecao']}** pacientes em comum de **{comparacao['n_geometrica']} + {comparacao['n_outra']} - interseção**.")
    if comparacao["jaccard"] < 0.3:
        st.warning(
            "Sobreposição baixa — esperado quando o fenótipo vem de K-Means: o cluster particiona por Voronoi entre "
            "TODOS os centroides simultaneamente, enquanto a consulta geométrica só considera distância a ESTE "
            "centroide. São mecanismos de definição de coorte genuinamente diferentes, não aproximações um do outro."
        )

    st.divider()
    st.subheader("Taxa de readmissão precoce em cada definição de coorte")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coorte geométrica", f"{100*taxa_geom:.2f}%" if taxa_geom is not None else "—")
    c2.metric("Fenótipo original (K-Means)", f"{100*taxa_original:.2f}%" if taxa_original is not None else "—")
    c3.metric("População geral (referência)", f"{100*taxa_geral:.2f}%" if taxa_geral is not None else "—")
