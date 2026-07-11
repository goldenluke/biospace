import components._bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.filters import render_filters
from components.state import require_pipeline
from biospace.phenotyping import (
    ClinicalKMeansPhenotyper,
    GaussianMixturePhenotyper,
    HDBSCANPhenotyper,
    SpectralPhenotyper,
)

st.set_page_config(page_title="Perfis · BioSpace", page_icon="🧬", layout="wide")

pipeline = require_pipeline()
df = render_filters(pipeline.display_df)

st.title("🧬 Perfis (Fenotipagem Clínica Automática)")
st.caption(
    "ClinicalKMeansPhenotyper: K escolhido automaticamente por silhouette, fenótipos "
    "rotulados por severidade dos centróides. Os fenótipos são regiões do "
    "RepresentationSpace — o algoritmo apenas os estima (Seção 8 da teoria)."
)

if len(df) == 0:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

st.subheader("Varredura de K (silhouette)")
elbow_df = pd.DataFrame([{"K": r.k, "inertia": r.inertia, "silhouette": r.silhouette} for r in pipeline.phenotyper.elbow_table])

col1, col2 = st.columns(2)
with col1:
    fig = px.line(elbow_df, x="K", y="silhouette", markers=True, title="Silhouette por K")
    fig.add_vline(x=pipeline.phenotyper.best_k, line_dash="dash", line_color="red")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with col2:
    fig = px.line(elbow_df, x="K", y="inertia", markers=True, title="Inércia por K (curva de cotovelo)")
    fig.add_vline(x=pipeline.phenotyper.best_k, line_dash="dash", line_color="red")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.info(f"K escolhido automaticamente: **{pipeline.phenotyper.best_k}** (maior silhouette)")

st.divider()

st.subheader("Comparar com outro algoritmo de fenotipagem")
st.caption(
    "Não altera a fenotipagem oficial acima — todos operam sobre o MESMO RepresentationSpace "
    "(Operator, Seção 12.2 da teoria: o algoritmo nunca vê dados brutos)."
)

algoritmo = st.selectbox(
    "Algoritmo", ["K-Means (K fixo)", "HDBSCAN (densidade)", "Gaussian Mixture (BIC automático)", "Spectral Clustering"]
)

col_a, col_b = st.columns(2)
with col_a:
    if algoritmo == "K-Means (K fixo)":
        k_manual = st.slider("K", 2, 10, value=pipeline.phenotyper.best_k)
    elif algoritmo == "HDBSCAN (densidade)":
        min_cluster_size = st.slider("min_cluster_size", 2, 30, value=5)
        st.caption("⚠️ Valores altos em espaços de muitas dimensões tendem a classificar tudo como ruído.")
    elif algoritmo == "Gaussian Mixture (BIC automático)":
        st.caption("Nº de componentes escolhido automaticamente por BIC, sem parâmetro manual.")
    else:
        n_clusters_spectral = st.slider("n_clusters", 2, 10, value=4)

with col_b:
    executar = st.button("Rodar este algoritmo", type="primary")

if executar:
    with st.spinner("Recalculando sobre o mesmo RepresentationSpace..."):
        if algoritmo == "K-Means (K fixo)":
            operator = ClinicalKMeansPhenotyper(k_range=[k_manual])
            label = f"K-Means (K={k_manual})"
        elif algoritmo == "HDBSCAN (densidade)":
            operator = HDBSCANPhenotyper(min_cluster_size=min_cluster_size)
            label = f"HDBSCAN (min_cluster_size={min_cluster_size})"
        elif algoritmo == "Gaussian Mixture (BIC automático)":
            operator = GaussianMixturePhenotyper()
            label = "Gaussian Mixture"
        else:
            operator = SpectralPhenotyper(n_clusters=n_clusters_spectral)
            label = f"Spectral (n_clusters={n_clusters_spectral})"

        phenotypes_alt = operator.fit(pipeline.space)

    st.session_state["_alt_phenotypes"] = phenotypes_alt
    st.session_state["_alt_label"] = label
    st.session_state["_alt_operator"] = operator

if "_alt_phenotypes" in st.session_state:
    phenotypes_alt = st.session_state["_alt_phenotypes"]
    label = st.session_state["_alt_label"]
    operator = st.session_state["_alt_operator"]
    order = pipeline.space.order()

    membership = {}
    for sid in pipeline.space.ids():
        vec = pipeline.space.get(sid).as_vector(order)
        membership[sid] = next((ph.name for ph in phenotypes_alt if ph.contains(vec)), None)

    n_sem_fenotipo = sum(1 for v in membership.values() if v is None)
    if n_sem_fenotipo:
        st.warning(f"{n_sem_fenotipo} paciente(s) sem fenótipo atribuído (ruído, no caso do HDBSCAN).")

    counts = pd.Series(membership).dropna().value_counts()
    if len(counts) > 0:
        fig = px.bar(x=counts.index, y=counts.values, labels={"x": f"Fenótipo ({label})", "y": "n"})
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    else:
        st.error("Nenhum fenótipo estimado com esta configuração.")

st.divider()

st.subheader("Perfil de cada fenótipo por domínio (z-score médio)")
st.caption(
    "Cada barra é a média, dentro do fenótipo, do z-score já ponderado por completude de "
    "cada domínio — mesmos eixos usados pela geometria (maior = mais grave, na maioria dos domínios)."
)

domain_names = pipeline.representation.domain_names()
rows = []
for ph in pipeline.phenotypes:
    members = [sid for sid in pipeline.space.ids() if ph.contains(pipeline.space.get(sid).as_vector(pipeline.space.order()))]
    if not members:
        continue
    for domain_name in domain_names:
        domain_values = []
        for sid in members:
            rep = pipeline.space.get(sid)
            if domain_name in rep.components:
                raw_features = rep.components[domain_name]
                if raw_features and not hasattr(raw_features[0], "value"):
                    st.error(
                        f"`RepresentationVector.components[{domain_name!r}]` contém "
                        f"{type(raw_features[0]).__name__}, não objetos Feature. Sua pasta "
                        "`biospace/` está com versões incompatíveis entre si — substitua toda a "
                        "pasta pela versão mais recente (não copie arquivos individualmente)."
                    )
                    st.stop()
                feature_values = [f.value for f in raw_features]
                domain_values.append(float(np.mean(feature_values)))
        if domain_values:
            rows.append({"fenotipo": ph.name, "dominio": domain_name, "z_medio": float(np.mean(domain_values))})

profile_df = pd.DataFrame(rows)
if not profile_df.empty:
    fig = px.bar(
        profile_df, x="dominio", y="z_medio", color="fenotipo", barmode="group",
        labels={"z_medio": "z-score médio (ponderado)", "dominio": "Domínio"},
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()

st.subheader("Fenótipo × classe de apneia (validação cruzada)")
crosstab = pd.crosstab(df["fenotipo"], df["classe_apneia"], normalize="index").round(3) * 100
st.dataframe(crosstab.style.format("{:.1f}%"), use_container_width=True)

st.subheader("Tabela completa por fenótipo")
resumo = df.groupby("fenotipo")[["idade", "imc", "ido", "spo2_minima", "fc_media_bpm"]].mean().round(1)
resumo["n"] = df.groupby("fenotipo").size()
st.dataframe(resumo, use_container_width=True)

st.divider()

st.subheader("Transição de fenótipos ao longo do tempo")
st.caption(
    "TransitionAnalyzer: matriz de transição entre exames consecutivos do MESMO paciente, "
    "com filtro de intervalo mínimo entre exames (evita confundir reexames no mesmo período "
    "com uma verdadeira transição de acompanhamento)."
)

from datetime import timedelta

from biospace.longitudinal import TransitionAnalyzer

gap_dias = st.slider("Intervalo mínimo entre exames consecutivos (dias)", 0, 180, value=30)
analyzer = TransitionAnalyzer(pipeline.phenotypes, order=pipeline.space.order())
P, names = analyzer.matrix(pipeline.cohort, min_gap=timedelta(days=gap_dias))

if len(names) > 0:
    matrix_df = pd.DataFrame(P, index=names, columns=names)
    fig = px.imshow(
        matrix_df, text_auto=".2f", color_continuous_scale="Blues",
        labels={"x": "Fenótipo destino", "y": "Fenótipo origem", "color": "P(destino|origem)"},
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    st.caption("Tempo até transição observado (dias), top 10 por frequência:")
    summary = analyzer.summary(pipeline.cohort, min_gap=timedelta(days=gap_dias))
    summary_rows = [
        {"de": a, "para": b, "n": s["n"], "media_dias": round(s["media_dias"], 1), "mediana_dias": round(s["mediana_dias"], 1)}
        for (a, b), s in sorted(summary.items(), key=lambda kv: -kv[1]["n"])[:10]
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
else:
    st.info("Sem transições suficientes com este filtro de tempo (a maioria dos pacientes tem apenas 1 exame).")
