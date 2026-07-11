import components._bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import RepresentationSpace
from biospace.geometry import (
    DTW,
    Cosine,
    Euclidean,
    GromovWasserstein,
    InformationGeometry,
    LearnedGeometry,
    Mahalanobis,
    PhenotypeConditionedGeometry,
    RiemannianGeometry,
    Wasserstein,
)
from biospace.plugins.sleep import classificar_apneia

st.set_page_config(page_title="Geometrias · BioSpace", page_icon="📐", layout="wide")

pipeline = require_pipeline()

st.title("📐 Geometrias")
st.caption(
    "Distâncias diferentes contam histórias diferentes sobre o mesmo RepresentationSpace. "
    "Nenhuma é 'a' distância correta — cada uma responde a uma pergunta diferente."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Pontuais (entre 2 pacientes)", "Trajetória (DTW / Gromov-Wasserstein)", "Aprendida (NCA)", "Riemanniana / Geometria da Doença"]
)

# =============================================================================
# TAB 1 — Geometrias pontuais
# =============================================================================
with tab1:
    st.subheader("Comparar 2 pacientes sob diferentes geometrias")
    ids = pipeline.space.ids()
    labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in ids}

    col1, col2 = st.columns(2)
    with col1:
        sid_a = st.selectbox("Paciente A", ids, format_func=lambda s: labels_map[s], key="geo_a")
    with col2:
        sid_b = st.selectbox("Paciente B", ids, index=min(1, len(ids) - 1), format_func=lambda s: labels_map[s], key="geo_b")

    order = pipeline.space.order()
    x = pipeline.space.get(sid_a).as_vector(order)
    y = pipeline.space.get(sid_b).as_vector(order)

    matrix, _ = pipeline.space.matrix()
    cov = np.cov(matrix, rowvar=False)

    geometrias = {
        "Euclidiana": Euclidean(),
        "Mahalanobis": Mahalanobis(cov),
        "Cosine": Cosine(),
        "Wasserstein": Wasserstein(),
        "Information (Fisher-Rao)": InformationGeometry(),
    }

    rows = [{"geometria": nome, "distância": geo.distance(x, y)} for nome, geo in geometrias.items()]
    df_dist = pd.DataFrame(rows)
    fig = px.bar(df_dist, x="geometria", y="distância", title=f"Distância entre {labels_map[sid_a]} e {labels_map[sid_b]}")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    st.caption(
        "Euclidiana/Mahalanobis medem distância física no espaço; Cosine ignora magnitude e compara só direção; "
        "Wasserstein/Information tratam o vetor como distribuição sobre os eixos (ver ressalva no README) — "
        "todas produzem números em escalas diferentes, não são diretamente comparáveis entre si."
    )

# =============================================================================
# TAB 2 — Geometrias de trajetória
# =============================================================================
with tab2:
    st.subheader("Comparar 2 trajetórias inteiras (não um único instante)")
    elegiveis = [sid for sid, traj in pipeline.cohort.trajectories.items() if len(traj) >= 2]
    if len(elegiveis) < 2:
        st.warning("Poucos pacientes com mais de 1 exame nesta coorte para comparar trajetórias.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            traj_a_id = st.selectbox(
                "Paciente A (trajetória)", elegiveis, format_func=lambda s: f"{labels_map.get(s, s)} ({len(pipeline.cohort.trajectories[s])} exames)"
            )
        with col2:
            traj_b_id = st.selectbox(
                "Paciente B (trajetória)", elegiveis, index=min(1, len(elegiveis) - 1),
                format_func=lambda s: f"{labels_map.get(s, s)} ({len(pipeline.cohort.trajectories[s])} exames)",
            )

        traj_a = pipeline.cohort.trajectories[traj_a_id]
        traj_b = pipeline.cohort.trajectories[traj_b_id]

        st.markdown("**DTW (Dynamic Time Warping)**")
        dtw = DTW(order=order)
        dtw_dist, path = dtw.align(traj_a, traj_b)
        st.metric("Distância DTW (normalizada)", f"{dtw_dist:.3f}")

        # Visualiza o alinhamento usando um indicador escalar simples (distância à baseline de cada trajetória)
        euclid = Euclidean()
        base_a, base_b = traj_a.at(0).as_vector(order), traj_b.at(0).as_vector(order)
        serie_a = [euclid.distance(traj_a.at(i).as_vector(order), base_a) for i in range(len(traj_a))]
        serie_b = [euclid.distance(traj_b.at(i).as_vector(order), base_b) for i in range(len(traj_b))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=serie_a, mode="lines+markers", name=f"A: {labels_map.get(traj_a_id, traj_a_id)}"))
        fig.add_trace(go.Scatter(y=serie_b, mode="lines+markers", name=f"B: {labels_map.get(traj_b_id, traj_b_id)}"))
        for i, j in path:
            fig.add_trace(go.Scatter(x=[i, j], y=[serie_a[i], serie_b[j]], mode="lines", line=dict(color="gray", width=0.5), showlegend=False, xaxis="x", yaxis="y"))
        st.caption("As linhas cinzas mostram quais exames o DTW considerou correspondentes entre as duas trajetórias.")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)

        st.divider()
        st.markdown("**Gromov-Wasserstein** (estrutura relacional interna — ignora direção/ordem)")
        st.caption(
            "⚠️ Muito mais lento que DTW (Sinkhorn iterativo) e, em testes com esta coorte, "
            "**não** mostrou relação com o fenótipo final do paciente (ver README) — é invariante a "
            "reflexão temporal (trajetórias crescente e decrescente com a mesma variabilidade interna "
            "parecem iguais para o GW). Use para comparar 'forma de variabilidade', não para prognóstico."
        )
        if st.button("Calcular Gromov-Wasserstein para este par (pode levar alguns segundos)"):
            with st.spinner("Rodando Sinkhorn..."):
                gw = GromovWasserstein(max_iter=150)
                try:
                    gw_dist = gw.distance(traj_a, traj_b)
                    st.metric("Distância Gromov-Wasserstein", f"{gw_dist:.4f}")
                except ValueError as e:
                    st.error(str(e))

# =============================================================================
# TAB 3 — Geometria aprendida
# =============================================================================
with tab3:
    st.subheader("Geometria aprendida (Neighbourhood Components Analysis)")
    st.caption(
        "Aprende uma transformação que separa melhor classes rotuladas. Usamos `classificar_apneia()` "
        "(rótulo independente, vem direto do IDO bruto — não da própria clusterização) para validar "
        "honestamente, evitando o problema de 'aprender a enxergar o que o K-Means já viu'."
    )

    if st.button("Ajustar geometria aprendida e comparar silhouette", type="primary"):
        with st.spinner("Ajustando NCA..."):
            labels = {}
            for sid in pipeline.space.ids():
                classe = classificar_apneia(pipeline.cohort.systems[sid].latest_values().get("ido"))
                if classe is not None:
                    labels[sid] = classe

            space_labeled = RepresentationSpace(domain_order=pipeline.space.order())
            for sid in labels:
                space_labeled.add(pipeline.space.get(sid))

            geo = LearnedGeometry()
            geo.fit(space_labeled, labels)

            matrix, ids_labeled = space_labeled.matrix()
            y = [labels[sid] for sid in ids_labeled]

            from sklearn.metrics import silhouette_score

            sil_original = silhouette_score(matrix, y)
            matrix_aprendida = np.array([geo.transform(row) for row in matrix])
            sil_aprendida = silhouette_score(matrix_aprendida, y)

        c1, c2 = st.columns(2)
        c1.metric("Silhouette — espaço original (Euclidiana)", f"{sil_original:.4f}")
        c2.metric("Silhouette — espaço aprendido (NCA)", f"{sil_aprendida:.4f}", f"{100*(sil_aprendida-sil_original)/abs(sil_original):+.0f}%")

        st.markdown("**Projeção 2D do espaço aprendido** (para visualização apenas — a distância de verdade usa todas as dimensões)")
        geo2d = LearnedGeometry(n_components=2)
        geo2d.fit(space_labeled, labels)
        proj = np.array([geo2d.transform(row) for row in matrix])
        plot_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "classe": y})
        fig = px.scatter(plot_df, x="x", y="y", color="classe", title="Pacientes no espaço aprendido (2D)")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)

# =============================================================================
# TAB 4 — Riemanniana + Geometria da Doença (dinâmica)
# =============================================================================
with tab4:
    st.subheader("Riemanniana: o espaço deixa de ser plano")
    st.caption(
        "Aproxima a distância geodésica ao longo da variedade real de dados (Isomap: geodésica por "
        "grafo k-NN), em vez de assumir que a linha reta entre dois pacientes é sempre a distância "
        "certa. Testado em dados sintéticos (espiral): pontos em voltas adjacentes (próximos em linha "
        "reta, longe ao longo da variedade) mostraram geodésica 63% maior — comportamento esperado."
    )

    k_neighbors = st.slider("Nº de vizinhos (k) do grafo", 4, 20, value=10)
    if st.button("Ajustar RiemannianGeometry sobre a coorte"):
        with st.spinner("Construindo grafo k-NN sobre a população..."):
            riem = RiemannianGeometry(k_neighbors=k_neighbors)
            riem.fit(pipeline.space)
        st.session_state["_riem_geo"] = riem

    if "_riem_geo" in st.session_state:
        riem = st.session_state["_riem_geo"]
        ids2 = pipeline.space.ids()
        colr1, colr2 = st.columns(2)
        with colr1:
            sid_ra = st.selectbox("Paciente A", ids2, format_func=lambda s: labels_map[s], key="riem_a")
        with colr2:
            sid_rb = st.selectbox("Paciente B", ids2, index=min(1, len(ids2) - 1), format_func=lambda s: labels_map[s], key="riem_b")

        xa = pipeline.space.get(sid_ra).as_vector(order)
        xb = pipeline.space.get(sid_rb).as_vector(order)
        d_euclid = Euclidean().distance(xa, xb)
        try:
            d_riem = riem.distance(xa, xb)
            c1, c2, c3 = st.columns(3)
            c1.metric("Euclidiana", f"{d_euclid:.3f}")
            c2.metric("Geodésica (Riemanniana)", f"{d_riem:.3f}")
            c3.metric("Razão geodésica/euclidiana", f"{d_riem/d_euclid:.2f}x" if d_euclid > 0 else "n/a")
        except ValueError as e:
            st.error(str(e))

    st.divider()

    st.subheader("Geometria da Doença: cada fenótipo aprende sua própria métrica")
    st.caption(
        "PhenotypeConditionedGeometry — d(x, y, t): a métrica muda conforme o ESTÁGIO de referência "
        "(qual fenótipo). Usa encolhimento Ledoit-Wolf (não regularização ingênua) — necessário porque "
        "alguns fenótipos têm menos pacientes que dimensões do espaço, o que tornaria a covariância crua "
        "instável (testamos: sem encolhimento, um fenótipo pequeno deu distância 90x maior que os outros "
        "— artefato numérico, corrigido)."
    )

    if st.button("Ajustar PhenotypeConditionedGeometry sobre a coorte"):
        with st.spinner("Ajustando covariância local por fenótipo (Ledoit-Wolf)..."):
            dyn_geo = PhenotypeConditionedGeometry()
            dyn_geo.fit(pipeline.space, pipeline.phenotypes, order=order)
        st.session_state["_dyn_geo"] = dyn_geo

    if "_dyn_geo" in st.session_state:
        dyn_geo = st.session_state["_dyn_geo"]

        st.markdown("**Diagnóstico por fenótipo** (shrinkage alto = fenótipo pequeno, confia mais na covariância populacional):")
        resumo_rows = [
            {"fenótipo": nome, **valores} for nome, valores in dyn_geo.covariance_summary().items()
        ]
        st.dataframe(pd.DataFrame(resumo_rows), use_container_width=True, hide_index=True)

        st.markdown("**A mesma distância, sob diferentes estágios de referência:**")
        ids3 = pipeline.space.ids()
        colr3, colr4 = st.columns(2)
        with colr3:
            sid_da = st.selectbox("Paciente A", ids3, format_func=lambda s: labels_map[s], key="dyn_a")
        with colr4:
            sid_db = st.selectbox("Paciente B", ids3, index=min(1, len(ids3) - 1), format_func=lambda s: labels_map[s], key="dyn_b")

        xa2 = pipeline.space.get(sid_da).as_vector(order)
        xb2 = pipeline.space.get(sid_db).as_vector(order)

        rows_dyn = [{"referência (t)": "Euclidiana (fixa)", "distância": Euclidean().distance(xa2, xb2)}]
        for ph in pipeline.phenotypes:
            rows_dyn.append({"referência (t)": ph.name, "distância": dyn_geo.distance(xa2, xb2, ph.name)})
        fig = px.bar(pd.DataFrame(rows_dyn), x="referência (t)", y="distância", title=f"d({labels_map[sid_da]}, {labels_map[sid_db]}, t)")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
