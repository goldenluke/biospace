import components._bootstrap  # noqa: F401

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.geometry import Euclidean
from biospace.gnn import SimpleGCN, prepare_node_classification_data
from biospace.graph import build_cohort_similarity_graph

st.set_page_config(page_title="GNN - Diabetes", page_icon="🕸️", layout="wide")

pipeline = require_pipeline()
order = pipeline.representation.domain_names()

st.title("🕸️ GNN — Graph Convolutional Network")
st.caption(
    "SimpleGCN (Kipf & Welling, 2017) em NumPy puro, sobre o grafo de similaridade de pacientes. "
    "Prevê fenótipo de forma semi-supervisionada: alguns pacientes têm rótulo (treino), outros não, "
    "mas todos participam da propagação de mensagens."
)

k_vizinhos = st.slider("k (vizinhos no grafo de similaridade)", 3, 15, value=8)
fracao_rotulada = st.slider("Fração de pacientes ROTULADA (o resto é 'teste')", 0.02, 0.5, value=0.1, step=0.02)
seed = st.number_input("Seed", value=0, step=1)

st.caption(
    "Achado real deste projeto: com MUITOS rótulos, o grafo pode atrapalhar; com POUCOS rótulos, "
    "o grafo ajuda bastante. Experimente frações baixas (~5%) para ver a vantagem."
)

if st.button("Treinar GCN vs. sem grafo", type="primary"):
    space = pipeline.cohort.snapshot()
    if not pipeline.phenotypes:
        st.error("Nenhum fenótipo disponível nesta coorte.")
        st.stop()

    labels = {}
    for sid in space.ids():
        vec = space.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in pipeline.phenotypes if ph.contains(vec)), None)

    with st.spinner("Construindo grafo de similaridade..."):
        grafo = build_cohort_similarity_graph(space, Euclidean(), k=k_vizinhos, order=order)

    rng = np.random.default_rng(int(seed))
    todos_ids = list(labels.keys())
    rng.shuffle(todos_ids)
    n_treino = max(4, int(fracao_rotulada * len(todos_ids)))
    treino_ids = todos_ids[:n_treino]
    teste_ids = set(todos_ids[n_treino:])

    dados = prepare_node_classification_data(space, grafo, labels, labeled_ids=treino_ids, order=order)
    A, X, y, labeled_mask, node_ids = dados["A"], dados["X"], dados["y"], dados["labeled_mask"], dados["node_ids"]
    test_mask = np.array([nid in teste_ids for nid in node_ids])
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    with st.spinner("Treinando GCN (com grafo) e baseline (sem grafo)..."):
        gcn = SimpleGCN(hidden_dim=16, learning_rate=0.05, random_state=0)
        gcn.fit(A, X_std, y, labeled_mask, epochs=300)
        acc_com_grafo = float((gcn.predict(A, X_std)[test_mask] == y[test_mask]).mean())

        A_id = np.eye(A.shape[0])
        gcn2 = SimpleGCN(hidden_dim=16, learning_rate=0.05, random_state=0)
        gcn2.fit(A_id, X_std, y, labeled_mask, epochs=300)
        acc_sem_grafo = float((gcn2.predict(A_id, X_std)[test_mask] == y[test_mask]).mean())

    st.session_state["_gnn_resultado"] = {
        "n_treino": n_treino, "n_teste": int(test_mask.sum()),
        "acc_com_grafo": acc_com_grafo, "acc_sem_grafo": acc_sem_grafo,
        "loss_history": gcn.loss_history_,
    }

if "_gnn_resultado" in st.session_state:
    r = st.session_state["_gnn_resultado"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes treino / teste", f"{r['n_treino']} / {r['n_teste']}")
    c2.metric("Acurácia COM grafo", f"{r['acc_com_grafo']:.1%}")
    c3.metric("Acurácia SEM grafo", f"{r['acc_sem_grafo']:.1%}", delta=f"{100*(r['acc_com_grafo']-r['acc_sem_grafo']):+.1f}pp (vs. com grafo)", delta_color="inverse")

    diferenca = r["acc_com_grafo"] - r["acc_sem_grafo"]
    if diferenca > 0.02:
        st.success(f"O grafo AJUDOU (+{100*diferenca:.1f} pontos percentuais).")
    elif diferenca < -0.02:
        st.info(f"O grafo ATRAPALHOU levemente ({100*diferenca:.1f} pontos percentuais).")
    else:
        st.info("Diferença pequena entre as duas abordagens nesta configuração.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=r["loss_history"], mode="lines", name="Perda de treino (GCN)"))
    fig.update_layout(title="Convergência do treino", xaxis_title="Época", yaxis_title="Perda")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
