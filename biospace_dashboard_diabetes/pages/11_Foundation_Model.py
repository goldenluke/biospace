import components._bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.foundation import MaskedFeaturePredictor

st.set_page_config(page_title="Foundation Model - Diabetes", page_icon="🧩", layout="wide")

pipeline = require_pipeline()
order = pipeline.representation.domain_names()

st.title("🧩 Foundation Model — Protótipo de Arquitetura (Fase 10)")
st.error(
    "⚠️ **Leia antes de tudo**: isto é uma PROVA DE CONCEITO ARQUITETURAL, treinada em centenas de "
    "pacientes — não um foundation model de verdade, que exigiria 'milhões de pacientes'. A "
    "diferença é de ORDEM DE GRANDEZA, não de grau."
)
st.caption(
    "Masked Feature Prediction — o mesmo padrão do BERT, mas mascarando Features fisiológicas em "
    "vez de palavras."
)

mask_fraction = st.slider("Fração mascarada por cópia de treino", 0.05, 0.4, value=0.15, step=0.05)
hidden_dim = st.slider("Dimensão da camada oculta", 8, 64, value=32, step=8)

if st.button("Treinar Masked Feature Predictor", type="primary"):
    space = pipeline.cohort.snapshot()
    with st.spinner("Treinando..."):
        modelo = MaskedFeaturePredictor(hidden_dim=hidden_dim, mask_fraction=mask_fraction, n_masks_per_patient=20, max_iter=2000, random_state=0)
        modelo.fit(space)
        resultado = modelo.masked_reconstruction_error(space, mask_fraction=mask_fraction, seed=1)
    st.session_state["_masked_model"] = modelo
    st.session_state["_masked_resultado"] = resultado

if "_masked_resultado" in st.session_state:
    modelo = st.session_state["_masked_model"]
    resultado = st.session_state["_masked_resultado"]

    space = pipeline.cohort.snapshot()
    matrix, ids = space.matrix()

    linhas = []
    for nome, mse in resultado["mse_por_feature"].items():
        idx = modelo.feature_names_.index(nome)
        var = float(np.var(matrix[:, idx]))
        linhas.append({"feature": nome, "mse_reconstrucao": mse, "variancia_baseline": var, "razao": mse / var if var > 1e-9 else float("nan")})
    df_resultado = pd.DataFrame(linhas).sort_values("razao")

    st.metric("MSE global", f"{resultado['mse_global']:.3f}")
    st.markdown("**Melhor reconstruídas:**")
    st.dataframe(df_resultado.head(10), use_container_width=True, hide_index=True)
    st.markdown("**Pior reconstruídas:**")
    st.dataframe(df_resultado.tail(10), use_container_width=True, hide_index=True)

    fig = px.bar(df_resultado, x="feature", y="razao", title="Razão MSE/variância por Feature")
    fig.add_hline(y=1.0, line_dash="dash", line_color="red")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

    st.divider()
    st.subheader("Testar num paciente específico")
    elegiveis = list(pipeline.cohort.trajectories.keys())
    labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in elegiveis}
    paciente_escolhido = st.selectbox("Paciente", elegiveis, format_func=lambda s: labels_map[s])
    features_para_mascarar = st.multiselect("Features para mascarar", modelo.feature_names_, default=modelo.feature_names_[:3])

    if features_para_mascarar:
        x = pipeline.cohort.trajectories[paciente_escolhido].latest().as_vector(order)
        indices = [modelo.feature_names_.index(nome) for nome in features_para_mascarar]
        reconstruido = modelo.predict_masked(x, masked_indices=indices)

        tabela = pd.DataFrame([
            {"feature": nome, "valor_real": x[idx], "valor_previsto": reconstruido[idx], "erro_absoluto": abs(x[idx] - reconstruido[idx])}
            for nome, idx in zip(features_para_mascarar, indices)
        ])
        st.dataframe(tabela, use_container_width=True, hide_index=True)
