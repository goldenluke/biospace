import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.causal import ObservationalEffectEstimator, check_baseline_balance

st.set_page_config(page_title="Inferência Causal - Diabetes", page_icon="🔀", layout="wide")

pipeline = require_pipeline()
order = pipeline.representation.domain_names()

st.title("🔀 Inferência Causal")
st.error(
    "⚠️ Leia antes de tudo: nada aqui é uma inferência causal identificada no sentido formal de "
    "Pearl. Sem grafo causal validado, sem ajuste por confundidores desconhecidos, sem "
    "randomização. É uma associação observacional (sujeita a confundimento por indicação)."
)

opcoes_tratamento = {"Metformina": "metformina", "Insulina": "insulina"}
tratamento_label = st.selectbox("Tratamento", list(opcoes_tratamento.keys()))
tratamento_feature = opcoes_tratamento[tratamento_label]

if st.button("Rodar balanceamento + efeito observacional", type="primary"):
    with st.spinner("Comparando linha de base entre quem inicia e quem não inicia o tratamento..."):
        try:
            balance = check_baseline_balance(pipeline.cohort, "treatment", tratamento_feature, order=order)
            estimator = ObservationalEffectEstimator("treatment", tratamento_feature, order=order)
            effect_report = estimator.estimate(pipeline.cohort)
        except ValueError as e:
            st.error(f"{e}\n\nTente gerar mais pacientes na tela inicial, ou escolher o outro tratamento.")
            st.stop()
    st.session_state["_balance"] = balance
    st.session_state["_effect"] = effect_report

if "_balance" not in st.session_state:
    st.info("Configure e clique em 'Rodar'.")
    st.stop()

balance = st.session_state["_balance"]
effect_report = st.session_state["_effect"]

c1, c2, c3 = st.columns(3)
c1.metric("Grupo tratado", balance.n_treated)
c2.metric("Grupo não-tratado", balance.n_untreated)
c3.metric("Features desequilibradas", f"{balance.n_imbalanced}/{len(balance.feature_names)}")

if not balance.is_balanced:
    st.warning("Grupos desequilibrados na linha de base — evidência de confundimento por indicação.")

smd_df = pd.DataFrame(balance.most_imbalanced(15), columns=["feature", "smd"])
fig = px.bar(smd_df, x="feature", y="smd", title="Maiores desequilíbrios de linha de base (SMD)")
fig.add_hline(y=0.1, line_dash="dash", line_color="red")
fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
fig.update_xaxes(tickangle=-45)
st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.divider()
st.subheader(f"Efeito observacional estimado ({effect_report.n_transitions} transições reais 0->1)")
top_df = pd.DataFrame(effect_report.top_changes(15), columns=["feature", "delta_medio"])
top_df["delta_std"] = top_df["feature"].map(effect_report.delta_std)
fig2 = px.bar(top_df, x="feature", y="delta_medio", error_y="delta_std", title="Maiores mudanças médias associadas ao início do tratamento")
fig2.update_xaxes(tickangle=-45)
st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
