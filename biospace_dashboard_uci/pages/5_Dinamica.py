import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import Cohort
from biospace.dynamics import MeanRevertingEvolutionOperator, StabilityOperator, check_feature_stability_robustness

st.set_page_config(page_title="Dinâmica - UCI", page_icon="📉", layout="wide")

pipeline = require_pipeline()

st.title("📉 Dinâmica de Reversão à Média")
st.caption(
    "Primeira vez que o módulo de dinâmica (`MeanRevertingEvolutionOperator`, o mesmo de SAOS) roda "
    "sobre trajetória REAL fora de sleep — não sintética, não transversal."
)

df = pipeline.display_df
multi = df[df["n_encontros"] >= 2]
st.metric("Pacientes usados no ajuste (≥2 encontros)", len(multi))

if len(multi) < 10:
    st.warning("Poucos pacientes multi-encontro nesta amostra — use a base completa na página inicial para resultado robusto.")

if st.button("Ajustar dinâmica e checar robustez", type="primary"):
    order = pipeline.representation.domain_names()
    cohort_multi = Cohort()
    for sid in multi["id"]:
        if len(pipeline.cohort.trajectories[sid]) >= 2:
            cohort_multi.systems[sid] = pipeline.cohort.systems[sid]
            cohort_multi.trajectories[sid] = pipeline.cohort.trajectories[sid]

    if len(cohort_multi.trajectories) < 5:
        st.error("Poucos pacientes com trajetória para ajustar a dinâmica nesta amostra.")
        st.stop()

    with st.spinner("Ajustando processo de Ornstein-Uhlenbeck por Feature..."):
        evo = MeanRevertingEvolutionOperator(order=order)
        evo.fit(cohort_multi)
        relatorio = StabilityOperator(evolution_operator=evo, n_worst=13).analyze(cohort_multi)

    st.session_state["_uci_dynamics"] = (evo, relatorio, cohort_multi, order)

if "_uci_dynamics" in st.session_state:
    evo, relatorio, cohort_multi, order = st.session_state["_uci_dynamics"]

    c1, c2 = st.columns(2)
    c1.metric("Features estáveis", f"{relatorio.n_stable}/{relatorio.n_features}")
    c2.metric("Globalmente estável?", "Sim" if relatorio.is_globally_stable else "Não")

    linhas = [{"feature": name, "phi_dia": fd.phi_per_day, "n_pares": fd.n_pairs, "estavel": fd.is_stable} for name, fd in relatorio.dynamics.items()]
    dyn_df = pd.DataFrame(linhas).sort_values("phi_dia", ascending=False)
    fig = px.bar(dyn_df, x="feature", y="phi_dia", color="estavel", color_discrete_map={True: "#00C853", False: "#FF7043"}, title="φ por Feature")
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
    st.dataframe(dyn_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Diagnóstico de robustez na Feature mais perto da instabilidade")
    st.caption(
        "O mesmo diagnóstico (`check_feature_stability_robustness`) que expôs um artefato de outlier "
        "em SAOS — aqui usado pra checar se a Feature mais instável é um artefato ou um sinal real."
    )
    feature_mais_instavel = dyn_df.iloc[0]["feature"]
    st.write(f"Feature mais perto da instabilidade nesta execução: **{feature_mais_instavel}**")

    if st.button("Rodar diagnóstico de robustez (remove 1 paciente por vez)"):
        with st.spinner("Reajustando removendo cada paciente extremo, um de cada vez..."):
            resultado_robustez = check_feature_stability_robustness(cohort_multi, feature_mais_instavel, order=order, max_patients_tested=30)
        st.text(resultado_robustez.summary())
        if resultado_robustez.conclusion_is_robust:
            st.success("Conclusão robusta — não é artefato de um paciente outlier.")
        else:
            st.warning("Conclusão NÃO robusta — depende de um paciente específico. Investigar (mesmo padrão do achado em SAOS).")
