import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.causal import (
    DigitalTwin,
    ObservationalEffectEstimator,
    Scenario,
    check_baseline_balance,
    estimate_matched_effect,
    estimate_propensity,
    match_on_propensity,
)
from biospace.dynamics import MeanRevertingEvolutionOperator
from biospace.geometry import Euclidean
from biospace.intervention import FeatureShiftIntervention

st.set_page_config(page_title="Inferência Causal · BioSpace", page_icon="🔀", layout="wide")

pipeline = require_pipeline()

st.title("🔀 Inferência Causal")
st.error(
    "⚠️ **Leia antes de tudo**: `do()` aqui aplica uma TRANSFORMAÇÃO no espaço de representação — "
    "NÃO é uma inferência causal identificada no sentido formal de Pearl. Não há grafo causal "
    "validado, nem ajuste por confundidores desconhecidos, nem randomização. É uma associação "
    "observacional (sujeita a confundimento por indicação) ou uma simulação hipotética — nunca "
    "um efeito causal comprovado. O nome é usado pela clareza do paralelo conceitual."
)

order = pipeline.representation.domain_names()
treatment_domain = "treatment"
opcoes_tratamento = {"AAM (aparelho de avanço mandibular)": "aam", "CPAP": "cpap"}

tab1, tab_prop, tab2 = st.tabs(["Balanceamento + Efeito Observacional", "Pareamento por Propensão", "Gêmeo Digital + Cenários"])

# =============================================================================
# TAB 1 — Balance + Effect
# =============================================================================
with tab1:
    st.subheader("Passo 1 (obrigatório): balanceamento de linha de base")
    st.caption(
        "Antes de estimar qualquer 'efeito', verifica se quem inicia tratamento já era diferente "
        "de quem não inicia, ANTES do tratamento — se sim, é evidência de confundimento por indicação."
    )

    tratamento_label = st.selectbox("Tratamento", list(opcoes_tratamento.keys()))
    tratamento_feature = opcoes_tratamento[tratamento_label]

    if st.button("Rodar balanceamento + efeito observacional", type="primary"):
        with st.spinner("Comparando linha de base entre quem inicia e quem não inicia o tratamento..."):
            try:
                balance = check_baseline_balance(
                    pipeline.cohort, treatment_domain=treatment_domain, treatment_feature=tratamento_feature, order=order
                )
                estimator = ObservationalEffectEstimator(
                    treatment_domain=treatment_domain, treatment_feature=tratamento_feature, order=order
                )
                effect_report = estimator.estimate(pipeline.cohort)
            except ValueError as e:
                st.error(f"{e}\n\nTente aumentar o nº de pacientes na tela inicial, ou escolher o outro tratamento (AAM/CPAP) — com poucos pacientes, um grupo específico pode ficar pequeno demais por acaso.")
                st.stop()
        st.session_state["_balance"] = balance
        st.session_state["_effect"] = effect_report
        st.session_state["_tratamento_label"] = tratamento_label

    if "_balance" in st.session_state:
        balance = st.session_state["_balance"]
        effect_report = st.session_state["_effect"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Grupo tratado (n)", balance.n_treated)
        c2.metric("Grupo não-tratado (n)", balance.n_untreated)
        c3.metric("Features desequilibradas", f"{balance.n_imbalanced}/{len(balance.feature_names)}")

        if not balance.is_balanced:
            st.warning(
                "**Grupos desequilibrados na linha de base** — evidência de confundimento por "
                "indicação. Qualquer diferença de desfecho abaixo pode refletir essas diferenças "
                "pré-existentes, não um efeito do tratamento."
            )

        smd_df = pd.DataFrame(balance.most_imbalanced(15), columns=["feature", "smd"])
        fig = px.bar(smd_df, x="feature", y="smd", title="Maiores desequilíbrios de linha de base (SMD)")
        fig.add_hline(y=0.1, line_dash="dash", line_color="red")
        fig.add_hline(y=-0.1, line_dash="dash", line_color="red")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)

        st.divider()
        st.subheader(f"Efeito observacional estimado ({effect_report.n_transitions} transições reais 0→1)")
        top_df = pd.DataFrame(effect_report.top_changes(15), columns=["feature", "delta_medio"])
        top_df["delta_std"] = top_df["feature"].map(effect_report.delta_std)
        fig2 = px.bar(top_df, x="feature", y="delta_medio", error_y="delta_std", title="Maiores mudanças médias associadas ao início do tratamento")
        fig2.update_xaxes(tickangle=-45)
        st.plotly_chart(apply_default_layout(fig2), use_container_width=True)
        st.caption("Barras de erro = desvio padrão entre pacientes, não incerteza da estimativa do efeito em si.")

# =============================================================================
# TAB PROPENSITY — Pareamento por escore de propensão
# =============================================================================
with tab_prop:
    st.subheader("Pareamento por escore de propensão (Rosenbaum & Rubin, 1983)")
    st.caption(
        "Tenta REDUZIR (nunca eliminar) o confundimento por indicação: ajusta P(tratamento | linha de "
        "base) via regressão logística com regularização L2 forte, e pareia cada tratado ao "
        "não-tratado mais parecido nessa escala, dentro de um raio (caliper). Só ajusta para "
        "confundidores OBSERVADOS — adesão, motivação, acesso a cuidado continuam invisíveis."
    )

    tratamento_label_2 = st.selectbox("Tratamento", list(opcoes_tratamento.keys()), key="prop_tratamento")
    tratamento_feature_2 = opcoes_tratamento[tratamento_label_2]
    caliper = st.slider("Caliper (escala logit — menor = pareamento mais rigoroso, menos pacientes pareados)", 0.05, 0.50, value=0.25, step=0.05)

    if st.button("Ajustar escore de propensão + parear", type="primary"):
        with st.spinner("Ajustando regressão logística regularizada e pareando..."):
            try:
                prop_model = estimate_propensity(pipeline.cohort, treatment_domain, tratamento_feature_2, order=order)
                match_result = match_on_propensity(pipeline.cohort, treatment_domain, tratamento_feature_2, order=order, caliper=caliper)
            except ValueError as e:
                st.error(f"{e}\n\nTente aumentar o nº de pacientes na tela inicial, ou escolher o outro tratamento (AAM/CPAP) — com poucos pacientes, um grupo específico pode ficar pequeno demais por acaso.")
                st.stop()
        st.session_state["_prop_model"] = prop_model
        st.session_state["_match_result"] = match_result

    if "_match_result" in st.session_state:
        prop_model = st.session_state["_prop_model"]
        match_result = st.session_state["_match_result"]

        c1, c2, c3 = st.columns(3)
        c1.metric("AUC de validação cruzada", f"{prop_model.cv_auc:.3f}" if prop_model.cv_auc == prop_model.cv_auc else "n/a")
        c2.metric("Pacientes pareados", f"{match_result.n_matched}/{match_result.n_treated_total} ({100*match_result.match_rate:.0f}%)")
        c3.metric("|SMD| médio: antes → depois", f"{match_result.mean_absolute_smd_before:.3f} → {match_result.mean_absolute_smd_after:.3f}")

        if prop_model.cv_auc == prop_model.cv_auc and prop_model.cv_auc > 0.95:
            st.warning("AUC muito alto — os grupos são tão distintos na linha de base que o pareamento pode ter dificuldade em achar bons pares.")

        if match_result.match_rate < 0.3:
            st.warning(
                f"⚠️ Taxa de pareamento baixa ({100*match_result.match_rate:.0f}%) — a maioria dos pacientes tratados "
                "não teve par comparável dentro do caliper. Isso é honesto (não força maus pareamentos), mas "
                "significa que os pacientes pareados podem não representar bem TODOS os que recebem o tratamento."
            )

        st.markdown(f"**{match_result.summary().splitlines()[-1]}**")

        try:
            matched_effect_report = estimate_matched_effect(pipeline.cohort, match_result, order=order)
            n_confiavel = sum(
                1 for name in matched_effect_report.effect
                if matched_effect_report.effect_std.get(name, 0) > 0
                and abs(matched_effect_report.effect[name]) / matched_effect_report.effect_std[name] > 1.0
            )
            st.caption(
                f"Efeito pareado calculado sobre {matched_effect_report.n_pairs_used} pares utilizáveis "
                f"({matched_effect_report.n_pairs_dropped_single_exam_control} descartados por o controle ter só 1 exame). "
                f"Apenas {n_confiavel} Feature(s) têm |efeito| > 1 desvio-padrão — trate o restante como ruído, não sinal."
            )
            top_matched_df = pd.DataFrame(matched_effect_report.top_effects(12), columns=["feature", "efeito_pareado"])
            top_matched_df["desvio_padrao"] = top_matched_df["feature"].map(matched_effect_report.effect_std)
            fig_matched = px.bar(
                top_matched_df, x="feature", y="efeito_pareado", error_y="desvio_padrao",
                title="Efeito pareado (diferença-em-diferenças) — barras de erro grandes = não confie no valor",
            )
            fig_matched.update_xaxes(tickangle=-45)
            st.plotly_chart(apply_default_layout(fig_matched), use_container_width=True)
        except ValueError as e:
            st.info(f"Não foi possível calcular o efeito pareado: {e}")

# =============================================================================
# TAB 2 — Digital Twin + Scenario
# =============================================================================
with tab2:
    st.subheader("Gêmeo Digital: clonar, aplicar do(), simular")
    st.caption("`digital_twin = patient.clone(); digital_twin.do(intervencao); digital_twin.simulate()`")

    if "_evo" not in st.session_state:
        st.info("Ajuste a dinâmica primeiro na página 'Sistemas Dinâmicos' (necessário para simular após do()).")
        st.stop()
    evo = st.session_state["_evo"]

    elegiveis = list(pipeline.cohort.trajectories.keys())
    labels_map = {sid: pipeline.cohort.systems[sid].metadata.get("paciente_original", sid) for sid in elegiveis}
    paciente_escolhido = st.selectbox("Paciente", elegiveis, format_func=lambda s: labels_map[s], key="twin_patient")
    traj = pipeline.cohort.trajectories[paciente_escolhido]

    st.markdown("**Definir cenários (braços de comparação)**")
    horizon_days = st.slider("Horizonte (dias)", 30, 720, value=180, step=30, key="scenario_horizon")
    step_days = st.slider("Passo (dias)", 15, 180, value=60, step=15, key="scenario_step")

    col1, col2 = st.columns(2)
    with col1:
        incluir_aam_obs = st.checkbox("Braço: AAM (efeito observacional, se já calculado na aba anterior)", value="_effect" in st.session_state)
    with col2:
        st.markdown("Braço hipotético (perda de peso):")
        shift_imc = st.slider("Δ IMC (z-score)", -3.0, 0.0, value=-1.0, step=0.1)

    if st.button("Rodar cenários", type="primary"):
        cenario = Scenario("Comparação de braços")
        if incluir_aam_obs and "_effect" in st.session_state:
            effect_report = st.session_state["_effect"]
            top_deltas = dict(effect_report.top_changes(8))
            top_deltas.pop("treatment.aam", None)
            shifts_aam = {name.split(".", 1)[1]: delta for name, delta in top_deltas.items() if "." in name}
            if shifts_aam:
                cenario.add_arm(f"AAM (observacional, {st.session_state.get('_tratamento_label', '')})", FeatureShiftIntervention(shifts=shifts_aam))
        cenario.add_arm("Perda de peso (hipotética)", FeatureShiftIntervention(shifts={"imc": shift_imc}))

        with st.spinner("Simulando braços..."):
            try:
                resultados = cenario.run(traj, evo, horizon_days=horizon_days, step_days=step_days, order=order)
            except KeyError as e:
                st.error(f"Um dos braços usou um nome de Feature que não existe no vetor: {e}")
                st.stop()
        st.session_state["_scenario_results"] = resultados

    if "_scenario_results" in st.session_state:
        resultados = st.session_state["_scenario_results"]
        euclid = Euclidean()

        fig3 = go.Figure()
        for label, r in resultados.items():
            estado_base = r.path[0][1]
            dias = [t for t, _ in r.path]
            distancias = [euclid.distance(estado_base, x) for _, x in r.path]
            fig3.add_trace(go.Scatter(x=dias, y=distancias, mode="lines+markers", name=label))
        fig3.update_layout(
            title=f"Trajetórias simuladas por braço — {labels_map[paciente_escolhido]}",
            xaxis_title="Dias", yaxis_title="Distância ao estado inicial (Euclidiana)",
        )
        st.plotly_chart(apply_default_layout(fig3), use_container_width=True)

        distancias_controle = Scenario.compare_to_control(resultados, euclid)
        st.markdown("**Distância do estado final de cada braço ao braço de controle:**")
        st.dataframe(
            pd.DataFrame([{"braço": k, "distância_ao_controle": v} for k, v in distancias_controle.items()]),
            use_container_width=True, hide_index=True,
        )

        with st.expander("Histórico de cada braço (auditoria)"):
            for label, r in resultados.items():
                st.write(f"**{label}**: {r.history}")
