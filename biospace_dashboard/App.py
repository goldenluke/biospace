"""
App.py
======

Página inicial do dashboard BioSpace SAOS. Carrega uma planilha real
(.xlsx) ou gera dados sintéticos de demonstração (agora LONGITUDINAIS —
múltiplos exames por paciente, com progressão e adoção de tratamento
realistas), roda o pipeline biospace inteiro (Cohort -> Representation
-> RepresentationSpace -> ClinicalKMeansPhenotyper) e guarda o resultado
em st.session_state para as demais páginas em pages/.
"""

import components._bootstrap  # noqa: F401  (garante que `biospace` seja importável)

import numpy as np
import streamlit as st

from components.pipeline import run_pipeline
from components.state import clear_pipeline, get_pipeline, get_source_kind, get_source_label, set_pipeline
from components.synthetic import generate_synthetic_dataframe

st.set_page_config(page_title="BioSpace · SAOS", page_icon="😴", layout="wide")

st.title("😴 BioSpace — Medicina Computacional de Precisão para SAOS")
st.caption(
    "Dashboard construído sobre o meta-modelo `biospace`: os algoritmos nunca tocam a "
    "planilha diretamente — eles operam sobre a `Representation` e o `RepresentationSpace` "
    "computados a partir dela."
)

st.divider()

col_upload, col_demo = st.columns(2)

with col_upload:
    st.subheader("1. Envie uma planilha real")
    uploaded = st.file_uploader("Arquivo Excel (.xlsx), mesmo formato de 'Exames realizados'", type=["xlsx"])
    if uploaded is not None:
        cache_key = f"file::{uploaded.name}::{uploaded.size}"
        if get_source_label() != cache_key:
            with st.spinner("Carregando planilha, construindo representação e estimando fenótipos..."):
                import pandas as pd

                raw_df = pd.read_excel(uploaded, header=1)
                pipeline = run_pipeline(raw_df)
            set_pipeline(pipeline, cache_key)
            st.success(f"Planilha carregada: {len(pipeline.cohort)} pacientes.")
            st.rerun()

with col_demo:
    st.subheader("2. Ou gere dados de demonstração")
    st.write(
        "Coorte sintética **longitudinal**: múltiplos exames por paciente ao longo de meses/anos, "
        "com progressão de severidade e adoção de tratamento (AAM/CPAP) correlacionada com "
        "severidade — o mesmo padrão de confundimento por indicação da planilha real."
    )
    n_per_group = st.slider("Pacientes por grupo de severidade (leve/moderada/grave)", 15, 200, 30, step=15)
    if st.button("Gerar dados sintéticos"):
        cache_key = f"synthetic::{n_per_group}"
        with st.spinner("Gerando coorte sintética longitudinal e rodando o pipeline biospace..."):
            raw_df = generate_synthetic_dataframe(n_per_group=n_per_group)
            pipeline = run_pipeline(raw_df)
        set_pipeline(pipeline, cache_key)
        st.success(f"Dados sintéticos gerados: {len(pipeline.cohort)} pacientes.")
        st.rerun()

st.divider()

pipeline = get_pipeline()
if pipeline is None:
    st.info("Nenhum dado carregado ainda. Use uma das opções acima para começar.")
else:
    source_kind = get_source_kind()
    badge = "🧪 Dados **sintéticos** (demonstração)" if source_kind == "synthetic" else "📄 Dados **reais** (planilha enviada)"
    col_badge, col_clear = st.columns([4, 1])
    with col_badge:
        st.subheader(f"Resumo da carga atual — {badge}")
    with col_clear:
        if st.button("🗑️ Limpar dados", help="Remove os dados carregados e qualquer análise já rodada nas outras páginas."):
            clear_pipeline()
            st.rerun()

    report = getattr(pipeline.cohort, "loader_report", {})
    n_exams_per_patient = [len(t) for t in pipeline.cohort.trajectories.values()]
    n_multi_exam = sum(1 for n in n_exams_per_patient if n >= 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes carregados", len(pipeline.cohort))
    c2.metric("Linhas descartadas", report.get("n_rows_discarded", 0))
    c3.metric("Dimensão da representação", pipeline.space.matrix()[0].shape[1])
    c4.metric("K escolhido (fenotipagem)", pipeline.phenotyper.best_k)

    c5, c6, c7 = st.columns(3)
    c5.metric("Pacientes com ≥2 exames", f"{n_multi_exam} ({100*n_multi_exam/len(pipeline.cohort):.0f}%)")
    c6.metric("Exames por paciente (mediana)", f"{np.median(n_exams_per_patient):.0f}")
    c7.metric("Exames por paciente (máximo)", int(np.max(n_exams_per_patient)))
    if n_multi_exam < 10:
        st.warning(
            "Poucos pacientes com múltiplos exames — páginas longitudinais (Sobrevivência, Early "
            "Warning, Sistemas Dinâmicos, Inferência Causal, DTW/Gromov-Wasserstein) terão pouco ou "
            "nada para mostrar. Gere mais pacientes de demonstração, ou envie uma planilha real maior."
        )

    st.write("**Domínios da representação:**", ", ".join(pipeline.representation.domain_names()))

    st.write("**Fenótipos estimados:**")
    for ph in pipeline.phenotypes:
        n_members = int((pipeline.display_df["fenotipo"] == ph.name).sum())
        st.write(f"- {ph.name} ({ph.interpretation}): n={n_members}")

    st.divider()
    st.subheader("Páginas disponíveis")
    st.caption("Use o menu à esquerda para navegar — agrupadas aqui por tipo de análise, para orientação.")

    grupos = {
        "📊 Transversal (um instante por paciente)": [
            "Visão Geral", "Apneia", "Hipoxemia", "Frequência Cardíaca",
            "Sintomas e Comorbidades", "Perfis (fenotipagem)", "Qualidade de Dados", "Paciente (busca individual)",
        ],
        "📈 Longitudinal (trajetória ao longo do tempo)": [
            "Sobrevivência (Kaplan-Meier)", "Early Warning (critical slowing down)", "Sistemas Dinâmicos (previsão/simulação em conjunto)",
        ],
        "📐 Geometria e Estrutura": [
            "Geometrias (Euclidiana/DTW/Riemanniana/...)", "Curvatura (temporal/densidade/estrutural)", "Domínios Latentes (Inflamação/Fragilidade/Autonômico)", "Ontologia (dicionário de dados)",
        ],
        "🔀 Causal": [
            "Inferência Causal (balanceamento, pareamento, gêmeo digital)",
        ],
        "🧪 Fronteira (protótipos de arquitetura)": [
            "GNN (Graph Convolutional Network)", "Foundation Model (masked feature prediction)",
        ],
    }
    cols = st.columns(len(grupos))
    for col, (titulo, paginas) in zip(cols, grupos.items()):
        with col:
            st.markdown(f"**{titulo}**")
            for p in paginas:
                st.markdown(f"- {p}")
