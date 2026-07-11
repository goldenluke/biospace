"""
App.py
======

Página inicial do dashboard BioSpace Diabetes. Só dados SINTÉTICOS —
não existe planilha real de diabetes neste projeto (diferente do
dashboard de sleep). O objetivo deste dashboard é demonstrar que o
mesmo ferramental (geometria, fenotipagem, dinâmica, causal) funciona
sobre um SEGUNDO plugin de doença, sem nenhuma alteração no núcleo.
"""

import components._bootstrap  # noqa: F401

import numpy as np
import streamlit as st

from components.pipeline import run_pipeline
from components.state import clear_pipeline, get_pipeline, get_source_label, set_pipeline
from biospace.plugins.diabetes import generate_synthetic_dataframe

st.set_page_config(page_title="BioSpace - Diabetes", page_icon="🩸", layout="wide")

st.title("🩸 BioSpace — Diabetes Tipo 2 (dados sintéticos)")
st.caption(
    "Segundo plugin de doença construído sobre o mesmo núcleo `biospace` do dashboard de SAOS — "
    "prova de que a arquitetura generaliza, não uma ferramenta clínica. Nenhum dado de paciente "
    "real: a coorte inteira é gerada localmente, com progressão longitudinal realista."
)
st.info(
    "🧪 Este dashboard só tem dados sintéticos — não existe planilha real de diabetes neste "
    "projeto. Compare com o dashboard de SAOS (que aceita upload de dados reais) para ver a "
    "diferença de propósito: lá é uma ferramenta validada; aqui é uma prova de arquitetura."
)

st.divider()

st.subheader("Gerar coorte sintética")
st.write(
    "3 grupos de severidade (controlado/moderado/descompensado), múltiplos exames por paciente, "
    "adoção de metformina/insulina correlacionada com severidade, e declínio renal correlacionado "
    "com exposição glicêmica crônica acumulada — não apenas a severidade do instante."
)
n_per_group = st.slider("Pacientes por grupo de severidade", 15, 150, 30, step=15)
if st.button("Gerar dados sintéticos", type="primary"):
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
    st.info("Nenhum dado carregado ainda. Gere uma coorte sintética acima para começar.")
else:
    col_badge, col_clear = st.columns([4, 1])
    with col_badge:
        st.subheader(f"Resumo da carga atual ({get_source_label()})")
    with col_clear:
        if st.button("🗑️ Limpar dados"):
            clear_pipeline()
            st.rerun()

    n_exams_per_patient = [len(t) for t in pipeline.cohort.trajectories.values()]
    n_multi_exam = sum(1 for n in n_exams_per_patient if n >= 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes carregados", len(pipeline.cohort))
    c2.metric("Dimensão da representação", pipeline.space.matrix()[0].shape[1])
    c3.metric("Pacientes com ≥2 exames", f"{n_multi_exam} ({100*n_multi_exam/len(pipeline.cohort):.0f}%)")
    c4.metric("Exames por paciente (mediana)", f"{np.median(n_exams_per_patient):.0f}")

    st.write("**Domínios da representação:**", ", ".join(pipeline.representation.domain_names()))

    st.write("**Distribuição de controle glicêmico (classificação clínica padrão, só para exibição):**")
    st.write(pipeline.display_df["classe_controle"].value_counts())

    st.divider()
    st.subheader("Páginas disponíveis")
    st.markdown(
        """
- **Visão Geral** — distribuições transversais (glicemia, HbA1c, IMC)
- **Controle Glicêmico** — glicemia/HbA1c por classe de controle
- **Função Renal** — eGFR/creatinina e o mecanismo de declínio por exposição
- **Perfis** — fenotipagem (KMeans genérico)
- **Domínio Latente** — proxy de resistência à insulina
- **Sistemas Dinâmicos** — EvolutionOperator/StabilityOperator + simulação em conjunto (Fase 9)
- **Curvatura** — temporal, densidade populacional, e estrutural via grafo (Fase 8)
- **GNN** — Graph Convolutional Network, classificação semi-supervisionada de fenótipo
- **Foundation Model** — masked feature prediction (protótipo de arquitetura, Fase 10)
- **Inferência Causal** — balanceamento + efeito observacional (metformina/insulina)
- **Paciente** — busca individual e trajetória
        """
    )
