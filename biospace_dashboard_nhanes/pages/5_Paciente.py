import components._bootstrap  # noqa: F401

import streamlit as st

from components.state import require_pipeline
from biospace.plugins.metabolic import classify_diabetes_status, classify_metabolic_syndrome_risk

st.set_page_config(page_title="Paciente - NHANES", page_icon="🔍", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🔍 Busca de Participante")

labels_map = {row["id"]: row["paciente"] for _, row in df.iterrows()}
escolhido = st.selectbox("Participante (SEQN)", list(labels_map.keys()), format_func=lambda s: labels_map[s])

if escolhido:
    linha = df[df["id"] == escolhido].iloc[0]
    vetor = pipeline.space.get(escolhido)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Idade", f"{linha['idade']:.0f} anos")
    c2.metric("HbA1c", f"{linha['hba1c_pct']:.1f}%" if linha["hba1c_pct"] == linha["hba1c_pct"] else "—")
    c3.metric("IMC", f"{linha['imc']:.1f}" if linha["imc"] == linha["imc"] else "—")
    c4.metric("Status laboratorial", linha["status_diabetes_laboratorial"])

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classificação de diabetes")
        st.write(f"**Status:** {classify_diabetes_status(vetor)}")
        autorrelato = {1.0: "Sim", 2.0: "Não", 3.0: "Borderline"}.get(linha["diabetes_autorreferido"], "Não informado")
        st.write(f"**Autorrelato:** {autorrelato}")
    with col2:
        st.subheader("Síndrome metabólica")
        st.json(classify_metabolic_syndrome_risk(vetor))

    st.divider()
    st.subheader("Vetor de representação completo")
    for dom in pipeline.representation.domain_names():
        st.write(f"**{dom}**")
        st.write({f.name: (f"{f.raw_value} (z={f.value:.2f})" if not f.is_missing else "ausente") for f in vetor.components[dom]})
