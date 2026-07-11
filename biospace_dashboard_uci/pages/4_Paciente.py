import components._bootstrap  # noqa: F401

import streamlit as st

from components.state import require_pipeline

st.set_page_config(page_title="Paciente - UCI", page_icon="🔍", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🔍 Busca de Paciente")

labels_map = {row["id"]: f"{row['paciente']} ({row['n_encontros']} encontros)" for _, row in df.iterrows()}
escolhido = st.selectbox("Paciente", list(labels_map.keys()), format_func=lambda s: labels_map[s])

if escolhido:
    linha = df[df["id"] == escolhido].iloc[0]
    vetor = pipeline.space.get(escolhido)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nº de encontros", int(linha["n_encontros"]))
    c2.metric("Fenótipo", linha["fenotipo"])
    c3.metric("Readmissão (último encontro)", linha["readmitted_ultimo_encontro"])
    c4.metric("Internações prévias", f"{linha['number_inpatient']:.0f}" if linha["number_inpatient"] == linha["number_inpatient"] else "—")

    st.divider()
    st.subheader("Vetor de representação (último encontro)")
    for dom in pipeline.representation.domain_names():
        st.write(f"**{dom}**")
        st.write({f.name: (f"{f.raw_value}" if not f.is_missing else "ausente") for f in vetor.components[dom]})
