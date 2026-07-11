import components._bootstrap  # noqa: F401

import streamlit as st

from components.state import require_pipeline
from biospace.ontology import Ontology

st.set_page_config(page_title="Ontologia · BioSpace", page_icon="📖", layout="wide")

pipeline = require_pipeline()

st.title("📖 Ontologia (Dicionário de Dados)")
st.caption(
    "Gerado automaticamente a partir da Representation atual — Ontology.from_representation() "
    "varre cada SemanticDomain e cataloga seus Observables. Não é escrito à mão; se um domínio "
    "mudar, o dicionário muda junto."
)

ontology = Ontology.from_representation(pipeline.representation, name="sleep")

c1, c2 = st.columns(2)
c1.metric("Domínios", len(ontology.domains))
c2.metric("Observables únicos", len(ontology.observables))

st.divider()

markdown_content = ontology.to_markdown()
st.markdown(markdown_content)

st.divider()
st.download_button(
    "⬇️ Baixar dicionário de dados (Markdown)",
    data=markdown_content,
    file_name="data_dictionary_sleep.md",
    mime="text/markdown",
)
