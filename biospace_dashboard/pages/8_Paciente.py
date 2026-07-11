import components._bootstrap  # noqa: F401

import streamlit as st

from components.state import require_pipeline

st.set_page_config(page_title="Paciente · BioSpace", page_icon="🧑‍⚕️", layout="wide")

pipeline = require_pipeline()
df = pipeline.display_df

st.title("🧑‍⚕️ Consulta Individual")

busca = st.text_input("Buscar por identificador do paciente (ex.: LS_000001, DEMO_0001)")

if busca:
    resultados = df[df["paciente"].astype(str).str.contains(busca, case=False, na=False)]
else:
    resultados = df.head(0)

if busca and len(resultados) == 0:
    st.warning("Nenhum paciente encontrado com esse identificador.")
elif len(resultados) > 0:
    ids = resultados["id"].tolist()
    selected_id = st.selectbox("Selecione o registro", ids, format_func=lambda i: resultados.set_index("id").loc[i, "paciente"])
    row = resultados.set_index("id").loc[selected_id]

    st.subheader(f"Paciente: {row['paciente']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Idade", row["idade"])
    c2.metric("IMC", round(row["imc"], 1) if row["imc"] == row["imc"] else "—")
    c3.metric("Classe (IDO)", row["classe_apneia"] or "—")
    c4.metric("Fenótipo", row["fenotipo"] or "—")

    tab1, tab2, tab3 = st.tabs(["Respiração / Hipoxemia", "Cardiovascular / Sono", "Clínico (texto)"])
    with tab1:
        st.write(f"**IDO:** {row['ido']}")
        st.write(f"**IDO-sono:** {row.get('ido_sono')}")
        st.write(f"**Nº de dessaturações:** {row.get('no_de_dessaturacoes')}")
        st.write(f"**SpO2 mínima / média / máxima:** {row.get('spo2_minima')} / {row.get('spo2_media')} / {row.get('spo2_maxima')}")
        st.write(f"**Carga hipóxica (%.min/h):** {row.get('carga_hipoxica_min_h')}")
        st.write(f"**Classe de hipoxemia:** {row.get('classe_hipoxemia')}")
    with tab2:
        st.write(f"**FC mínima / média / máxima:** {row.get('fc_minima_bpm')} / {row.get('fc_media_bpm')} / {row.get('fc_maxima_bpm')}")
        st.write(f"**Amplitude cardíaca:** {row.get('amplitude_fc')}")
        st.write(f"**Tempo total de sono (min):** {row.get('tempo_total_de_sono_min')}")
        st.write(f"**Eficiência do sono (%):** {row.get('eficiencia_do_sono')}")
    with tab3:
        st.write(f"**Comorbidades:** {row.get('doencas') or '—'}")
        st.write(f"**Sintomas:** {row.get('sintomas') or '—'}")
        st.write(f"**Tratamentos:** {row.get('tratamentos') or '—'}")

    st.divider()
    st.subheader("Trajetória (se houver mais de um exame)")
    system_id = selected_id
    if system_id in pipeline.cohort.trajectories:
        traj = pipeline.cohort.trajectories[system_id]
        if len(traj) > 1:
            for i in range(len(traj)):
                vec = traj.at(i)
                st.write(f"- {vec.timestamp}: domínios={list(vec.components.keys())}")
        else:
            st.caption("Apenas um exame registrado para este paciente — nada para mostrar como trajetória ainda.")
else:
    st.caption("Digite um identificador acima para buscar um paciente.")
