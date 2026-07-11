import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.early_warning import CriticalSlowingDownDetector

st.set_page_config(page_title="Early Warning · BioSpace", page_icon="🚨", layout="wide")

pipeline = require_pipeline()

st.title("🚨 Early Warning Signals (Critical Slowing Down)")
st.caption(
    "CriticalSlowingDownDetector: 3 indicadores (variância, autocorrelação lag-1, assimetria) em janela "
    "deslizante sobre a trajetória de cada paciente, com significância por dados substitutos AR(1) "
    "(Dakos et al. 2012) e um critério de PESO DE EVIDÊNCIA (maioria dos indicadores concorda, não unanimidade)."
)

with st.expander("⚠️ Leia antes de interpretar qualquer resultado aqui", expanded=True):
    st.markdown(
        """
- Só pacientes com pelo menos `min_points` exames produzem resultado — a maioria da coorte real
  tem poucos exames (mediana de 4), então normalmente uma **minoria** dos pacientes é elegível.
- **Janela por pontos** (padrão): `window_size` exames consecutivos. **Janela por dias**: todos os
  exames dentro de uma janela de tempo fixa — mitiga (não elimina) o problema de amostragem
  irregular, mas cobertura populacional varia bastante com o tamanho da janela escolhida
  (testamos: 90 dias cobre 23/53 elegíveis, 180 dias cobre 34/53, 365 dias cobre 44/53).
- O `warning` agora exige que a MAIORIA dos indicadores disponíveis (não todos) concordem — mais
  robusto a um único indicador ruidoso, mas ainda é um sinal **heurístico e exploratório**.
- Testamos a correlação entre τ e progressão real de severidade nesta mesma coorte e o resultado
  foi **misto e não conclusivo** (ver seção "Validação" abaixo) — trate com cautela.
        """
    )

st.divider()

# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    tipo_indicador = st.selectbox(
        "Indicador univariado",
        ["Distância multivariada à baseline", "IDO (apnea.ido)", "SpO2 mínima (hypoxia.spo2_minima)", "FC média (cardiovascular.fc_media_bpm)"],
    )
    detrend_method = st.selectbox("Método de detrend", ["linear", "gaussian"], help="Gaussiano captura tendências não lineares.")
with col2:
    min_points = st.slider("Mínimo de exames por paciente", 5, 15, value=8)
    n_surrogates = st.slider("Nº de substitutos AR(1)", 50, 500, value=200, step=50)

window_mode = st.radio("Modo de janela", ["points (nº de exames)", "days (tempo decorrido)"], horizontal=True)

col3, col4 = st.columns(2)
if window_mode.startswith("points"):
    with col3:
        window_size = st.slider("Tamanho da janela (nº de exames)", 3, 8, value=4)
    window_kwargs = {"window_mode": "points", "window_size": window_size}
else:
    with col3:
        window_days = st.slider("Tamanho da janela (dias)", 30, 400, value=180, step=30)
    with col4:
        min_points_per_window = st.slider("Mínimo de exames por janela", 2, 6, value=3)
    window_kwargs = {"window_mode": "days", "window_days": float(window_days), "min_points_per_window": min_points_per_window}

st.caption("O teste de substitutos roda o pipeline completo `n_surrogates` vezes por paciente elegível — pode levar de segundos a ~1 min.")

if st.button("Rodar detecção", type="primary"):
    with st.spinner(f"Rodando Critical Slowing Down sobre a coorte ({n_surrogates} substitutos por paciente elegível)..."):
        common_kwargs = dict(min_points=min_points, n_surrogates=n_surrogates, detrend_method=detrend_method, **window_kwargs)
        if tipo_indicador == "Distância multivariada à baseline":
            detector = CriticalSlowingDownDetector.for_distance_from_baseline(order=pipeline.space.order(), **common_kwargs)
        else:
            domain_name, feature_name = {
                "IDO (apnea.ido)": ("apnea", "ido"),
                "SpO2 mínima (hypoxia.spo2_minima)": ("hypoxia", "spo2_minima"),
                "FC média (cardiovascular.fc_media_bpm)": ("cardiovascular", "fc_media_bpm"),
            }[tipo_indicador]
            detector = CriticalSlowingDownDetector.for_feature(domain_name, feature_name, **common_kwargs)

        results = detector.fit(pipeline.cohort)

    st.session_state["_ews_results"] = results
    st.session_state["_ews_indicador"] = tipo_indicador

if "_ews_results" not in st.session_state:
    st.info("Configure os parâmetros acima e clique em 'Rodar detecção'.")
    st.stop()

results = st.session_state["_ews_results"]
indicador_usado = st.session_state["_ews_indicador"]

elegiveis = {sid: r for sid, r in results.items() if r.sufficient_data}
com_indicador = {sid: r for sid, r in elegiveis.items() if r.n_indicators_available > 0}
com_warning = {sid: r for sid, r in elegiveis.items() if r.warning}

st.divider()
st.subheader(f"Resultado ({indicador_usado})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pacientes na coorte", len(results))
c2.metric("Elegíveis (dados suficientes)", len(elegiveis))
c3.metric("Com >=1 indicador calculável", len(com_indicador))
c4.metric("Com sinal de alerta (peso de evidência)", len(com_warning))

if not elegiveis:
    st.warning("Nenhum paciente elegível com estes parâmetros — tente reduzir o mínimo de exames.")
    st.stop()
if not com_indicador:
    st.warning(
        "Nenhum paciente produziu indicadores calculáveis com esta configuração de janela — "
        "tente uma janela maior (mais dias, ou reduza o mínimo de pontos por janela)."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Tabela de resultados
# -----------------------------------------------------------------------------
rows = []
for sid, r in com_indicador.items():
    paciente = pipeline.cohort.systems[sid].metadata.get("paciente_original", sid)
    rows.append(
        {
            "paciente": paciente,
            "id": sid,
            "n_exames": r.n_points,
            "n_indicadores": r.n_indicators_available,
            "tau_variancia": round(r.tau_variance, 3) if r.tau_variance is not None else None,
            "tau_autocorrelacao": round(r.tau_autocorrelation, 3) if r.tau_autocorrelation is not None else None,
            "tau_assimetria": round(r.tau_skewness, 3) if r.tau_skewness is not None else None,
            "indicadores_concordantes": f"{r.n_indicators_rising_significant}/{r.n_indicators_available}",
            "warning": r.warning,
        }
    )
results_df = pd.DataFrame(rows).sort_values("indicadores_concordantes", ascending=False)
st.dataframe(results_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------------------------------------------------------
# Detalhe de um paciente
# -----------------------------------------------------------------------------
st.subheader("Detalhe por paciente")
paciente_escolhido = st.selectbox("Paciente", results_df["id"].tolist(), format_func=lambda sid: results_df.set_index("id").loc[sid, "paciente"])
r = com_indicador[paciente_escolhido]

c1, c2, c3 = st.columns(3)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.series.times_days, y=r.series.variance, mode="lines+markers", name="Variância"))
    fig.update_layout(title="Variância", xaxis_title="Dias desde o 1º exame", yaxis_title="Variância")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.series.times_days, y=r.series.autocorrelation, mode="lines+markers", name="Autocorrelação", line_color="orange"))
    fig.update_layout(title="Autocorrelação lag-1", xaxis_title="Dias desde o 1º exame", yaxis_title="Autocorrelação")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
with c3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.series.times_days, y=r.series.skewness, mode="lines+markers", name="Assimetria", line_color="green"))
    fig.update_layout(title="Assimetria (skewness)", xaxis_title="Dias desde o 1º exame", yaxis_title="Assimetria")
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)

st.code(r.summary(), language=None)

st.divider()

# -----------------------------------------------------------------------------
# Validação: tau x progressão real de severidade
# -----------------------------------------------------------------------------
st.subheader("Validação: τ correlaciona com progressão real de severidade?")
st.caption(
    "Progressão = diferença de posição na ordem de severidade entre o fenótipo do PRIMEIRO e do "
    "ÚLTIMO exame de cada paciente (fenótipos já estimados na página 'Perfis'). "
    "⚠️ Resultado histórico com esta coorte real: MISTO e não conclusivo (ver README) — "
    "trocar o indicador já inverteu o sinal da correlação. Não tire conclusões clínicas de n pequeno."
)


def _severity_from_interpretation(ph) -> float:
    try:
        return float(ph.interpretation.split("severidade_relativa=")[1].split(",")[0])
    except (IndexError, ValueError):
        return float("-inf")


ordem_severidade = {ph.name: i for i, ph in enumerate(sorted(pipeline.phenotypes, key=_severity_from_interpretation))}
order = pipeline.space.order()

pares = []
for sid, r in com_indicador.items():
    if r.tau_variance is None:
        continue
    traj = pipeline.cohort.trajectories[sid]
    fen_primeiro = next((ph.name for ph in pipeline.phenotypes if ph.contains(traj.at(0).as_vector(order))), None)
    fen_ultimo = next((ph.name for ph in pipeline.phenotypes if ph.contains(traj.at(-1).as_vector(order))), None)
    if fen_primeiro is None or fen_ultimo is None:
        continue
    delta = ordem_severidade[fen_ultimo] - ordem_severidade[fen_primeiro]
    pares.append({"tau_variancia": r.tau_variance, "progressao_severidade": delta})

if len(pares) >= 4:
    val_df = pd.DataFrame(pares)
    if val_df["tau_variancia"].nunique() < 2 or val_df["progressao_severidade"].nunique() < 2:
        st.info("Todos os pacientes elegíveis têm o mesmo τ ou a mesma progressão de severidade — sem variação suficiente para calcular correlação.")
        st.stop()
    rho, p = spearmanr(val_df["tau_variancia"], val_df["progressao_severidade"])
    st.metric("Correlação (Spearman) τ_variância × progressão", f"ρ={rho:.3f}", f"p={p:.3f}")

    fig = px.scatter(
        val_df, x="tau_variancia", y="progressao_severidade",
        labels={"tau_variancia": "τ_variância", "progressao_severidade": "Δ posição de severidade (fenótipo)"},
    )
    st.plotly_chart(apply_default_layout(fig), use_container_width=True)
else:
    st.info("Poucos pacientes elegíveis com fenótipo definido no início e no fim para calcular a correlação.")
