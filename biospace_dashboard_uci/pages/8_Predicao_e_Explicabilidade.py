import components._bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from components.charts import apply_default_layout
from components.state import require_pipeline
from biospace.core import RepresentationSpace
from biospace.explainability import explain_predictor
from biospace.prediction import SklearnPredictor

st.set_page_config(page_title="Predição e Explicabilidade - UCI", page_icon="🎯", layout="wide")

pipeline = require_pipeline()

st.title("🎯 Predição Clássica + Explicabilidade (SHAP)")
st.caption(
    "A mesma pergunta da página de Sobrevivência, respondida por dois métodos completamente diferentes: "
    "RandomForest e Regressão Logística, treinados só com Features do 1º encontro (sem olhar pro futuro), "
    "prevendo readmissão precoce nos encontros seguintes."
)
st.warning(
    "**Achado documentado, triangulado com Sobrevivência e Alerta Precoce**: os dois classificadores ficam "
    "no mesmo teto de acaso (AUC≈0,50-0,53) que o Cox por fenótipo — três métodos diferentes convergindo. "
    "O SHAP (abaixo) mostra por quê: importância difusa entre as Features, nenhuma dominando."
)

df = pipeline.display_df
multi = df[df["n_encontros"] >= 4]
st.metric("Pacientes com ≥4 encontros nesta amostra", len(multi))

if len(multi) < 30:
    st.error("Poucos pacientes com ≥4 encontros nesta amostra — use a base completa na página inicial.")
    st.stop()

if st.button("Treinar classificadores e explicar com SHAP", type="primary"):
    order = pipeline.representation.domain_names()

    with st.spinner("Extraindo baseline, treinando RandomForest/LogisticRegression, validação cruzada, calculando SHAP..."):
        ids_elegiveis, labels = [], {}
        for sid in multi["id"]:
            obs = pipeline.cohort.systems[sid].observations
            if len(obs) < 4:
                continue
            evento = any(o.metadata.get("readmitted") == "<30" for o in obs[3:])
            ids_elegiveis.append(sid)
            labels[sid] = int(evento)

        space_baseline = RepresentationSpace(domain_order=order)
        for sid in ids_elegiveis:
            space_baseline.add(pipeline.cohort.trajectories[sid].at(0))

        matrix, ids_ordem = space_baseline.matrix()
        y = [labels[sid] for sid in ids_ordem]

        erro = None
        try:
            cv = StratifiedKFold(n_splits=min(5, sum(y)), shuffle=True, random_state=0) if 0 < sum(y) < len(y) else None
            if cv is None:
                raise ValueError("Amostra sem variação suficiente no rótulo (todos com ou sem evento) para validação cruzada.")

            scores_rf = cross_val_score(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced"), matrix, y, cv=cv, scoring="roc_auc")
            scores_lr = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced"), matrix, y, cv=cv, scoring="roc_auc")

            predictor = SklearnPredictor(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced"))
            predictor.fit(space_baseline, labels)
            relatorio_shap = explain_predictor(predictor, space_baseline, representation=pipeline.representation)
        except Exception as e:
            erro = str(e)
            scores_rf = scores_lr = relatorio_shap = None

        st.session_state["_uci_prediction"] = (scores_rf, scores_lr, relatorio_shap, erro, len(ids_elegiveis))

if "_uci_prediction" in st.session_state:
    scores_rf, scores_lr, relatorio_shap, erro, n = st.session_state["_uci_prediction"]

    if erro:
        st.error(f"Não foi possível treinar nesta amostra: {erro}")
        st.info("Tente a base completa na página inicial — amostras pequenas costumam ter poucos casos de evento.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pacientes na análise", n)
        c2.metric("AUC RandomForest (5-fold CV)", f"{scores_rf.mean():.3f} ± {scores_rf.std():.3f}")
        c3.metric("AUC LogisticRegression (5-fold CV)", f"{scores_lr.mean():.3f} ± {scores_lr.std():.3f}")

        st.divider()
        st.subheader("Explicabilidade (SHAP) — RandomForest")
        st.caption("Importância |SHAP| média por Feature — quanto mais concentrada num topo, mais o modelo depende de poucas variáveis. Aqui, difusa.")

        df_shap = pd.DataFrame(relatorio_shap.mean_abs_shap.items(), columns=["feature", "shap_medio"]).sort_values("shap_medio", ascending=True)
        fig = px.bar(df_shap, x="shap_medio", y="feature", orientation="h", title="Importância SHAP por Feature (baseline, 1º encontro)")
        st.plotly_chart(apply_default_layout(fig), use_container_width=True)
