"""
tests.test_survival
=======================

Testa `biospace.survival` com dado FABRICADO e controlado antes de
qualquer aplicação a dado real -- mesma disciplina do resto do
projeto: a lógica de extração de tempo-até-evento e os ajustes
estatísticos (KM, log-rank, Cox) são verificados aqui contra cenários
onde a resposta certa é conhecida de antemão.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.core import BiologicalSystem, Cohort, Observation
from biospace.survival import build_discrete_time_to_event, fit_cox_model, kaplan_meier_by_group


def _cohort_controlado():
    cohort = Cohort()

    sa = BiologicalSystem(identifier="A")
    for i, readm in enumerate([None, "NO", "<30", "NO"]):
        sa.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={}, metadata={"readmitted": readm}))
    cohort.systems["A"] = sa

    sb = BiologicalSystem(identifier="B")
    for i, readm in enumerate([None, "NO", "NO", "NO"]):
        sb.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={}, metadata={"readmitted": readm}))
    cohort.systems["B"] = sb

    sc = BiologicalSystem(identifier="C")
    sc.observe(Observation(timestamp=datetime(2020, 1, 1), source="t", values={}, metadata={"readmitted": None}))
    cohort.systems["C"] = sc

    return cohort


def test_event_found_gives_correct_duration_and_event_flag():
    """Paciente A: evento na 3a observacao (indice 2, 2a observacao subsequente a baseline) -- duration=2, event=1."""
    resultado = build_discrete_time_to_event(_cohort_controlado(), event_fn=lambda o: o.metadata.get("readmitted") == "<30")
    linha_a = resultado.df[resultado.df["system_id"] == "A"].iloc[0]
    assert linha_a["duration"] == 2
    assert linha_a["event"] == 1


def test_no_event_gives_censored_at_last_observation():
    """Paciente B: nunca tem evento -- censurado (event=0) em duration=3 (3 observacoes subsequentes a baseline, nenhuma com evento)."""
    resultado = build_discrete_time_to_event(_cohort_controlado(), event_fn=lambda o: o.metadata.get("readmitted") == "<30")
    linha_b = resultado.df[resultado.df["system_id"] == "B"].iloc[0]
    assert linha_b["duration"] == 3
    assert linha_b["event"] == 0


def test_single_observation_patient_is_excluded_and_counted():
    """Paciente C: so 1 observacao -- excluido do dataset, mas contado explicitamente, nao descartado silenciosamente."""
    resultado = build_discrete_time_to_event(_cohort_controlado(), event_fn=lambda o: o.metadata.get("readmitted") == "<30")
    assert "C" not in set(resultado.df["system_id"])
    assert resultado.n_excluded_single_observation == 1
    assert resultado.n_included == 2


def test_baseline_covariate_uses_only_first_observation():
    """A covariavel deve vir da 1a observacao, nunca de uma observacao posterior (mesmo se o valor mudar ao longo da trajetoria)."""
    cohort = Cohort()
    s = BiologicalSystem(identifier="X")
    for i, (readm, marcador) in enumerate([(None, "baseline"), ("NO", "depois1"), ("<30", "depois2")]):
        s.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={}, metadata={"readmitted": readm, "marcador": marcador}))
    cohort.systems["X"] = s

    resultado = build_discrete_time_to_event(
        cohort, event_fn=lambda o: o.metadata.get("readmitted") == "<30", covariates={"marcador": lambda o: o.metadata.get("marcador")}
    )
    assert resultado.df.iloc[0]["marcador"] == "baseline", "Covariavel deveria vir da 1a observacao, nao de uma posterior ao evento."


def _cohort_dois_grupos(seed=0, n_por_grupo=30, prob_alto=0.6, prob_baixo=0.15):
    cohort = Cohort()
    rng = np.random.default_rng(seed)
    for grupo, prob_evento in [("alto_risco", prob_alto), ("baixo_risco", prob_baixo)]:
        for p in range(n_por_grupo):
            sid = f"{grupo}_{p}"
            s = BiologicalSystem(identifier=sid)
            n_obs = rng.integers(2, 8)
            teve_evento = False
            for i in range(n_obs):
                if i > 0 and not teve_evento and rng.random() < prob_evento:
                    readm, teve_evento = "<30", True
                else:
                    readm = "NO"
                s.observe(Observation(timestamp=datetime(2020, 1, 1) + timedelta(days=i), source="t", values={}, metadata={"readmitted": readm, "grupo": grupo}))
            cohort.systems[sid] = s
    return cohort


def test_kaplan_meier_detects_known_risk_difference_between_groups():
    """TESTE DECISIVO: grupo com probabilidade de evento 4x maior (fabricado, verdade conhecida) deve produzir log-rank significativo e mediana de sobrevivencia muito menor."""
    resultado = build_discrete_time_to_event(
        _cohort_dois_grupos(), event_fn=lambda o: o.metadata.get("readmitted") == "<30", covariates={"grupo": lambda o: o.metadata.get("grupo")}
    )
    relatorio = kaplan_meier_by_group(resultado.df, group_col="grupo")
    assert relatorio.logrank_p < 0.01, f"Esperava log-rank significativo (diferenca fabricada de risco) -- obteve p={relatorio.logrank_p:.4f}"
    assert relatorio.median_survival["alto_risco"] < relatorio.median_survival["baixo_risco"]


def test_cox_model_recovers_known_hazard_ratio_direction():
    """Cox deve dar HR>1 (mais risco) para o grupo fabricado como alto risco, com p pequeno."""
    resultado = build_discrete_time_to_event(
        _cohort_dois_grupos(), event_fn=lambda o: o.metadata.get("readmitted") == "<30", covariates={"grupo": lambda o: o.metadata.get("grupo")}
    )
    df = resultado.df.copy()
    df["alto_risco"] = (df["grupo"] == "alto_risco").astype(int)
    relatorio = fit_cox_model(df, covariate_cols=["alto_risco"])
    assert relatorio.hazard_ratios["alto_risco"] > 2.0, f"Esperava HR>2 (risco fabricado 4x maior) -- obteve {relatorio.hazard_ratios['alto_risco']:.3f}"
    assert relatorio.p_values["alto_risco"] < 0.01


def test_no_difference_when_groups_have_identical_risk():
    """CONTROLE NEGATIVO: dois grupos com a MESMA probabilidade de evento nao deveriam produzir log-rank significativo -- confirma que o teste discrimina, nao acusa diferenca sempre."""
    resultado = build_discrete_time_to_event(
        _cohort_dois_grupos(seed=1, prob_alto=0.3, prob_baixo=0.3), event_fn=lambda o: o.metadata.get("readmitted") == "<30",
        covariates={"grupo": lambda o: o.metadata.get("grupo")},
    )
    relatorio = kaplan_meier_by_group(resultado.df, group_col="grupo")
    assert relatorio.logrank_p > 0.05, f"Esperava NAO significativo (mesma probabilidade de evento nos dois grupos) -- obteve p={relatorio.logrank_p:.4f}"


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_baseline_phenotype_weakly_predicts_future_readmission_on_real_uci_data():
    """
    ACHADO REAL, que NUANCIA (nao contradiz) o achado publicado no
    artigo de diabetes: a associacao fenotipo->readmissao publicada
    (~2x, artigo, achado transversal -- fenotipo ajustado sobre
    features do estado MAIS RECENTE de cada paciente) e' um tipo de
    tarefa estatistica bem diferente e mais facil que PREDICAO
    prospectiva genuina (fenotipo ajustado SO no 1o encontro de cada
    paciente, sem olhar pro futuro -- Contrato de Temporalidade 5.7 --
    prevendo sobrevivencia livre de readmissao precoce nos encontros
    SEGUINTES).

    Nesta segunda formulacao, mais honesta para uma alegacao preditiva:
    log-rank ainda e' estatisticamente significativo (p<0.01, n=16.773,
    amostra grande), mas o efeito pratico e' modesto -- mediana de
    sobrevivencia varia so entre 4 e 5 encontros entre os grupos, e o
    indice de concordancia do Cox fica proximo de 0.5 (quase aleatorio)
    -- consistente com 75% dos pacientes tendo so 1-2 encontros de
    acompanhamento apos o baseline, pouca variacao temporal disponivel
    para o modelo discriminar. Reportado como limite honesto de dado,
    nao do metodo.
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
    from biospace.phenotyping import KMeansPhenotyper
    from biospace.survival import fit_cox_model, kaplan_meier_by_group

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", include_diagnosis_category=False)
    order = representation.domain_names()

    resultado = build_discrete_time_to_event(cohort, event_fn=lambda o: o.metadata.get("readmitted") == "<30")
    assert resultado.n_included > 16000, f"Esperava >16 mil pacientes multi-encontro (achado documentado: 16.773) -- obteve {resultado.n_included}"

    ids_multi = set(resultado.df["system_id"])
    space_baseline = RepresentationSpace(domain_order=order)
    for sid in ids_multi:
        space_baseline.add(cohort.trajectories[sid].at(0))

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space_baseline)
    labels = {}
    for sid in ids_multi:
        vec = space_baseline.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

    resultado.df["fenotipo_baseline"] = resultado.df["system_id"].map(labels)
    df = resultado.df.dropna(subset=["fenotipo_baseline"])

    relatorio_km = kaplan_meier_by_group(df, group_col="fenotipo_baseline")
    assert relatorio_km.logrank_p < 0.05, f"Esperava log-rank significativo mesmo que fraco (achado documentado) -- obteve p={relatorio_km.logrank_p:.4f}"

    import pandas as pd

    df_cox = pd.get_dummies(df, columns=["fenotipo_baseline"], drop_first=True)
    cols_dummy = [c for c in df_cox.columns if c.startswith("fenotipo_baseline_")]
    relatorio_cox = fit_cox_model(df_cox, covariate_cols=cols_dummy)
    assert relatorio_cox.concordance_index < 0.6, (
        f"Esperava concordancia MODESTA (achado documentado: predicao prospectiva e' muito mais fraca que a "
        f"caracterizacao transversal publicada) -- obteve C-index={relatorio_cox.concordance_index:.3f}"
    )
