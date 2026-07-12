"""
tests.test_prediction
=========================

`biospace.prediction` (Predictor/SklearnPredictor) existia sem
NENHUM teste antes desta rodada — achado numa auditoria do projeto.
Testado aqui com dado fabricado e controlado antes de qualquer
aplicação a dado real, mesma disciplina do resto do projeto.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from biospace.core import Feature, Observable, Observation, Representation, RepresentationSpace, SemanticDomain
from biospace.prediction import SklearnPredictor


class _ObsA(Observable):
    key = "a"


class _ObsB(Observable):
    key = "b"


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_ObsA(), _ObsB()])

    def encode(self, measurements):
        return [Feature(name="a", value=float(measurements["a"].value), raw_value=float(measurements["a"].value)),
                Feature(name="b", value=float(measurements["b"].value), raw_value=float(measurements["b"].value))]


_REPRESENTATION = Representation([_Dom()])


def _fabricar_space(n=100, seed=0):
    """Dado linearmente separavel por 'a': label = 1 se a>0, senao 0 -- verdade conhecida, RandomForest/LogisticRegression deveriam acertar quase tudo."""
    from biospace.core import BiologicalSystem, Cohort

    rng = np.random.default_rng(seed)
    cohort = Cohort()
    labels = {}
    for i in range(n):
        a = rng.normal(0, 1)
        b = rng.normal(0, 1)
        sid = f"s{i}"
        system = BiologicalSystem(identifier=sid)
        system.observe(Observation(timestamp=datetime(2020, 1, 1), source="t", values={"a": a, "b": b}))
        cohort.update(system, _REPRESENTATION, timestamp=datetime(2020, 1, 1))
        labels[sid] = 1 if a > 0 else 0

    return cohort.snapshot(), labels


def test_fit_recovers_known_separable_labels_with_high_accuracy():
    """TESTE DECISIVO: dado linearmente separavel (verdade conhecida) -- RandomForest deveria acertar quase todas as predicoes in-sample."""
    space, labels = _fabricar_space(n=200, seed=0)
    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=50, random_state=0))
    predicoes = predictor.fit(space, labels)

    acertos = sum(1 for sid, pred in predicoes.items() if pred == labels[sid])
    acuracia = acertos / len(labels)
    assert acuracia > 0.9, f"Esperava acuracia alta em dado linearmente separavel -- obteve {acuracia:.3f}"


def test_predict_matches_fit_predictions_on_same_space():
    """predict() apos fit() no MESMO space deveria dar as mesmas predicoes que fit() ja devolveu (mesmo estimador, mesmo dado)."""
    space, labels = _fabricar_space(n=100, seed=1)
    predictor = SklearnPredictor(LogisticRegression())
    predicoes_fit = predictor.fit(space, labels)
    predicoes_predict = predictor.predict(space)
    assert predicoes_fit == predicoes_predict


def test_fit_raises_clear_error_on_missing_labels():
    """Rotulos faltando para algum system_id deveriam falhar alto e claro, nao silenciosamente ignorar ou quebrar com KeyError generico do sklearn."""
    space, labels = _fabricar_space(n=10, seed=2)
    labels_incompletos = {k: v for i, (k, v) in enumerate(labels.items()) if i < 5}
    predictor = SklearnPredictor(RandomForestClassifier(random_state=0))
    with pytest.raises(KeyError, match="[Rr]ótulos|[Ff]alta"):
        predictor.fit(space, labels_incompletos)


def test_works_polymorphically_with_different_sklearn_estimators():
    """Trocar de RandomForest para LogisticRegression nao deveria exigir nenhuma outra mudanca -- o ponto central do design (SklearnPredictor envelopa QUALQUER estimador compativel)."""
    space, labels = _fabricar_space(n=150, seed=3)
    for estimator in [RandomForestClassifier(n_estimators=30, random_state=0), LogisticRegression()]:
        predictor = SklearnPredictor(estimator)
        predicoes = predictor.fit(space, labels)
        acuracia = sum(1 for sid, pred in predicoes.items() if pred == labels[sid]) / len(labels)
        assert acuracia > 0.85, f"{estimator.__class__.__name__}: esperava acuracia alta -- obteve {acuracia:.3f}"


def test_predict_works_on_different_space_than_fit():
    """predict() deveria funcionar sobre um RepresentationSpace DIFERENTE do usado em fit() -- o caso de uso real (pacientes novos)."""
    space_treino, labels_treino = _fabricar_space(n=150, seed=4)
    space_teste, labels_teste = _fabricar_space(n=50, seed=5)

    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=50, random_state=0))
    predictor.fit(space_treino, labels_treino)
    predicoes_teste = predictor.predict(space_teste)

    assert set(predicoes_teste.keys()) == set(space_teste.ids())
    acuracia = sum(1 for sid, pred in predicoes_teste.items() if pred == labels_teste[sid]) / len(labels_teste)
    assert acuracia > 0.8, f"Esperava generalizar bem para dado nao visto (linearmente separavel) -- obteve {acuracia:.3f}"


def test_describe_mentions_the_wrapped_estimator_class():
    predictor = SklearnPredictor(RandomForestClassifier())
    assert "RandomForestClassifier" in predictor.describe()


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_classical_supervised_learning_does_not_beat_phenotype_based_prediction_on_uci():
    """
    ACHADO REAL, TRIANGULADO com `test_survival.py` e `test_early_warning.py`:
    RandomForest e LogisticRegression, treinados sobre TODAS as
    Features do 1o encontro (baseline, mesmo desenho de
    biospace.survival) para prever readmissao precoce SUBSEQUENTE (4o
    encontro em diante), NAO superam o C-index~0.52 ja encontrado pelo
    modelo de Cox baseado em fenotipo -- validado com 5-fold CV,
    RandomForest fica em ~0.50-0.52, LogisticRegression em ~0.53. As
    TRES abordagens (fenotipo+Cox, RandomForest, LogisticRegression)
    convergem para a MESMA conclusao -- forte evidencia de que a
    limitacao e' do DADO (informacao insuficiente no 1o encontro para
    predizer o futuro), nao do METODO escolhido.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from biospace.core import RepresentationSpace
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", include_diagnosis_category=False)
    order = representation.domain_names()

    ids_elegiveis = []
    labels = {}
    for sid, system in cohort.systems.items():
        obs = system.observations
        if len(obs) < 4:
            continue
        evento = any(o.metadata.get("readmitted") == "<30" for o in obs[3:])
        ids_elegiveis.append(sid)
        labels[sid] = int(evento)

    assert len(ids_elegiveis) > 2900, f"Esperava >2900 pacientes elegiveis (achado documentado: 3011) -- obteve {len(ids_elegiveis)}"

    space_baseline = RepresentationSpace(domain_order=order)
    for sid in ids_elegiveis:
        space_baseline.add(cohort.trajectories[sid].at(0))

    matrix, ids_ordem = space_baseline.matrix()
    y = np.array([labels[sid] for sid in ids_ordem])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores_rf = cross_val_score(RFC(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced"), matrix, y, cv=cv, scoring="roc_auc")
    scores_lr = cross_val_score(LR(max_iter=1000, class_weight="balanced"), matrix, y, cv=cv, scoring="roc_auc")

    assert scores_rf.mean() < 0.60, f"Esperava AUC proximo do acaso para RandomForest (achado documentado) -- obteve {scores_rf.mean():.3f}"
    assert scores_lr.mean() < 0.60, f"Esperava AUC proximo do acaso para LogisticRegression (achado documentado) -- obteve {scores_lr.mean():.3f}"
