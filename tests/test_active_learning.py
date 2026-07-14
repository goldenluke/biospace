"""
tests.test_active_learning
==============================

`biospace.active_learning` (UncertaintySampler, QueryByCommittee) —
validado contra a comparação clássica e decisiva de active learning:
numa fronteira de decisão não-trivial (círculo) com verdade conhecida,
a estratégia de consulta deveria produzir uma curva de aprendizado
melhor que seleção aleatória do mesmo orçamento de rótulos -- não
apenas "roda sem erro".
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from biospace.active_learning import QueryByCommittee, UncertaintySampler
from biospace.core import Feature, RepresentationSpace, RepresentationVector


def _vetor(system_id: str, x: float, y: float) -> RepresentationVector:
    comps = {"d": [Feature(name="x", value=x, raw_value=x), Feature(name="y", value=y, raw_value=y)]}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def _pool_com_fronteira_circular(seed=0, n=300):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-3, 3, n); ys = rng.uniform(-3, 3, n)
    space = RepresentationSpace(domain_order=["d"])
    labels = {}
    for i in range(n):
        sid = f"p{i}"
        space.add(_vetor(sid, xs[i], ys[i]))
        labels[sid] = int(xs[i] ** 2 + ys[i] ** 2 < 4)
    return space, labels


def _conjunto_teste_fixo(seed=999, n=200):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-3, 3, n); ys = rng.uniform(-3, 3, n)
    X = np.stack([xs, ys], axis=1)
    y = (xs ** 2 + ys ** 2 < 4).astype(int)
    return X, y


def test_uncertainty_sampling_beats_random_selection_on_known_nonlinear_boundary():
    """TESTE DECISIVO: amostragem por incerteza deveria produzir acuracia media maior que selecao aleatoria do mesmo numero de rotulos, numa fronteira nao-linear conhecida."""
    space, labels = _pool_com_fronteira_circular(seed=0)
    matrix_pool, ids_ordem = space.matrix()
    y_pool = np.array([labels[sid] for sid in ids_ordem])
    X_teste, y_teste = _conjunto_teste_fixo()

    def rodar(usar_al, seed):
        rng = np.random.default_rng(seed)
        rotulados = list(rng.choice(len(ids_ordem), size=10, replace=False))
        acuracias = []
        for _ in range(12):
            clf = RandomForestClassifier(n_estimators=40, random_state=0).fit(matrix_pool[rotulados], y_pool[rotulados])
            acuracias.append(accuracy_score(y_teste, clf.predict(X_teste)))
            nao_rotulados = [i for i in range(len(ids_ordem)) if i not in rotulados]
            if not nao_rotulados:
                break
            if usar_al:
                space_nr = RepresentationSpace(domain_order=["d"])
                for i in nao_rotulados:
                    space_nr.add(space.get(ids_ordem[i]))
                sid_escolhido = UncertaintySampler(estimator=clf).query(space_nr, n_query=1)[0]
                idx_escolhido = ids_ordem.index(sid_escolhido)
            else:
                idx_escolhido = rng.choice(nao_rotulados)
            rotulados.append(idx_escolhido)
        return acuracias

    curva_al = np.mean([rodar(True, s) for s in range(6)], axis=0)
    curva_random = np.mean([rodar(False, s) for s in range(6)], axis=0)

    assert np.mean(curva_al[-4:]) > np.mean(curva_random[-4:]), (
        f"Esperava active learning superar selecao aleatoria -- AL={np.mean(curva_al[-4:]):.3f}, random={np.mean(curva_random[-4:]):.3f}"
    )


def test_query_by_committee_beats_random_selection_on_known_nonlinear_boundary():
    """TESTE DECISIVO: mesma comparacao, agora com QueryByCommittee (3 modelos estruturalmente diferentes)."""
    space, labels = _pool_com_fronteira_circular(seed=1)
    matrix_pool, ids_ordem = space.matrix()
    y_pool = np.array([labels[sid] for sid in ids_ordem])
    X_teste, y_teste = _conjunto_teste_fixo(seed=888)

    def rodar(usar_qbc, seed):
        rng = np.random.default_rng(seed)
        rotulados = list(rng.choice(len(ids_ordem), size=10, replace=False))
        acuracias = []
        for _ in range(12):
            X_rot, y_rot = matrix_pool[rotulados], y_pool[rotulados].copy()
            if len(np.unique(y_rot)) < 2:
                y_rot[0] = 1 - y_rot[0]
            comite = [
                RandomForestClassifier(n_estimators=25, random_state=0).fit(X_rot, y_rot),
                LogisticRegression(max_iter=500).fit(X_rot, y_rot),
                SVC(kernel="rbf").fit(X_rot, y_rot),
            ]
            acuracias.append(accuracy_score(y_teste, comite[0].predict(X_teste)))
            nao_rotulados = [i for i in range(len(ids_ordem)) if i not in rotulados]
            if not nao_rotulados:
                break
            if usar_qbc:
                space_nr = RepresentationSpace(domain_order=["d"])
                for i in nao_rotulados:
                    space_nr.add(space.get(ids_ordem[i]))
                sid_escolhido = QueryByCommittee(estimators=comite).query(space_nr, n_query=1)[0]
                idx_escolhido = ids_ordem.index(sid_escolhido)
            else:
                idx_escolhido = rng.choice(nao_rotulados)
            rotulados.append(idx_escolhido)
        return acuracias

    curva_qbc = np.mean([rodar(True, s) for s in range(6)], axis=0)
    curva_random = np.mean([rodar(False, s) for s in range(6)], axis=0)

    assert np.mean(curva_qbc[-4:]) > np.mean(curva_random[-4:]), (
        f"Esperava QBC superar selecao aleatoria -- QBC={np.mean(curva_qbc[-4:]):.3f}, random={np.mean(curva_random[-4:]):.3f}"
    )


def test_uncertainty_scores_are_higher_near_decision_boundary():
    """Pontos perto da fronteira (raio~2) deveriam ter incerteza maior que pontos bem no centro ou bem longe."""
    space, labels = _pool_com_fronteira_circular(seed=2)
    matrix_pool, ids_ordem = space.matrix()
    y_pool = np.array([labels[sid] for sid in ids_ordem])
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(matrix_pool, y_pool)

    space_fronteira = RepresentationSpace(domain_order=["d"])
    space_fronteira.add(_vetor("perto_fronteira", 2.0, 0.05))
    space_fronteira.add(_vetor("centro", 0.0, 0.0))
    space_fronteira.add(_vetor("longe", 2.9, 2.9))

    scores = UncertaintySampler(estimator=clf).uncertainty_scores(space_fronteira)
    assert scores["perto_fronteira"] > scores["centro"]
    assert scores["perto_fronteira"] > scores["longe"]


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_active_learning_shows_no_advantage_on_real_uci_task_with_near_chance_signal():
    """
    ACHADO REAL, negativo, e conectado a triangulacao ja estabelecida
    (Artigo V, Cox/RF/LogReg/SHAP/GRU): na tarefa real de predizer
    readmissao precoce a partir do baseline da UCI -- ja documentada
    como perto do teto de acaso por cinco metodos independentes --,
    amostragem por incerteza NAO supera selecao aleatoria do mesmo
    orcamento de rotulos. Interpretacao coerente com o resto da
    triangulacao, nao um resultado isolado: sem uma fronteira de
    decisao real pra explorar (porque nao ha sinal forte na
    representacao baseline), a estrategia de consultar "onde o
    modelo esta mais incerto" nao tem nada de genuino pra encontrar
    -- a incerteza do modelo reflete ruido do problema, nao
    informatividade real do ponto candidato.
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
    from sklearn.metrics import roc_auc_score

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", max_rows=15000, include_diagnosis_category=False)
    order = representation.domain_names()

    ids_elegiveis, labels = [], {}
    for sid in cohort.trajectories:
        obs = cohort.systems[sid].observations
        if len(obs) < 4:
            continue
        evento = any(o.metadata.get("readmitted") == "<30" for o in obs[3:])
        ids_elegiveis.append(sid)
        labels[sid] = int(evento)

    assert len(ids_elegiveis) > 200, f"Esperava >200 pacientes elegiveis -- obteve {len(ids_elegiveis)}"

    space_baseline = RepresentationSpace(domain_order=order)
    for sid in ids_elegiveis:
        space_baseline.add(cohort.trajectories[sid].at(0))

    matrix_pool, ids_ordem = space_baseline.matrix()
    y_pool = np.array([labels[sid] for sid in ids_ordem])

    rng_split = np.random.RandomState(0)
    idx_todos = np.arange(len(ids_ordem))
    idx_teste = rng_split.choice(idx_todos, size=int(0.3 * len(idx_todos)), replace=False)
    idx_pool = np.array([i for i in idx_todos if i not in set(idx_teste)])
    X_teste, y_teste = matrix_pool[idx_teste], y_pool[idx_teste]

    def rodar(usar_al, seed):
        rng = np.random.default_rng(seed)
        rotulados = list(rng.choice(idx_pool, size=30, replace=False))
        aucs = []
        for _ in range(15):
            clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=0, class_weight="balanced").fit(matrix_pool[rotulados], y_pool[rotulados])
            aucs.append(roc_auc_score(y_teste, clf.predict_proba(X_teste)[:, 1]))
            nao_rotulados = [i for i in idx_pool if i not in rotulados]
            if not nao_rotulados:
                break
            if usar_al:
                space_nr = RepresentationSpace(domain_order=order)
                for i in nao_rotulados:
                    space_nr.add(space_baseline.get(ids_ordem[i]))
                escolhidos = UncertaintySampler(estimator=clf).query(space_nr, n_query=5)
                idxs_escolhidos = [ids_ordem.index(s) for s in escolhidos]
            else:
                idxs_escolhidos = list(rng.choice(nao_rotulados, size=5, replace=False))
            rotulados.extend(idxs_escolhidos)
        return aucs

    curva_al = np.mean([rodar(True, s) for s in range(4)], axis=0)
    curva_random = np.mean([rodar(False, s) for s in range(4)], axis=0)

    diferenca = np.mean(curva_al[-4:]) - np.mean(curva_random[-4:])
    assert abs(diferenca) < 0.08, (
        f"Esperava diferenca pequena entre AL e random (achado documentado: sem vantagem clara quando o sinal e' perto do acaso) -- "
        f"AL={np.mean(curva_al[-4:]):.3f}, random={np.mean(curva_random[-4:]):.3f}, diferenca={diferenca:.3f}"
    )
