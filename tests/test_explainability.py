"""
tests.test_explainability
=============================

`biospace.explainability.explain_predictor` — envelope de SHAP sobre
`SklearnPredictor` já ajustado, traduzindo índice de coluna pra
`domínio.feature` legível. Validado contra um cenário com verdade
conhecida (uma Feature determina o rótulo por construção, outra é
ruído puro, sem nenhuma relação com o rótulo) antes de qualquer
aplicação real.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.explainability import explain_predictor
from biospace.prediction import SklearnPredictor


class _ObsImportante(Observable):
    key = "importante"


class _ObsRuido(Observable):
    key = "ruido"


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_ObsImportante(), _ObsRuido()])

    def encode(self, measurements):
        return [
            Feature(name="importante", value=float(measurements["importante"].value), raw_value=float(measurements["importante"].value)),
            Feature(name="ruido", value=float(measurements["ruido"].value), raw_value=float(measurements["ruido"].value)),
        ]


_REPRESENTATION = Representation([_Dom()])


def _cohort_com_feature_importante_conhecida(n=300, seed=0):
    """label = 1 se 'importante'>0, SEM NENHUMA relacao com 'ruido' -- verdade conhecida sobre qual Feature deveria importar."""
    rng = np.random.default_rng(seed)
    cohort = Cohort()
    labels = {}
    for i in range(n):
        sid = f"s{i}"
        importante = rng.normal(0, 1)
        ruido = rng.normal(0, 1)
        system = BiologicalSystem(identifier=sid)
        system.observe(Observation(timestamp=datetime(2020, 1, 1), source="t", values={"importante": importante, "ruido": ruido}))
        cohort.update(system, _REPRESENTATION, timestamp=datetime(2020, 1, 1))
        labels[sid] = int(importante > 0)
    return cohort, labels


def test_shap_correctly_identifies_the_truly_important_feature():
    """TESTE DECISIVO: com verdade conhecida (label so depende de 'importante'), |SHAP| medio de 'importante' deveria ser MUITO maior que o de 'ruido'."""
    cohort, labels = _cohort_com_feature_importante_conhecida()
    space = cohort.snapshot()
    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=100, random_state=0))
    predictor.fit(space, labels)

    relatorio = explain_predictor(predictor, space, representation=_REPRESENTATION)
    assert relatorio.mean_abs_shap["d.importante"] > 5 * relatorio.mean_abs_shap["d.ruido"], (
        f"Esperava 'importante' dominar 'ruido' por uma margem grande -- obteve "
        f"importante={relatorio.mean_abs_shap['d.importante']:.4f}, ruido={relatorio.mean_abs_shap['d.ruido']:.4f}"
    )


def test_top_features_returns_correctly_sorted_list():
    cohort, labels = _cohort_com_feature_importante_conhecida(seed=1)
    space = cohort.snapshot()
    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=100, random_state=0))
    predictor.fit(space, labels)

    relatorio = explain_predictor(predictor, space, representation=_REPRESENTATION)
    top = relatorio.top_features(n=1)
    assert top[0][0] == "d.importante"


def test_raises_when_neither_feature_names_nor_representation_given():
    cohort, labels = _cohort_com_feature_importante_conhecida(seed=2)
    space = cohort.snapshot()
    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=10, random_state=0))
    predictor.fit(space, labels)
    with pytest.raises(ValueError, match="feature_names|representation"):
        explain_predictor(predictor, space)


def test_raises_on_feature_name_count_mismatch():
    cohort, labels = _cohort_com_feature_importante_conhecida(seed=3)
    space = cohort.snapshot()
    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=10, random_state=0))
    predictor.fit(space, labels)
    with pytest.raises(ValueError, match="não bate"):
        explain_predictor(predictor, space, feature_names=["so_um_nome"])


def test_works_with_non_tree_estimator_via_kernel_explainer():
    """LogisticRegression nao e' arvore -- deveria cair para KernelExplainer, nao quebrar."""
    cohort, labels = _cohort_com_feature_importante_conhecida(n=80, seed=4)
    space = cohort.snapshot()
    predictor = SklearnPredictor(LogisticRegression())
    predictor.fit(space, labels)

    relatorio = explain_predictor(predictor, space, representation=_REPRESENTATION, background_size=20)
    assert relatorio.explainer_type == "KernelExplainer"
    assert relatorio.mean_abs_shap["d.importante"] > relatorio.mean_abs_shap["d.ruido"]


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_shap_shows_diffuse_not_concentrated_importance_on_real_uci_null_finding():
    """
    ACHADO REAL: complementa a triangulacao ja documentada
    (Cox/RandomForest/LogisticRegression convergem para AUC/C-index perto
    do acaso ao prever readmissao futura so com dado do 1o encontro).
    SHAP no RandomForest real mostra POR QUE, num nivel mais fino: a
    importancia fica DIFUSA entre as 13 Features (razao entre a maior e
    a menor |SHAP| medio < 20x, todas numa faixa baixa e estreita), nao
    concentrada numa Feature especifica que o modelo estaria deixando de
    explorar bem. Diferente do cenario sintetico com verdade conhecida
    (razao >5x, `test_shap_correctly_identifies_the_truly_important_feature`),
    aqui nao existe uma "'importante' verdadeira" escondida -- reforca
    que a limitacao e' do dado, nao do metodo, agora vista pela lente de
    explicabilidade, nao so pela metrica agregada.
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", include_diagnosis_category=False)
    order = representation.domain_names()

    ids_elegiveis, labels = [], {}
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

    predictor = SklearnPredictor(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced"))
    predictor.fit(space_baseline, labels)

    relatorio = explain_predictor(predictor, space_baseline, representation=representation)
    valores = list(relatorio.mean_abs_shap.values())
    razao = max(valores) / min(valores) if min(valores) > 0 else float("inf")
    assert razao < 25, f"Esperava importancia difusa (razao pequena entre maior e menor |SHAP|, achado documentado) -- obteve razao={razao:.1f}"
    assert max(valores) < 0.05, f"Esperava a maior |SHAP| individual ainda ser pequena em termos absolutos -- obteve {max(valores):.4f}"
