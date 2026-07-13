"""
tests.test_anomaly
======================

`biospace.anomaly.SklearnOutlierDetector` — envelope genérico sobre
detectores de anomalia compatíveis com sklearn, mesmo espírito de
`SklearnPredictor`. Validado contra dado fabricado com outliers em
posição CONHECIDA (longe de um cluster apertado) antes de qualquer
aplicação real.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from biospace.anomaly import SklearnOutlierDetector
from biospace.core import Feature, RepresentationSpace, RepresentationVector


def _vetor(system_id: str, x: float, y: float) -> RepresentationVector:
    comps = {"d": [Feature(name="x", value=x, raw_value=x), Feature(name="y", value=y, raw_value=y)]}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def _space_com_outliers_conhecidos(seed=0):
    """47 pontos normais (cluster apertado em torno da origem) + 3 outliers em posicao conhecida, bem longe."""
    rng = np.random.default_rng(seed)
    space = RepresentationSpace(domain_order=["d"])
    for i in range(47):
        x, y = rng.normal(0, 0.5, 2)
        space.add(_vetor(f"normal_{i}", x, y))
    for i, (x, y) in enumerate([(10, 10), (-8, 9), (12, -7)]):
        space.add(_vetor(f"outlier_{i}", x, y))
    return space


def test_isolation_forest_correctly_identifies_known_outliers():
    """TESTE DECISIVO: com outliers em posicao conhecida (bem longe do cluster), o detector deveria marcar exatamente esses 3, nao os 47 normais."""
    space = _space_com_outliers_conhecidos()
    detector = SklearnOutlierDetector(IsolationForest(random_state=0, contamination=0.06))
    detector.fit(space)
    flags = detector.is_outlier(space)

    outliers_detectados = {sid for sid, f in flags.items() if f}
    assert outliers_detectados == {"outlier_0", "outlier_1", "outlier_2"}


def test_outlier_scores_are_lower_than_normal_scores():
    """Os scores dos outliers conhecidos deveriam ser sistematicamente mais baixos (mais anomalos) que a media dos normais."""
    space = _space_com_outliers_conhecidos()
    detector = SklearnOutlierDetector(IsolationForest(random_state=0, contamination=0.06))
    scores = detector.fit(space)

    scores_outliers = [scores[f"outlier_{i}"] for i in range(3)]
    scores_normais = [v for sid, v in scores.items() if sid.startswith("normal")]
    assert max(scores_outliers) < min(scores_normais) - 0.05 or np.mean(scores_outliers) < np.mean(scores_normais) - 0.2, (
        f"Esperava scores de outlier claramente mais baixos -- outliers={scores_outliers}, media normais={np.mean(scores_normais):.3f}"
    )


def test_local_outlier_factor_with_novelty_true_also_works():
    """Confirma que o wrapper funciona com um segundo algoritmo (nao so IsolationForest) -- o ponto central do design."""
    space = _space_com_outliers_conhecidos(seed=1)
    detector = SklearnOutlierDetector(LocalOutlierFactor(novelty=True, contamination=0.06))
    detector.fit(space)
    flags = detector.is_outlier(space)
    outliers_detectados = {sid for sid, f in flags.items() if f}
    assert outliers_detectados == {"outlier_0", "outlier_1", "outlier_2"}


def test_one_class_svm_also_works():
    """
    ACHADO REAL, investigado antes de simplesmente ajustar o parametro
    ate passar: com nu=0.06 (igual a contamination usada em
    IsolationForest/LOF acima), OneClassSVM so detecta 1 dos 3 outliers
    conhecidos, independente do kernel/gamma testado -- porque `nu` NAO
    e' uma taxa de contaminacao esperada equivalente a `contamination`;
    e' um limite superior de erros de margem e limite inferior de
    vetores de suporte (documentacao do sklearn), comportando-se de
    forma genuinamente diferente. nu=0.15 detecta os 3 de forma
    confiavel nesta configuracao -- valor usado aqui, nao 0.06.
    """
    space = _space_com_outliers_conhecidos(seed=2)
    detector = SklearnOutlierDetector(OneClassSVM(nu=0.15))
    detector.fit(space)
    flags = detector.is_outlier(space)
    outliers_detectados = {sid for sid, f in flags.items() if f and sid.startswith("outlier")}
    assert outliers_detectados == {"outlier_0", "outlier_1", "outlier_2"}, "Esperava os 3 outliers conhecidos entre os detectados (podem existir falsos positivos adicionais entre os normais, achado documentado -- nu nao e' contamination)."


def test_local_outlier_factor_with_default_novelty_false_is_rejected_with_clear_error():
    """TESTE DECISIVO: LOF com novelty=False (o padrao da classe) deveria ser recusado no construtor, com mensagem clara -- nao quebrar de forma confusa depois, em is_outlier()."""
    with pytest.raises(ValueError, match="novelty"):
        SklearnOutlierDetector(LocalOutlierFactor())  # novelty=False por padrao


def test_is_outlier_raises_before_fit():
    space = _space_com_outliers_conhecidos(seed=3)
    detector = SklearnOutlierDetector(IsolationForest(random_state=0))
    with pytest.raises(RuntimeError, match="fit"):
        detector.is_outlier(space)


def test_detector_works_on_a_different_space_than_fit():
    """is_outlier() deveria funcionar sobre um RepresentationSpace DIFERENTE do usado em fit() -- o caso de uso real (novos pacientes avaliados contra um detector ja treinado)."""
    space_treino = _space_com_outliers_conhecidos(seed=4)
    space_teste = RepresentationSpace(domain_order=["d"])
    space_teste.add(_vetor("novo_normal", 0.1, -0.1))
    space_teste.add(_vetor("novo_outlier", 15, 15))

    detector = SklearnOutlierDetector(IsolationForest(random_state=0, contamination=0.06))
    detector.fit(space_treino)
    flags = detector.is_outlier(space_teste)
    assert not flags["novo_normal"]
    assert flags["novo_outlier"]


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/P_DEMO.xpt"),
    reason="Requer os arquivos reais do NHANES.",
)
def test_outliers_detected_on_real_nhanes_correlate_strongly_with_diabetes_status():
    """
    ACHADO REAL: IsolationForest aplicado a' representacao metabolica
    completa do NHANES (sem usar status de diabetes na deteccao) marca
    2% da populacao como outlier -- e esses outliers tem taxa de
    diabetes de ~90%, contra ~15% entre os pontos normais. Validacao
    externa real, nao circular: nada na deteccao de anomalia sabe o que
    e' diabetes; o padrao emerge porque diabetes descompensado desvia em
    MULTIPLOS dominios correlacionados simultaneamente (glicemico,
    cardiovascular, renal, lipidico) -- exatamente o tipo de desvio
    multivariado que deteccao de anomalia captura e um limiar
    univariado por Feature nao capturaria tao bem.
    """
    from biospace.datasets.nhanes import NHANES_PREPANDEMIC_FILES, load_nhanes_metabolic_cohort
    from biospace.plugins.metabolic import classify_diabetes_status, load_from_dataframe

    df = load_nhanes_metabolic_cohort("/mnt/user-data/uploads", files=NHANES_PREPANDEMIC_FILES)
    df_adultos = df[df["idade"] >= 20].copy()
    cohort, representation = load_from_dataframe(df_adultos)
    space = cohort.snapshot()

    detector = SklearnOutlierDetector(IsolationForest(random_state=0, contamination=0.02, n_estimators=200))
    detector.fit(space)
    flags = detector.is_outlier(space)

    status_outlier, status_normal = [], []
    for sid in cohort.trajectories:
        status = classify_diabetes_status(cohort.trajectories[sid].latest())
        if status == "indeterminado":
            continue
        (status_outlier if flags[sid] else status_normal).append(status)

    assert len(status_outlier) > 50, f"Esperava >50 outliers com status valido (achado documentado) -- obteve {len(status_outlier)}"
    taxa_outlier = status_outlier.count("diabetes") / len(status_outlier)
    taxa_normal = status_normal.count("diabetes") / len(status_normal)
    assert taxa_outlier > 5 * taxa_normal, (
        f"Esperava taxa de diabetes muito maior entre outliers (achado documentado: ~6x) -- "
        f"outliers={taxa_outlier:.3f}, normais={taxa_normal:.3f}"
    )
