"""
biospace.explainability.shap_explainer
==========================================

Explica um `SklearnPredictor` já ajustado via SHAP (SHapley Additive
exPlanations) — não reimplementa nada estatístico, envelopa `shap`
(`TreeExplainer` para modelos de árvore, que é exato e rápido; cai
para `KernelExplainer`, aproximado e mais lento, para qualquer outro
estimador) e traduz o resultado de índice numérico de coluna para
`domínio.feature` legível, usando a mesma convenção de nomes já usada
em todo o resto do projeto.

Por que isso importa mais que "só rodar o SHAP": um AUC baixo (achado
já documentado em `biospace.prediction` — predição prospectiva de
readmissão na UCI fica perto do acaso) não diz SOZINHO se é porque
nenhuma Feature carrega sinal, ou se há sinal em alguma Feature
específica que o modelo não está conseguindo usar bem. SHAP responde
essa segunda pergunta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from biospace.core import Representation, RepresentationSpace
    from biospace.prediction import SklearnPredictor

__all__ = ["ShapExplanationReport", "explain_predictor"]


@dataclass
class ShapExplanationReport:
    feature_names: list[str]
    mean_abs_shap: dict[str, float]
    shap_values: np.ndarray  # (n_amostras, n_features) -- ja reduzido a classe positiva se binario
    explainer_type: str

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """As `n` Features com maior |SHAP| médio -- as que mais influenciam a predição do modelo, na direção que for."""
        return sorted(self.mean_abs_shap.items(), key=lambda kv: -kv[1])[:n]

    def summary(self) -> str:
        linhas = [f"ShapExplanationReport ({self.explainer_type}, {len(self.feature_names)} Features):"]
        for nome, valor in self.top_features(5):
            linhas.append(f"  {nome}: |SHAP| médio={valor:.4f}")
        return "\n".join(linhas)


def _feature_names_in_matrix_order(representation: "Representation") -> list[str]:
    """Reconstrói os nomes 'domínio.feature' na MESMA ordem que `RepresentationSpace.matrix()` produz colunas -- convenção já usada manualmente em todo o resto do projeto, agora centralizada aqui."""
    order = representation.domain_names()
    mapa_dominios = {d.name: d for d in representation.domains}
    nomes = []
    for nome_dominio in order:
        dominio = mapa_dominios[nome_dominio]
        for observable in dominio.observables:
            nomes.append(f"{nome_dominio}.{observable.key}")
    return nomes


def explain_predictor(
    predictor: "SklearnPredictor",
    space: "RepresentationSpace",
    representation: Optional["Representation"] = None,
    feature_names: Optional[list[str]] = None,
    background_size: int = 100,
    random_state: int = 0,
) -> ShapExplanationReport:
    """
    Explica as predições de `predictor` (já ajustado — `.fit()` deve
    ter sido chamado antes) sobre `space`, via SHAP.

    `feature_names`: se não passado, reconstruído de `representation`
    (obrigatório passar um dos dois). `background_size`: para
    `KernelExplainer` (modelos não-árvore), o SHAP precisa de uma
    amostra de referência menor que a população inteira por custo
    computacional — irrelevante para `TreeExplainer` (exato, não usa
    amostra de fundo).
    """
    import shap

    if feature_names is None:
        if representation is None:
            raise ValueError("Passe `feature_names` ou `representation` -- não há como nomear as colunas sem um dos dois.")
        feature_names = _feature_names_in_matrix_order(representation)

    matrix, ids = space.matrix()
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Número de Features na matriz ({matrix.shape[1]}) não bate com `feature_names` ({len(feature_names)}) "
            "-- confira se `representation` corresponde à mesma Representation usada para construir `space`."
        )

    estimator = predictor.estimator
    eh_arvore = estimator.__class__.__name__ in {
        "RandomForestClassifier", "RandomForestRegressor", "GradientBoostingClassifier",
        "GradientBoostingRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "DecisionTreeClassifier",
    }

    if eh_arvore:
        explainer = shap.TreeExplainer(estimator)
        valores_brutos = explainer.shap_values(matrix)
        tipo = "TreeExplainer"
    else:
        rng = np.random.default_rng(random_state)
        idx_fundo = rng.choice(len(matrix), size=min(background_size, len(matrix)), replace=False)
        explainer = shap.KernelExplainer(estimator.predict_proba, matrix[idx_fundo])
        valores_brutos = explainer.shap_values(matrix, silent=True)
        tipo = "KernelExplainer"

    valores = np.asarray(valores_brutos)
    if valores.ndim == 3:
        # (n_amostras, n_features, n_classes) -- reduzir a classe positiva (indice 1) se binario
        valores = valores[:, :, 1] if valores.shape[2] == 2 else valores[:, :, 0]
    elif isinstance(valores_brutos, list):
        valores = np.asarray(valores_brutos[1] if len(valores_brutos) == 2 else valores_brutos[0])

    media_abs = np.abs(valores).mean(axis=0)
    mean_abs_shap = {nome: float(v) for nome, v in zip(feature_names, media_abs)}

    return ShapExplanationReport(feature_names=feature_names, mean_abs_shap=mean_abs_shap, shap_values=valores, explainer_type=tipo)
