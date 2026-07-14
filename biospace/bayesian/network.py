"""
biospace.bayesian.network
=============================

Rede Bayesiana (grafo acíclico dirigido + distribuições de
probabilidade condicional) sobre um `RepresentationSpace` — envelope
sobre `pgmpy`: aprendizado de estrutura (Hill-Climb Search, score
BIC) + estimação de parâmetros (máxima verossimilhança) + inferência
por eliminação de variáveis.

Diferente de `biospace.causal` (que testa balanceamento e ajusta
associação observacional, mas nunca afirma causalidade), uma rede
bayesiana aprendida por busca de estrutura encontra uma
DEPENDÊNCIA estatística compatível com os dados — não uma
estrutura causal verdadeira, apenas uma das estruturas
estatisticamente equivalentes (mesma classe de equivalência de
Markov) que explicam a mesma distribuição igualmente bem. Tratamos
a estrutura aprendida como hipótese de dependência a ser investigada,
nunca como grafo causal confirmado — a mesma disciplina que
`biospace.causal` já aplica, com um método diferente.

pgmpy exige variáveis discretas; `discretize()` converte cada
característica contínua em faixas por quantil, documentado
explicitamente (não escondido) porque perde informação de
granularidade fina.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["BayesianNetworkOperator", "discretize"]


def discretize(matrix: np.ndarray, feature_names: list[str], n_bins: int = 4) -> pd.DataFrame:
    """Discretiza cada coluna por quantil (bins de tamanho aproximadamente igual) -- perde granularidade fina, documentado, nao escondido."""
    df = pd.DataFrame(matrix, columns=feature_names)
    df_discreto = pd.DataFrame(index=df.index)
    for col in df.columns:
        try:
            df_discreto[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            df_discreto[col] = 0
    return df_discreto.astype(str)


@dataclass
class BayesianNetworkOperator:
    n_bins: int = 4
    max_indegree: Optional[int] = 3
    model = None
    feature_names: Optional[list] = None

    def fit(self, space: "RepresentationSpace", feature_names: list[str]) -> "BayesianNetworkOperator":
        from pgmpy.estimators import HillClimbSearch
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.parameter_estimator import DiscreteMLE

        matrix, ids = space.matrix()
        self.feature_names = feature_names
        df_discreto = discretize(matrix, feature_names, n_bins=self.n_bins)

        buscador = HillClimbSearch(df_discreto)
        estrutura = buscador.estimate(scoring_method="bic-d", max_indegree=self.max_indegree, show_progress=False)

        self.model = DiscreteBayesianNetwork(estrutura.edges())
        self.model.add_nodes_from(df_discreto.columns)
        self.model.fit(df_discreto, estimator=DiscreteMLE())
        return self

    def edges(self) -> list[tuple[str, str]]:
        if self.model is None:
            raise RuntimeError("Chame fit() antes de edges().")
        return list(self.model.edges())

    def query(self, target: str, evidence: dict[str, str]) -> dict:
        """Distribuicao de probabilidade de `target` dado `evidence` (ex.: {'idade': '3'}) -- valores discretizados, nao brutos."""
        if self.model is None:
            raise RuntimeError("Chame fit() antes de query().")
        from pgmpy.inference import VariableElimination

        infer = VariableElimination(self.model)
        resultado = infer.query(variables=[target], evidence=evidence, show_progress=False)
        return dict(zip(resultado.state_names[target], resultado.values))
