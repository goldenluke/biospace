"""
biospace.latent.factor_analysis
==================================

FactorAnalysisLatentDomain: implementação concreta e genérica de
LatentDomain via Análise Fatorial (sklearn.decomposition.FactorAnalysis)
— um modelo estatístico de variável latente de verdade (observado =
matriz_de_cargas @ fator_latente + ruído), não uma combinação linear
arbitrária "vestida" de inferência.

O fator extraído é o eixo de variação COOMPARTILHADA entre os domínios-
fonte — uma reconstrução estatística legítima de "algo em comum" que
essas fontes carregam. A INTERPRETAÇÃO CLÍNICA desse fator (que nome dar
a ele, o que ele "significa") é uma HIPÓTESE, declarada explicitamente
via `hypothesis` (herdado de LatentDomain) — nunca um fato validado só
porque o método estatístico é rigoroso.

IMPORTANTE: diferente dos demais domínios (que são stateless e
determinísticos por sistema), este domínio precisa ser AJUSTADO
(`fit()`) sobre uma população antes de poder transformar qualquer
sistema individual — análogo ao `Reference` de `_ReferenceDomain` no
plugin sleep, mas aqui o parâmetro ajustado é a matriz de cargas
fatoriais, não médias/desvios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from biospace.core import Feature, LatentDomain, SemanticDomain

if TYPE_CHECKING:
    from biospace.core import Cohort

__all__ = ["FactorAnalysisLatentDomain"]


class FactorAnalysisLatentDomain(LatentDomain):
    """
    Ver docstring do módulo. Subclasses concretas devem definir `name` e
    `hypothesis` (obrigatório — ver `LatentDomain`); `n_factors` e
    `random_state` controlam o ajuste.
    """

    n_factors: int = 1

    def __init__(self, source_domains: Sequence[SemanticDomain], random_state: int = 42):
        super().__init__(source_domains)
        self.random_state = random_state
        self._model = None
        self._feature_order: list[str] = []
        self.is_fitted = False
        self.loadings_: Optional[dict[str, np.ndarray]] = None

    def _flatten(self, source_features: dict[str, list[Feature]]) -> tuple[np.ndarray, list[str]]:
        """Achata as Features de todos os domínios-fonte em um vetor único, com nomes qualificados."""
        names: list[str] = []
        values: list[float] = []
        for domain_name in sorted(source_features.keys()):
            for f in source_features[domain_name]:
                names.append(f"{domain_name}.{f.name}")
                values.append(f.value)
        return np.array(values, dtype=float), names

    def fit(self, cohort: "Cohort") -> "FactorAnalysisLatentDomain":
        """
        Ajusta a Análise Fatorial sobre a população de uma Cohort (usa o
        estado mais recente de cada sistema). Necessário antes de
        `transform()`/`infer()` funcionarem.
        """
        from sklearn.decomposition import FactorAnalysis

        rows = []
        feature_order: Optional[list[str]] = None
        for system in cohort.systems.values():
            source_features = self.collect_source_features(system)
            row, names = self._flatten(source_features)
            if feature_order is None:
                feature_order = names
            elif names != feature_order:
                raise ValueError(
                    "Domínios-fonte produziram conjuntos de Features inconsistentes entre "
                    "sistemas — todos os sistemas da Cohort devem gerar as mesmas Features "
                    "nos domínios-fonte para ajustar a Análise Fatorial."
                )
            rows.append(row)

        X = np.stack(rows)
        self._feature_order = feature_order or []
        self._model = FactorAnalysis(n_components=self.n_factors, random_state=self.random_state)
        self._model.fit(X)
        self.loadings_ = {
            name: self._model.components_[:, i] for i, name in enumerate(self._feature_order)
        }
        self.is_fitted = True
        return self

    def top_loadings(self, factor_index: int = 0, n: int = 5) -> list[tuple[str, float]]:
        """As `n` Features com maior carga (em valor absoluto) no fator `factor_index` — para inspeção/auditoria."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}.fit(cohort) deve ser chamado antes de inspecionar cargas.")
        pairs = [(name, float(loadings[factor_index])) for name, loadings in self.loadings_.items()]
        return sorted(pairs, key=lambda p: -abs(p[1]))[:n]

    def infer(self, source_features: dict[str, list[Feature]]) -> list[Feature]:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name}.fit(cohort) deve ser chamado antes de transform()/infer() — "
                "a Análise Fatorial precisa ser ajustada sobre uma população primeiro."
            )
        row, names = self._flatten(source_features)
        if names != self._feature_order:
            raise ValueError(
                "As Features dos domínios-fonte para este sistema não correspondem às usadas "
                "no ajuste (fit) — verifique se os mesmos domínios-fonte, na mesma configuração, "
                "estão sendo usados."
            )
        scores = self._model.transform(row.reshape(1, -1))[0]
        provenance = tuple(names)
        return [
            Feature(name=f"factor_{i + 1}", value=float(score), raw_value=None, weight=1.0, provenance=provenance)
            for i, score in enumerate(scores)
        ]
