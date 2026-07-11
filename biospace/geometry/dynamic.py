"""
biospace.geometry.dynamic
============================

DynamicGeometry: d(x, y, t) — a MÉTRICA em si depende de `t` (não
necessariamente tempo cronológico; aqui, ESTÁGIO da doença — qual
fenótipo/região do espaço serve de referência). Isso é distinto de
Geometry (métrica fixa, d(x,y) sempre a mesma fórmula em qualquer lugar
do espaço) — por isso uma interface irmã, não uma extensão de Geometry
(mesma lógica que já separou Geometry de TrajectoryGeometry).

PhenotypeConditionedGeometry: implementação concreta — "cada doença
aprende sua própria geometria" na prática mais direta possível: ajusta
uma covariância (Mahalanobis) LOCAL para cada fenótipo já estimado
(ClinicalKMeansPhenotyper, por exemplo), usando SÓ os pacientes daquele
fenótipo. `t` escolhe qual covariância local usar — um nome de fenótipo
diretamente, ou um ponto de referência (do qual o fenótipo é inferido via
`Phenotype.contains`).

MOTIVAÇÃO FISIOLÓGICA: a mesma diferença absoluta entre dois pacientes
pode ser mais ou menos "significativa" dependendo de ONDE no espectro de
severidade eles estão — variabilidade entre pacientes leves pode ter uma
estrutura de covariância bem diferente da variabilidade entre pacientes
graves. Uma métrica ÚNICA (Euclidean/Mahalanobis global) trata as duas
situações da mesma forma; `PhenotypeConditionedGeometry` não.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from biospace.core import Phenotype, RepresentationSpace

__all__ = ["DynamicGeometry", "PhenotypeConditionedGeometry"]


class DynamicGeometry(ABC):
    """Interface para métricas cujo tensor depende de um estado `t` (estágio de doença, não necessariamente tempo cronológico)."""

    name: str = "dynamic_geometry"

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray, t) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PhenotypeConditionedGeometry(DynamicGeometry):
    """
    Ver docstring do módulo. `t` pode ser o NOME de um fenótipo (str) ou um `np.ndarray` (o fenótipo é inferido).

    AVISO NUMÉRICO IMPORTANTE: com poucos membros por fenótipo relativo à
    dimensionalidade do espaço (n < p, comum aqui — 52 dimensões, alguns
    fenótipos com poucas dezenas de pacientes), a covariância amostral
    "crua" é necessariamente singular ou quase-singular (rank <= n-1).
    Testado nos dados reais: um fenótipo com n=40 (< p=52) produziu uma
    distância absurda (1288, vs. 14-16 dos demais) com regularização
    ingênua (epsilon na diagonal) — artefato numérico, não sinal
    fisiológico. Por isso este operador usa ENCOLHIMENTO DE COVARIÂNCIA
    (Ledoit-Wolf, 2004 — `sklearn.covariance.LedoitWolf`), que mistura a
    covariância local com a identidade (escalada), na proporção correta
    para o regime n < p, em vez de uma regularização ad-hoc.
    """

    name = "phenotype_conditioned"

    def __init__(self):
        self.phenotypes: list["Phenotype"] = []
        self.order: Optional[Sequence[str]] = None
        self._inv_cov: dict[str, np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._n_members: dict[str, int] = {}
        self._shrinkage: dict[str, float] = {}
        self.is_fitted = False

    def fit(
        self, space: "RepresentationSpace", phenotypes: Sequence["Phenotype"], order: Optional[Sequence[str]] = None
    ) -> "PhenotypeConditionedGeometry":
        """Ajusta uma covariância (Mahalanobis) LOCAL para cada fenótipo, com encolhimento Ledoit-Wolf."""
        from sklearn.covariance import LedoitWolf

        self.phenotypes = list(phenotypes)
        self.order = order or space.order()

        ids = space.ids()
        vectors = {sid: space.get(sid).as_vector(self.order) for sid in ids}

        for ph in self.phenotypes:
            members = [v for v in vectors.values() if ph.contains(v)]
            self._n_members[ph.name] = len(members)
            dim = len(next(iter(vectors.values())))
            if len(members) < 2:
                self._mean[ph.name] = np.zeros(dim)
                self._inv_cov[ph.name] = np.eye(dim)
                self._shrinkage[ph.name] = 1.0
                continue
            stacked = np.stack(members)
            lw = LedoitWolf().fit(stacked)
            self._mean[ph.name] = lw.location_
            self._inv_cov[ph.name] = np.linalg.pinv(lw.covariance_)
            self._shrinkage[ph.name] = float(lw.shrinkage_)

        self.is_fitted = True
        return self

    def _resolve_phenotype_name(self, t: Union[str, np.ndarray]) -> str:
        if isinstance(t, str):
            if t not in self._inv_cov:
                raise KeyError(f"Fenótipo '{t}' não foi ajustado. Fenótipos disponíveis: {list(self._inv_cov.keys())}")
            return t
        for ph in self.phenotypes:
            if ph.contains(t):
                return ph.name
        raise ValueError(
            "Não foi possível inferir o fenótipo do ponto de referência `t` — ele não cai em nenhum "
            "fenótipo ajustado. Informe o nome do fenótipo diretamente."
        )

    def distance(self, x: np.ndarray, y: np.ndarray, t: Union[str, np.ndarray]) -> float:
        if not self.is_fitted:
            raise RuntimeError("PhenotypeConditionedGeometry.fit(space, phenotypes) deve ser chamado antes de distance().")
        phenotype_name = self._resolve_phenotype_name(t)
        inv_cov = self._inv_cov[phenotype_name]
        diff = x - y
        return float(np.sqrt(max(diff @ inv_cov @ diff.T, 0.0)))

    def covariance_summary(self) -> dict[str, dict[str, float]]:
        """Resumo da covariância local por fenótipo (nº de membros, encolhimento aplicado, traço/log-det do inverso) — para auditar a confiabilidade de cada fenótipo."""
        summary = {}
        for name, inv_cov in self._inv_cov.items():
            summary[name] = {
                "n_membros": self._n_members[name],
                "shrinkage": self._shrinkage.get(name, float("nan")),
                "traco_inv_cov": float(np.trace(inv_cov)),
                "log_det_inv_cov": float(np.linalg.slogdet(inv_cov)[1]),
            }
        return summary

    def __repr__(self) -> str:
        status = f"{len(self.phenotypes)} fenótipos ajustados" if self.is_fitted else "não ajustada"
        return f"PhenotypeConditionedGeometry({status})"
