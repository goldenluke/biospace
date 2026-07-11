"""
biospace.geometry.learned
============================

LearnedGeometry: aprende uma transformação linear do espaço de
representação que maximiza a separação entre classes rotuladas, via
Neighbourhood Components Analysis (NCA, Goldberger et al. 2004) — usando
a implementação de produção do scikit-learn, não uma reimplementação
própria (metric learning de verdade tem detalhes de otimização não
triviais; reaproveitar código testado é a escolha mais honesta aqui).

Depois de ajustada, a distância é a Euclidiana no espaço TRANSFORMADO —
equivalente a uma Mahalanobis aprendida a partir de RÓTULOS, não de
estatística populacional (`Mahalanobis`, que usa covariância) nem de uma
fórmula fixa (`Euclidean`).

AVISO DE CIRCULARIDADE — leia antes de usar: esta geometria só é tão
significativa quanto os rótulos usados para ajustá-la. Se os rótulos
vierem do PRÓPRIO `ClinicalKMeansPhenotyper` (fenótipos estimados sobre
esta mesma Representation), a geometria aprendida está, em certo
sentido, "aprendendo a enxergar melhor o que o K-Means já viu" — útil
para INTERPRETAR quais eixos mais discriminam os fenótipos já
encontrados (uma espécie de importância de feature), mas não é uma
validação independente deles. Prefira rótulos genuinamente
independentes da Representation (ex.: `classificar_apneia()`, que vem
direto do IDO bruto, não da clusterização) quando o objetivo for validar
a geometria, não apenas interpretar um agrupamento já feito.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import Geometry

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["LearnedGeometry"]


class LearnedGeometry(Geometry):
    name = "learned"

    def __init__(self, n_components: Optional[int] = None, random_state: int = 42):
        """`n_components`: dimensão do espaço transformado (None = mesma dimensão do espaço original)."""
        self.n_components = n_components
        self.random_state = random_state
        self._nca = None
        self.is_fitted = False
        self.classes_: list[str] = []

    def fit(self, space: "RepresentationSpace", labels: dict[str, str]) -> "LearnedGeometry":
        """
        Ajusta a transformação NCA usando `labels` (dict system_id ->
        rótulo) sobre os sistemas presentes em `space`. Requer pelo menos
        2 classes rotuladas e mais de 1 exemplo por classe (limitação do
        próprio NCA).
        """
        from sklearn.neighbors import NeighborhoodComponentsAnalysis

        matrix, ids = space.matrix()
        missing = [sid for sid in ids if sid not in labels]
        if missing:
            raise KeyError(f"Faltam rótulos para {len(missing)} sistema(s), ex.: {missing[:5]}")

        y = [labels[sid] for sid in ids]
        self.classes_ = sorted(set(y))

        self._nca = NeighborhoodComponentsAnalysis(n_components=self.n_components, random_state=self.random_state)
        self._nca.fit(matrix, y)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Projeta um ponto bruto de X no espaço aprendido (útil para visualização/diagnóstico)."""
        if not self.is_fitted:
            raise RuntimeError("LearnedGeometry.fit(space, labels) deve ser chamado antes de transform().")
        return self._nca.transform(x.reshape(1, -1))[0]

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise RuntimeError("LearnedGeometry.fit(space, labels) deve ser chamado antes de distance().")
        x_t = self.transform(x)
        y_t = self.transform(y)
        return float(np.linalg.norm(x_t - y_t))

    def describe(self) -> str:
        status = f"ajustada em {len(self.classes_)} classes" if self.is_fitted else "não ajustada"
        return f"LearnedGeometry(NCA, {status})"
