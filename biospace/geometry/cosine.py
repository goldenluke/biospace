"""
biospace.geometry.cosine
===========================

Distância de cosseno: 1 - similaridade de cosseno entre dois pontos do
espaço de representação. Diferente de Euclidean/Mahalanobis, ignora a
MAGNITUDE dos vetores e compara apenas a DIREÇÃO — útil quando o que
importa é o "perfil relativo" entre eixos (ex.: dois pacientes com
severidade absoluta muito diferente, mas o mesmo padrão relativo entre
domínios), não a distância física entre os estados.

AVISO DE RIGOR: a distância de cosseno convencional (1 - similaridade)
NÃO é uma métrica no sentido formal — ela viola a desigualdade
triangular em geral (só a "distância angular", arccos(similaridade)/π, é
uma métrica de verdade). Implementamos a convenção usual de ML/ciência
de dados (a mesma que `scipy.spatial.distance.cosine`), não a variante
angular, por ser a mais reconhecível — mas isso significa que
`check_lipschitz_continuity`/outros contratos que assumem uma métrica
formal podem não valer estritamente para esta geometria.
"""

from __future__ import annotations

import numpy as np

from .base import Geometry

__all__ = ["Cosine"]


class Cosine(Geometry):
    name = "cosine"

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        norm_x = float(np.linalg.norm(x))
        norm_y = float(np.linalg.norm(y))
        if norm_x < 1e-12 or norm_y < 1e-12:
            # Convenção: um vetor nulo é tratado como "sem direção definida",
            # portanto maximamente diferente de qualquer vetor não-nulo —
            # em vez de retornar NaN/erro silenciosamente.
            return 1.0 if (norm_x > 1e-12 or norm_y > 1e-12) else 0.0
        cosine_similarity = float(np.dot(x, y) / (norm_x * norm_y))
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))  # robustez a erro de ponto flutuante
        return 1.0 - cosine_similarity
