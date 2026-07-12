"""
biospace.geometry.cohort_query
=================================

O item "coortes automáticas" da teoria: coortes deixam de ser consultas
SQL (limiares arbitrários sobre uma Feature de cada vez) e passam a ser
SUBCONJUNTOS GEOMÉTRICOS do espaço de representação — proximidade a um
ponto de consulta, não uma conjunção de condições discretas.

`Geometry.k_nearest()`/`Geometry.neighborhood()` (núcleo) já cobrem a
consulta quando o ponto de referência é um paciente EXISTENTE na
Cohort. O que faltava, e que de fato realiza "critérios contínuos" —
não apenas "o paciente mais parecido com outro paciente" — é consultar
por um ponto ARBITRÁRIO (ex.: o centroide de um cluster, um perfil
clínico hipotético): não precisa corresponder a nenhum paciente real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace
    from biospace.geometry import Geometry

__all__ = ["GeometricCohort", "cohort_around"]


@dataclass
class GeometricCohort:
    """Um subconjunto do espaço de representação definido por proximidade geométrica, não por limiares de Feature."""

    member_ids: list[str]
    query_description: str
    radius: Optional[float] = None
    k: Optional[int] = None

    def __len__(self) -> int:
        return len(self.member_ids)

    def overlap_with(self, other_ids: set[str]) -> dict[str, float]:
        """
        Compara esta coorte geométrica contra outro conjunto de ids
        (tipicamente uma coorte definida por limiar/SQL) — Jaccard e as
        duas frações de sobreposição assimétricas, não só uma métrica
        agregada que esconde qual lado domina.
        """
        membros = set(self.member_ids)
        intersecao = membros & other_ids
        uniao = membros | other_ids
        return {
            "n_geometrica": len(membros),
            "n_outra": len(other_ids),
            "n_intersecao": len(intersecao),
            "jaccard": len(intersecao) / len(uniao) if uniao else 0.0,
            "fracao_da_geometrica_tambem_na_outra": len(intersecao) / len(membros) if membros else 0.0,
            "fracao_da_outra_tambem_na_geometrica": len(intersecao) / len(other_ids) if other_ids else 0.0,
        }


def cohort_around(
    space: "RepresentationSpace",
    geometry: "Geometry",
    query: Union[str, np.ndarray],
    order: Optional[list[str]] = None,
    k: Optional[int] = None,
    radius: Optional[float] = None,
) -> GeometricCohort:
    """
    `query`: um system_id EXISTENTE na `space` (proximidade a um
    paciente real) OU um vetor arbitrário já na mesma dimensão de `order`
    (proximidade a um ponto hipotético — ex.: centroide de fenótipo,
    perfil clínico prototípico). Exatamente um de `k`/`radius` deve ser
    passado — misturar os dois critérios não tem uma semântica única
    (qual prevalece?), então é rejeitado explicitamente, não resolvido
    silenciosamente com uma prioridade arbitrária.
    """
    if (k is None) == (radius is None):
        raise ValueError("Passe exatamente um de `k` ou `radius`, não os dois nem nenhum -- misturar os dois critérios não tem semântica única.")

    order = order or space.order()

    if isinstance(query, str):
        ref = space.get(query).as_vector(order)
        excluir = {query}
        descricao = f"em torno do paciente {query!r}"
    else:
        ref = np.asarray(query, dtype=float)
        excluir = set()
        descricao = "em torno de um ponto de consulta arbitrário (não corresponde a nenhum paciente real)"

    distancias = sorted(
        (geometry.distance(ref, space.get(sid).as_vector(order)), sid) for sid in space.ids() if sid not in excluir
    )

    if k is not None:
        membros = [sid for _, sid in distancias[:k]]
        descricao += f", k={k} mais próximos"
    else:
        membros = [sid for d, sid in distancias if d <= radius]
        descricao += f", raio={radius}"

    return GeometricCohort(member_ids=membros, query_description=descricao, radius=radius, k=k)
