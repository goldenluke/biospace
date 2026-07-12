"""
tests.test_geometric_neighborhood
=====================================

`Geometry.neighborhood(space, system_id, radius)` (§7.6 da teoria
formal) existia sem NENHUM teste antes desta rodada — achado numa
auditoria do projeto, mesmo padrão de `prediction/`, `risk/`,
`latent/`. Testado aqui com pontos em posições geométricas CONHECIDAS
(distância euclidiana calculável à mão) antes de qualquer aplicação
real.
"""

from __future__ import annotations

from datetime import datetime

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.geometry import Euclidean


def _vetor(system_id: str, x: float, y: float) -> RepresentationVector:
    comps = {"d": [Feature(name="x", value=x, raw_value=x), Feature(name="y", value=y, raw_value=y)]}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def _space_em_linha():
    """5 pontos em linha reta: origem (0,0) e pontos a distancia 1,2,3,4 dela -- distancias euclidianas exatas, calculaveis a mao."""
    space = RepresentationSpace(domain_order=["d"])
    space.add(_vetor("origem", 0.0, 0.0))
    for i in range(1, 5):
        space.add(_vetor(f"p{i}", float(i), 0.0))
    return space


def test_neighborhood_includes_only_points_within_exact_radius():
    """TESTE DECISIVO: com pontos a distancia exata 1,2,3,4 da origem, raio=2.5 deveria incluir p1,p2 e excluir p3,p4 -- fronteira exata, nao aproximada."""
    space = _space_em_linha()
    geometria = Euclidean()
    vizinhanca = geometria.neighborhood(space, "origem", radius=2.5)
    assert set(vizinhanca) == {"p1", "p2"}


def test_neighborhood_excludes_the_query_point_itself():
    space = _space_em_linha()
    geometria = Euclidean()
    vizinhanca = geometria.neighborhood(space, "origem", radius=10.0)
    assert "origem" not in vizinhanca


def test_neighborhood_boundary_is_inclusive():
    """d(x,y) <= radius (inclusivo, nao estrito) -- um ponto EXATAMENTE no raio deveria ser incluido."""
    space = _space_em_linha()
    geometria = Euclidean()
    vizinhanca = geometria.neighborhood(space, "origem", radius=2.0)
    assert "p2" in vizinhanca, "p2 esta a distancia EXATA 2.0 -- deveria ser incluido com raio=2.0 (inclusivo)."
    assert "p3" not in vizinhanca


def test_zero_radius_returns_empty_neighborhood():
    space = _space_em_linha()
    geometria = Euclidean()
    vizinhanca = geometria.neighborhood(space, "origem", radius=0.0)
    assert vizinhanca == []


def test_large_enough_radius_includes_everyone_else():
    space = _space_em_linha()
    geometria = Euclidean()
    vizinhanca = geometria.neighborhood(space, "origem", radius=100.0)
    assert set(vizinhanca) == {"p1", "p2", "p3", "p4"}


def test_neighborhood_is_symmetric_for_euclidean_distance():
    """Para uma metrica de verdade (simetrica), y esta na vizinhanca de x SSE x esta na vizinhanca de y, com o mesmo raio."""
    space = _space_em_linha()
    geometria = Euclidean()
    assert "p1" in geometria.neighborhood(space, "origem", radius=1.0)
    assert "origem" in geometria.neighborhood(space, "p1", radius=1.0)


def test_k_nearest_returns_closest_points_in_correct_order():
    """TESTE DECISIVO: k=2 mais proximos da origem, com pontos a distancia 1,2,3,4, deveria ser exatamente [p1,p2], NESSA ORDEM (distancia crescente)."""
    space = _space_em_linha()
    geometria = Euclidean()
    resultado = geometria.k_nearest(space, "origem", k=2)
    assert resultado == ["p1", "p2"]


def test_k_nearest_excludes_query_point_itself():
    space = _space_em_linha()
    geometria = Euclidean()
    resultado = geometria.k_nearest(space, "origem", k=4)
    assert "origem" not in resultado


def test_k_nearest_truncates_gracefully_when_k_exceeds_population():
    """k=10 pedido, so 4 outros pontos existem -- deveria devolver os 4, sem erro."""
    space = _space_em_linha()
    geometria = Euclidean()
    resultado = geometria.k_nearest(space, "origem", k=10)
    assert set(resultado) == {"p1", "p2", "p3", "p4"}
    assert len(resultado) == 4


def test_k_nearest_and_neighborhood_agree_at_matching_radius():
    """k_nearest(k=2) e neighborhood(radius=distancia_do_2o_vizinho) deveriam concordar -- as duas consultas descrevendo o mesmo conjunto por caminhos diferentes."""
    space = _space_em_linha()
    geometria = Euclidean()
    k_resultado = set(geometria.k_nearest(space, "origem", k=2))
    raio_resultado = set(geometria.neighborhood(space, "origem", radius=2.0))
    assert k_resultado == raio_resultado == {"p1", "p2"}
