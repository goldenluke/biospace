"""
tests.test_riemannian_and_composite
======================================

Duas coberturas que faltavam na suíte original:

  1. RiemannianGeometry — validada no cenário sintético clássico (espiral)
     que usamos ao construir o recurso: pontos em voltas ADJACENTES devem
     ter geodésica MAIOR que Euclidiana; pontos genuinamente vizinhos na
     mesma volta devem ter geodésica ≈ Euclidiana. Também testa o erro
     claro levantado quando o grafo fica desconectado.
  2. CompositeRepresentation — regressão: ao qualificar nomes de Feature
     por prefixo de domínio, o campo `uncertainty` não estava sendo
     copiado (esquecido na primeira versão) — corrigido.
"""

from __future__ import annotations

import numpy as np
import pytest

from biospace.core import CompositeRepresentation, Feature, RepresentationVector
from biospace.geometry import Euclidean, RiemannianGeometry


class _DomainStub:
    """Domínio mínimo, só para montar um RepresentationVector de teste sem depender do plugin sleep."""

    def __init__(self, name, features):
        self.name = name
        self._features = features

    def transform(self, system, as_of=None):
        return self._features


def test_composite_representation_preserves_uncertainty():
    """Regressão: qualificar o nome da Feature por prefixo de domínio não pode perder `uncertainty`."""
    filho = _DomainStub("apnea", [Feature(name="ido", value=1.5, uncertainty=0.3)])
    composto = CompositeRepresentation(name="respiratory", children=[filho])

    features = composto.transform(system=None)  # _DomainStub ignora `system`
    assert len(features) == 1
    assert features[0].name == "apnea.ido"
    assert features[0].uncertainty == pytest.approx(0.3), "uncertainty deveria ter sido preservado ao qualificar o nome."


def test_composite_representation_none_uncertainty_stays_none():
    filho = _DomainStub("apnea", [Feature(name="ido", value=1.5)])  # uncertainty=None (padrão)
    composto = CompositeRepresentation(name="respiratory", children=[filho])

    features = composto.transform(system=None)
    assert features[0].uncertainty is None


def _spiral_space():
    """Espiral de Arquimedes — mesmo cenário sintético usado para validar RiemannianGeometry no projeto."""
    from datetime import datetime

    from biospace.core import RepresentationSpace

    space = RepresentationSpace()
    n = 150
    thetas = np.linspace(0.5, 4 * np.pi, n)
    raios = thetas
    xs = raios * np.cos(thetas)
    ys = raios * np.sin(thetas)
    for i in range(n):
        vec = RepresentationVector(
            system_id=f"p{i}",
            timestamp=datetime(2020, 1, 1),
            components={"d": [Feature(name="x", value=float(xs[i])), Feature(name="y", value=float(ys[i]))]},
        )
        space.add(vec)
    return space


def test_riemannian_geodesic_exceeds_euclidean_across_spiral_loops():
    """Pontos em voltas ADJACENTES da espiral: próximos em linha reta, mas distantes ao longo da variedade."""
    space = _spiral_space()
    euclid = Euclidean()
    riem = RiemannianGeometry(k_neighbors=6)
    riem.fit(space)

    x_a = space.get("p140").as_vector()
    x_b = space.get("p100").as_vector()

    d_euclid = euclid.distance(x_a, x_b)
    d_riem = riem.distance(x_a, x_b)

    assert d_riem > d_euclid, "A geodésica deveria ser maior que a distância em linha reta ao atravessar voltas da espiral."
    assert d_riem / d_euclid > 1.2, "Esperava uma diferença substancial (>20%), não apenas ruído numérico."


def test_riemannian_geodesic_matches_euclidean_for_true_neighbors():
    """Pontos genuinamente vizinhos (mesma volta da espiral): geodésica deve ≈ Euclidiana."""
    space = _spiral_space()
    euclid = Euclidean()
    riem = RiemannianGeometry(k_neighbors=6)
    riem.fit(space)

    x_c = space.get("p10").as_vector()
    x_d = space.get("p12").as_vector()

    d_euclid = euclid.distance(x_c, x_d)
    d_riem = riem.distance(x_c, x_d)

    assert d_riem == pytest.approx(d_euclid, rel=0.15), "Para vizinhos genuínos, geodésica e Euclidiana deveriam ser bem próximas."


def test_riemannian_requires_fit_before_distance():
    riem = RiemannianGeometry()
    with pytest.raises(RuntimeError):
        riem.distance(np.array([0.0, 0.0]), np.array([1.0, 1.0]))


def test_riemannian_disconnected_graph_raises_clear_error():
    """Dois clusters genuinamente isolados (sem nenhum caminho no grafo k-NN) devem levantar erro claro, não um valor incorreto."""
    from datetime import datetime

    from biospace.core import RepresentationSpace

    space = RepresentationSpace()
    # Dois blocos MUITO distantes entre si, cada um denso internamente
    for i in range(10):
        vec = RepresentationVector(
            system_id=f"a{i}", timestamp=datetime(2020, 1, 1),
            components={"d": [Feature(name="x", value=float(i)), Feature(name="y", value=0.0)]},
        )
        space.add(vec)
    for i in range(10):
        vec = RepresentationVector(
            system_id=f"b{i}", timestamp=datetime(2020, 1, 1),
            components={"d": [Feature(name="x", value=float(i) + 10_000.0), Feature(name="y", value=0.0)]},
        )
        space.add(vec)

    riem = RiemannianGeometry(k_neighbors=3)
    riem.fit(space)

    x_a = space.get("a0").as_vector()
    x_b = space.get("b0").as_vector()
    with pytest.raises(ValueError):
        riem.distance(x_a, x_b)
