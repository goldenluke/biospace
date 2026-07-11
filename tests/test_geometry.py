"""
tests.test_geometry
======================

Propriedades básicas das geometrias pontuais, e duas regressões de
achados reais do projeto:

  1. DTW distingue DIREÇÃO temporal (crescente vs decrescente);
     Gromov-Wasserstein NÃO distingue (propriedade matemática correta,
     não um bug — mas precisa continuar assim de propósito).
  2. PhenotypeConditionedGeometry não pode mais explodir numericamente
     quando um grupo tem poucos membros relativos à dimensionalidade
     (bug real: sem encolhimento, um grupo n=40 < p=52 deu distância
     ~90x maior que os demais).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from biospace.core import Cohort
from biospace.geometry import Cosine, DTW, Euclidean, GromovWasserstein
from biospace.plugins.sleep import SleepRepresentation, SleepSystem
from biospace.plugins.sleep.builders import exam


def test_euclidean_basic_metric_properties():
    euclid = Euclidean()
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 0.0, 3.0])
    assert euclid.distance(x, x) == 0.0
    assert euclid.distance(x, y) == euclid.distance(y, x)
    assert euclid.distance(x, y) > 0.0


def test_cosine_same_direction_different_magnitude_is_zero():
    cosine = Cosine()
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])  # mesma direção, magnitude diferente
    assert cosine.distance(x, y) == pytest.approx(0.0, abs=1e-9)


def test_cosine_opposite_direction_is_two():
    cosine = Cosine()
    x = np.array([1.0, 2.0, 3.0])
    assert cosine.distance(x, -x) == pytest.approx(2.0, abs=1e-9)


def _build_trajectory(ido_values: list[float], exam_values_factory) -> tuple:
    representation = SleepRepresentation()
    cohort = Cohort()
    system = SleepSystem()
    t0 = datetime(2020, 1, 1)
    for i, ido in enumerate(ido_values):
        system.observe(exam(exam_values_factory(ido=ido, ido_sono=ido), timestamp=t0 + timedelta(days=i * 30)))
        cohort.update(system, representation, timestamp=t0 + timedelta(days=i * 30))
    return cohort.trajectories[system.id], representation.domain_names()


def test_dtw_distinguishes_direction(exam_values_factory):
    """Trajetórias crescente e decrescente (mesma forma, direção oposta) devem ter DTW claramente maior que duas crescentes parecidas."""
    traj_up_a, order = _build_trajectory([10, 15, 20, 25, 30], exam_values_factory)
    traj_up_b, _ = _build_trajectory([10, 14, 19, 24, 30], exam_values_factory)
    traj_down, _ = _build_trajectory([30, 25, 20, 15, 10], exam_values_factory)

    dtw = DTW(order=order)
    d_similar = dtw.distance(traj_up_a, traj_up_b)
    d_opposite = dtw.distance(traj_up_a, traj_down)

    assert d_opposite > d_similar, "DTW deveria distinguir direção temporal — trajetórias opostas devem ficar mais distantes."


def test_gromov_wasserstein_is_direction_invariant(exam_values_factory):
    """
    Propriedade matemática correta (não bug): GW só enxerga estrutura
    relacional interna, não ordem/direção — uma trajetória e sua
    'espelhada' no tempo devem ficar QUASE tão próximas quanto duas
    trajetórias genuinamente parecidas.
    """
    traj_up, order = _build_trajectory([10, 15, 20, 25, 30], exam_values_factory)
    traj_up_similar, _ = _build_trajectory([10, 14, 19, 24, 30], exam_values_factory)
    traj_down, _ = _build_trajectory([30, 25, 20, 15, 10], exam_values_factory)

    gw = GromovWasserstein(epsilon=0.05, max_iter=200)
    d_similar = gw.distance(traj_up, traj_up_similar)
    d_opposite = gw.distance(traj_up, traj_down)

    # Não exigimos igualdade exata (a otimização é aproximada), só que
    # a diferença de direção não infle a distância dramaticamente —
    # ambas devem ficar na mesma ordem de grandeza.
    assert d_opposite < d_similar * 3, (
        "GW deveria ser aproximadamente invariante à direção temporal — "
        "se a distância 'oposta' explodiu, a propriedade matemática esperada quebrou."
    )


def test_phenotype_conditioned_geometry_does_not_explode_with_small_group(small_cohort):
    """
    Regressão do bug real: um grupo pequeno (poucos membros relativos à
    dimensionalidade) não pode mais produzir uma distância absurdamente
    maior que os demais grupos (era ~90x antes do encolhimento Ledoit-Wolf).
    """
    from biospace.geometry import PhenotypeConditionedGeometry
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = small_cohort
    space = cohort.snapshot()
    order = space.order()

    if len(space) < 4:
        pytest.skip("Cohort sintética pequena demais para K=2 com 2+ membros por grupo.")

    phenotyper = KMeansPhenotyper(n_clusters=2)
    phenotypes = phenotyper.fit(space)

    geo = PhenotypeConditionedGeometry()
    geo.fit(space, phenotypes, order=order)

    ids = space.ids()
    xa, xb = space.get(ids[0]).as_vector(order), space.get(ids[1]).as_vector(order)

    distances = [geo.distance(xa, xb, ph.name) for ph in phenotypes]
    if min(distances) > 1e-9:
        ratio = max(distances) / min(distances)
        assert ratio < 20, (
            f"Razão entre a maior e a menor distância entre fenótipos foi {ratio:.1f}x — "
            "esperado ficar numa faixa razoável (era ~90x antes da correção com Ledoit-Wolf)."
        )
