"""
tests.test_trajectory_updater
=================================

`biospace.longitudinal.TrajectoryUpdater` existia sem NENHUM teste
antes desta rodada — achado numa auditoria do projeto, mesmo padrão de
`prediction/`, `early_warning/`, `risk/`, `latent/`.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.longitudinal import NonMonotonicObservationError, TrajectoryUpdater


class _Obs(Observable):
    key = "x"


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_Obs()])

    def encode(self, measurements):
        v = float(measurements["x"].value)
        return [Feature(name="x", value=v, raw_value=v)]


_REPRESENTATION = Representation([_Dom()])


def test_apply_ingests_observation_without_creating_new_patient():
    """TESTE DECISIVO: apply() deveria atualizar o MESMO system_id, nunca criar um paciente novo -- ingestao, nao recriacao."""
    cohort = Cohort()
    system = BiologicalSystem(identifier="p1")
    updater = TrajectoryUpdater(_REPRESENTATION)

    updater.apply(cohort, system, Observation(timestamp=datetime(2020, 1, 1), source="t", values={"x": 1.0}))
    updater.apply(cohort, system, Observation(timestamp=datetime(2020, 2, 1), source="t", values={"x": 2.0}))

    assert len(cohort.trajectories) == 1
    assert len(cohort.trajectories["p1"]) == 2


def test_non_monotonic_observation_raises_by_default():
    cohort = Cohort()
    system = BiologicalSystem(identifier="p1")
    updater = TrajectoryUpdater(_REPRESENTATION)

    updater.apply(cohort, system, Observation(timestamp=datetime(2020, 2, 1), source="t", values={"x": 1.0}))
    with pytest.raises(NonMonotonicObservationError):
        updater.apply(cohort, system, Observation(timestamp=datetime(2020, 1, 1), source="t", values={"x": 2.0}))


def test_non_monotonic_observation_allowed_when_disabled():
    cohort = Cohort()
    system = BiologicalSystem(identifier="p1")
    updater = TrajectoryUpdater(_REPRESENTATION, enforce_monotonic_time=False)

    updater.apply(cohort, system, Observation(timestamp=datetime(2020, 2, 1), source="t", values={"x": 1.0}))
    updater.apply(cohort, system, Observation(timestamp=datetime(2020, 1, 1), source="t", values={"x": 2.0}))
    assert len(cohort.trajectories["p1"]) == 2


def test_on_update_callback_fires_with_system_and_vector():
    cohort = Cohort()
    system = BiologicalSystem(identifier="p1")
    chamadas = []
    updater = TrajectoryUpdater(_REPRESENTATION, on_update=lambda s, v: chamadas.append((s.id, v.system_id)))

    updater.apply(cohort, system, Observation(timestamp=datetime(2020, 1, 1), source="t", values={"x": 1.0}))
    assert len(chamadas) == 1
    assert chamadas[0] == ("p1", "p1")


def test_first_observation_never_raises_regardless_of_timestamp():
    """Sem observacao anterior, nao ha "monotonicidade" a violar -- a primeira observacao de um paciente novo deveria sempre funcionar."""
    cohort = Cohort()
    system = BiologicalSystem(identifier="p1")
    updater = TrajectoryUpdater(_REPRESENTATION)
    vector = updater.apply(cohort, system, Observation(timestamp=datetime(1900, 1, 1), source="t", values={"x": 1.0}))
    assert vector is not None
