"""
biospace.core.cohort
=======================

Cohort (C): C = {Γ_1, ..., Γ_n} — uma coleção viva de trajetórias. Cada
nova observação apenas atualiza a coorte; nunca a reconstrói.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .representation_space import RepresentationSpace
from .trajectory import Trajectory

if TYPE_CHECKING:
    from .biological_system import BiologicalSystem
    from .phenotype import Phenotype
    from .representation import Representation, RepresentationVector

__all__ = ["Cohort"]


class Cohort:
    def __init__(self):
        self.trajectories: dict[str, Trajectory] = {}
        self.systems: dict[str, "BiologicalSystem"] = {}

    def update(
        self,
        system: "BiologicalSystem",
        representation: "Representation",
        timestamp: Optional[datetime] = None,
    ) -> "RepresentationVector":
        vector = representation.transform(system, timestamp)
        self.systems[system.id] = system
        traj = self.trajectories.setdefault(system.id, Trajectory(system.id))
        traj.update(vector)
        system.trajectory = traj
        return vector

    def snapshot(self, index: int = -1, domain_order: Optional[Sequence[str]] = None) -> RepresentationSpace:
        """
        Corte transversal da coorte: um ponto por trajetória nāo-vazia.

        AVISO: se `index` estiver fora do alcance de uma trajetória
        específica (ex.: `index=5` mas aquele paciente só tem 2 exames),
        essa trajetória cai SILENCIOSAMENTE para o último ponto (`-1`) em
        vez de levantar erro — intencional (trajetórias têm comprimentos
        diferentes; forçar todas a terem `index` exigiria descartar
        pacientes com poucos exames), mas significa que um snapshot pode
        misturar "o 6º exame" de uns com "o exame mais recente" de
        outros sem aviso explícito. Para saber qual regra se aplicou a
        cada sistema, compare `len(cohort.trajectories[sid])` com `index`
        você mesmo antes de chamar isto, se isso importar para sua análise.
        """
        space = RepresentationSpace(domain_order=domain_order)
        for traj in self.trajectories.values():
            if len(traj) == 0:
                continue
            idx = index if -len(traj) <= index < len(traj) else -1
            space.add(traj.at(idx))
        return space

    def transition_matrix(
        self, phenotypes: Sequence["Phenotype"], order: Optional[Sequence[str]] = None
    ) -> tuple[np.ndarray, list[str]]:
        names = [ph.name for ph in phenotypes]
        idx = {n: i for i, n in enumerate(names)}
        counts = np.zeros((len(names), len(names)))
        for traj in self.trajectories.values():
            seq = traj.phenotype_sequence(phenotypes, order)
            for a, b in zip(seq, seq[1:]):
                if a is not None and b is not None:
                    counts[idx[a], idx[b]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return counts / row_sums, names

    def __len__(self) -> int:
        return len(self.trajectories)

    def __repr__(self) -> str:
        return f"Cohort(n_systems={len(self)})"
