"""
biospace.causal.scenario
===========================

Scenario: compara MÚLTIPLOS braços de intervenção — CPAP, AAM, "Cirurgia",
"Perda de peso" — aplicados ao MESMO estado inicial (um gêmeo digital
clonado para cada braço), simulando a evolução espontânea depois de cada
`do()` e comparando as trajetórias resultantes.

AVISO (mesmo do módulo `do_operator`): cada braço é ou uma associação
observacional (com aviso de confundimento) ou uma intervenção hipotética
(`FeatureShiftIntervention`, um "e se" sem dado real por trás) — rotule
cada braço honestamente conforme sua natureza ao construir o Scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .do_operator import DigitalTwin

if TYPE_CHECKING:
    from biospace.core import Trajectory
    from biospace.dynamics import EvolutionOperator
    from biospace.geometry import Geometry
    from biospace.intervention import InterventionOperator

__all__ = ["ScenarioArmResult", "Scenario"]


@dataclass
class ScenarioArmResult:
    label: str
    path: list[tuple[float, np.ndarray]]
    final_state: np.ndarray
    history: list[str] = field(default_factory=list)


class Scenario:
    """
    `add_arm(label, intervention)`: registra um braço nomeado. Um braço
    especial `"controle"` (sem nenhum `do()`, só `simulate()`) é
    automaticamente incluído se não for informado, servindo de
    referência ("o que aconteceria sem nenhuma intervenção").
    """

    def __init__(self, name: str):
        self.name = name
        self.arms: dict[str, Optional["InterventionOperator"]] = {}

    def add_arm(self, label: str, intervention: "InterventionOperator") -> "Scenario":
        self.arms[label] = intervention
        return self

    def run(
        self,
        trajectory: "Trajectory",
        evolution_operator: "EvolutionOperator",
        horizon_days: float,
        step_days: float = 30.0,
        order: Optional[Sequence[str]] = None,
    ) -> dict[str, ScenarioArmResult]:
        arms = dict(self.arms)
        if "controle" not in arms:
            arms = {"controle": None, **arms}

        results: dict[str, ScenarioArmResult] = {}
        for label, intervention in arms.items():
            twin = DigitalTwin.clone_from(trajectory, order=order)
            if intervention is not None:
                twin.do(intervention)
            path = twin.simulate(evolution_operator, horizon_days, step_days)
            results[label] = ScenarioArmResult(label=label, path=path, final_state=path[-1][1], history=list(twin.history))
        return results

    @staticmethod
    def compare_to_control(
        results: dict[str, ScenarioArmResult], geometry: "Geometry", control_label: str = "controle"
    ) -> dict[str, float]:
        """Distância (segundo `geometry`) do estado final de cada braço ao estado final do braço de controle."""
        if control_label not in results:
            raise KeyError(f"Braço de controle {control_label!r} não encontrado nos resultados.")
        control_final = results[control_label].final_state
        return {
            label: geometry.distance(r.final_state, control_final)
            for label, r in results.items()
            if label != control_label
        }

    def __repr__(self) -> str:
        return f"Scenario({self.name!r}, braços={list(self.arms.keys())})"
