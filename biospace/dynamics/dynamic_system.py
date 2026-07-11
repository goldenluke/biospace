"""
biospace.dynamics.dynamic_system
===================================

DynamicSystem: combina uma Trajectory observada (o passado) com um
EvolutionOperator AJUSTADO (a regra aprendida de evolução espontânea),
permitindo extrapolar — prever ou simular — além do que foi observado:

    Trajectory -> EvolutionOperator -> Future State

Isso é deliberadamente uma camada FINA por cima de infraestrutura já
existente (Trajectory, EvolutionOperator) — não introduz nenhum
mecanismo novo de cálculo, só compõe os dois de forma conveniente para
uso em previsão/simulação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .evolution import EvolutionOperator

if TYPE_CHECKING:
    from biospace.core import Trajectory

__all__ = ["DynamicSystem"]


@dataclass
class DynamicSystem:
    """
    `trajectory`: a Trajectory observada (passado real) de UM sistema.
    `evolution_operator`: um EvolutionOperator JÁ AJUSTADO (via
    `.fit(cohort)`) — tipicamente ajustado sobre a COORTE INTEIRA, não
    apenas sobre este paciente, para ter dados suficientes; aplicado
    aqui a um paciente específico para prever seu futuro individual.
    `order`: ordem de domínios usada para achatar os vetores — deve
    corresponder à mesma ordem usada ao ajustar `evolution_operator`.
    """

    trajectory: "Trajectory"
    evolution_operator: EvolutionOperator
    order: Optional[Sequence[str]] = None

    def current_state(self) -> np.ndarray:
        """O estado mais recente observado (último ponto da trajetória)."""
        return self.trajectory.latest().as_vector(self.order)

    def predict(self, delta_t_days: float) -> np.ndarray:
        """Prevê o estado `delta_t_days` no futuro, a partir do ÚLTIMO ponto observado."""
        return self.evolution_operator.predict(self.current_state(), delta_t_days)

    def simulate(self, horizon_days: float, step_days: float = 30.0) -> list[tuple[float, np.ndarray]]:
        """
        Simula uma sequência de estados futuros a cada `step_days`, até
        `horizon_days`, encadeando previsões sucessivas (cada passo usa o
        estado previsto do passo anterior como novo "presente"). Retorna
        uma lista de (dias_desde_agora, estado_previsto), incluindo o
        estado atual em t=0.
        """
        x = self.current_state()
        t = 0.0
        path: list[tuple[float, np.ndarray]] = [(0.0, x)]
        while t < horizon_days - 1e-9:
            step = min(step_days, horizon_days - t)
            x = self.evolution_operator.predict(x, step)
            t += step
            path.append((t, x))
        return path
