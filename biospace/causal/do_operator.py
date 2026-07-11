"""
biospace.causal.do_operator
==============================

DigitalTwin: o operador do() de Pearl, aplicado literalmente sobre a
representação computacional de um paciente:

    digital_twin = DigitalTwin.clone_from(trajectory)
    digital_twin.do(CPAP)          # τ: X -> X, InterventionOperator
    digital_twin.simulate(...)     # EvolutionOperator, dinâmica espontânea após a intervenção

Isto é deliberadamente uma camada FINA reaproveitando dois mecanismos já
existentes e já testados — `InterventionOperator` (biospace.intervention)
para o `do()`, e `EvolutionOperator` (biospace.dynamics) para
`simulate()` — não um novo motor de cálculo.

AVISO CENTRAL (repetido de propósito): `do()` aqui aplica uma
TRANSFORMAÇÃO no espaço de representação — seja ela hipotética
(`FeatureShiftIntervention`, um "e se" fisiologicamente motivado, sem
dado real por trás) ou estimada de dados observacionais
(`ObservationalEffectEstimator`, sujeita a confundimento — ver
`biospace.causal.balance`). NENHUMA das duas é uma inferência causal
IDENTIFICADA no sentido formal de Pearl (não há grafo causal validado,
nem ajuste por confundidores desconhecidos). O nome `do()` é usado pela
clareza conceitual do paralelo, não como uma alegação de que a
identificação causal foi resolvida.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationVector, Trajectory
    from biospace.dynamics import EvolutionOperator
    from biospace.intervention import InterventionOperator

__all__ = ["DigitalTwin"]


@dataclass
class DigitalTwin:
    """
    `state`: o estado atual do gêmeo (RepresentationVector) — começa como
    uma CÓPIA PROFUNDA do último ponto real da trajetória de origem
    (nunca compartilha objetos Feature com o paciente real: aplicar
    `do()` no gêmeo nunca deve alterar o histórico real do paciente).
    `history`: log legível de cada `do()`/`simulate()` aplicado, nesta ordem.
    """

    source_system_id: str
    state: "RepresentationVector"
    order: Optional[Sequence[str]] = None
    history: list[str] = field(default_factory=list)

    @classmethod
    def clone_from(cls, trajectory: "Trajectory", order: Optional[Sequence[str]] = None) -> "DigitalTwin":
        """`patient.clone()` — clona o ÚLTIMO estado observado da trajetória real."""
        latest = trajectory.latest()
        cloned_components = copy.deepcopy(latest.components)
        from biospace.core import RepresentationVector

        cloned_state = RepresentationVector(
            system_id=f"{latest.system_id}_twin", timestamp=latest.timestamp, components=cloned_components
        )
        return cls(source_system_id=latest.system_id, state=cloned_state, order=order)

    def do(self, intervention: "InterventionOperator") -> "DigitalTwin":
        """Aplica uma intervenção (τ: X -> X) ao estado atual do gêmeo. Muta e retorna `self` (encadeável)."""
        self.state = intervention.apply(self.state)
        self.history.append(f"do({intervention.describe()})")
        return self

    def current_vector(self) -> np.ndarray:
        return self.state.as_vector(self.order)

    def simulate(
        self, evolution_operator: "EvolutionOperator", horizon_days: float, step_days: float = 30.0
    ) -> list[tuple[float, np.ndarray]]:
        """
        Projeta a evolução ESPONTÂNEA (via `EvolutionOperator`, ajustado
        sobre uma Cohort — ver `biospace.dynamics`) a partir do estado
        atual do gêmeo (já refletindo qualquer `do()` aplicado antes).
        Os pontos retornados são vetores brutos SIMULADOS — não têm
        proveniência de Measurement real (são previsões, não observações).

        DETERMINÍSTICA — um único futuro (a média da distribuição
        preditiva). Para múltiplos futuros possíveis com faixa de
        incerteza, ver `simulate_ensemble()`.
        """
        x = self.current_vector()
        t = 0.0
        path: list[tuple[float, np.ndarray]] = [(0.0, x)]
        while t < horizon_days - 1e-9:
            step = min(step_days, horizon_days - t)
            x = evolution_operator.predict(x, step)
            t += step
            path.append((t, x))
        self.history.append(f"simulate(horizon_days={horizon_days}, step_days={step_days})")
        return path

    def simulate_ensemble(
        self,
        evolution_operator: "EvolutionOperator",
        horizon_days: float,
        step_days: float = 30.0,
        n_samples: int = 200,
        seed: int = 42,
    ) -> dict:
        """
        Fase 9 — Simulação: em vez de UM futuro determinístico
        (`simulate()`), gera `n_samples` trajetórias INDEPENDENTES via
        `evolution_operator.sample()` (a versão estocástica de
        `predict()`, com a variância teoricamente correta do processo de
        Ornstein-Uhlenbeck — ver `MeanRevertingEvolutionOperator.sample()`)
        — um gêmeo digital de verdade relata incerteza, não um único
        ponto. Requer que `evolution_operator` implemente `.sample()`
        (nem todo `EvolutionOperator` precisa — é uma capacidade
        opcional, não parte da interface mínima).

        Retorna um dict com `times` (nº de pontos no tempo), `paths`
        (array `n_samples x n_pontos x n_features` — todas as
        trajetórias simuladas), `mean` e `std` (média/desvio padrão da
        distribuição preditiva por instante — `n_pontos x n_features`).
        """
        if not hasattr(evolution_operator, "sample"):
            raise TypeError(
                f"{type(evolution_operator).__name__} não implementa `.sample()` — simulação estocástica "
                "exige uma versão estocástica de predict(). Use `simulate()` (determinística) em vez disso."
            )

        rng = np.random.default_rng(seed)
        x0 = self.current_vector()

        times = [0.0]
        t = 0.0
        while t < horizon_days - 1e-9:
            t += min(step_days, horizon_days - t)
            times.append(t)

        n_points = len(times)
        n_features = len(x0)
        paths = np.empty((n_samples, n_points, n_features), dtype=float)

        for s in range(n_samples):
            x = x0.copy()
            paths[s, 0] = x
            t = 0.0
            for i in range(1, n_points):
                step = times[i] - t
                x = evolution_operator.sample(x, step, rng)
                paths[s, i] = x
                t = times[i]

        self.history.append(f"simulate_ensemble(horizon_days={horizon_days}, step_days={step_days}, n_samples={n_samples})")
        return {
            "times": np.array(times),
            "paths": paths,
            "mean": paths.mean(axis=0),
            "std": paths.std(axis=0),
        }

    def __repr__(self) -> str:
        return f"DigitalTwin(origem={self.source_system_id!r}, historico={self.history})"
