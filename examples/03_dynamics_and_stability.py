"""
examples/03_dynamics_and_stability.py
========================================

Trajectory -> EvolutionOperator -> Future State. Ajusta um processo de
Ornstein-Uhlenbeck discreto por Feature sobre pares consecutivos de uma
coorte sintética (progressão espontânea, sem intervenção), avalia
estabilidade, e simula a evolução de um paciente específico.

Rode com: python3 examples/03_dynamics_and_stability.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

import numpy as np

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.dynamics import DynamicSystem, MeanRevertingEvolutionOperator, StabilityOperator


class SeverityObservable(Observable):
    key = "severity"


class SeverityDomain(SemanticDomain):
    name = "severity_domain"

    def __init__(self):
        super().__init__([SeverityObservable()])

    def encode(self, measurements):
        m = measurements.get("severity")
        return [Feature(name="severity", value=float(m.value) if m else 0.0)]


def build_mean_reverting_cohort(n_patients: int = 40, n_exams: int = 6, seed: int = 0):
    """
    Cada paciente reverte, com ruído, a um nível de equilíbrio populacional
    de 20 (phi=0.9/dia) -- simula uma condição crônica estável, sem
    tendência sistemática de piora.
    """
    rng = np.random.default_rng(seed)
    representation = Representation([SeverityDomain()])
    cohort = Cohort()
    mu, phi = 20.0, 0.9

    for i in range(n_patients):
        t0 = datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        x = rng.normal(mu, 10)  # estado inicial, longe do equilíbrio
        system = BiologicalSystem(identifier=f"p{i}")

        t = t0
        system.observe(Observation(timestamp=t, source="exame", values={"severity": x}))
        cohort.update(system, representation, timestamp=t)

        for _ in range(1, n_exams):
            dt = int(rng.integers(20, 90))
            t = t + timedelta(days=dt)
            x = mu + phi**dt * (x - mu) + rng.normal(0, 1.0)
            system.observe(Observation(timestamp=t, source="exame", values={"severity": x}))
            cohort.update(system, representation, timestamp=t)

    return cohort, representation


def main():
    cohort, representation = build_mean_reverting_cohort()
    order = representation.domain_names()

    print("--- Ajustando EvolutionOperator sobre a coorte ---")
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)
    print(evo.describe())
    for name, fd in evo.dynamics_.items():
        print(f"  {fd}")

    print("\n--- Avaliação de estabilidade ---")
    stability = StabilityOperator(evolution_operator=evo).analyze(cohort)
    print(stability.summary())

    print("\n--- Previsão/simulação para um paciente ---")
    sid = next(iter(cohort.trajectories))
    traj = cohort.trajectories[sid]
    ds = DynamicSystem(trajectory=traj, evolution_operator=evo, order=order)
    print(f"Estado atual: {ds.current_state()}")
    print(f"Previsão em 180 dias: {ds.predict(180.0)}")
    print("Simulação (passo de 60 dias, horizonte de 360 dias):")
    for t, x in ds.simulate(horizon_days=360, step_days=60):
        print(f"  t={t:.0f}d: severity={x[0]:.2f}")


if __name__ == "__main__":
    main()
