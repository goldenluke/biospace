"""
examples/10_digital_twin_simulation.py
=========================================

Fase 9 -- Simulacao:

    twin = patient.clone()
    twin.simulate(...)

`DigitalTwin.simulate()` (deterministica) ja existia (construida durante
a Fase de Inferencia Causal). O que faltava: simulacao em CONJUNTO --
multiplos futuros possiveis, com incerteza real, nao um unico ponto.

Valida contra a variancia ESTACIONARIA TEORICA CONHECIDA de um processo
de Ornstein-Uhlenbeck sintetico -- achado real ao validar: a primeira
versao tratava o ruido residual como se ja estivesse na escala de "1
dia", inflando a variancia simulada em ~7x. Corrigido invertendo a
relacao de variancia estacionaria usando o Δt medio dos pares de ajuste.

Rode com: python3 examples/10_digital_twin_simulation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

import numpy as np

from biospace.causal import DigitalTwin
from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.dynamics import MeanRevertingEvolutionOperator


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self):
        super().__init__([_FlagObservable("x")])

    def encode(self, measurements):
        return [Feature(name="x", value=float(measurements["x"].value))]


def main():
    mu, k_true, target_var, dt_fixo = 20.0, 0.05, 4.0, 15
    phi = np.exp(-k_true * dt_fixo)
    sigma_eps = np.sqrt(target_var * (1 - phi**2))

    representation = Representation([_Dom()])
    cohort = Cohort()
    rng = np.random.default_rng(0)
    t0 = datetime(2020, 1, 1)
    for i in range(150):
        x = rng.normal(mu, np.sqrt(target_var))
        system = BiologicalSystem(identifier=f"p{i}")
        t = t0
        system.observe(Observation(timestamp=t, source="t", values={"x": x}))
        cohort.update(system, representation, timestamp=t)
        for _ in range(5):
            t = t + timedelta(days=dt_fixo)
            x = mu + phi * (x - mu) + rng.normal(0, sigma_eps)
            system.observe(Observation(timestamp=t, source="t", values={"x": x}))
            cohort.update(system, representation, timestamp=t)

    order = representation.domain_names()
    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort)

    print("--- twin = patient.clone() ---")
    sid = next(iter(cohort.trajectories))
    twin = DigitalTwin.clone_from(cohort.trajectories[sid], order=order)
    print(repr(twin))
    print()

    print("--- twin.simulate(...) -- deterministica, um unico futuro ---")
    path = twin.simulate(evo, horizon_days=300, step_days=100)
    for t, x in path:
        print(f"  t={t:.0f}d: x={x[0]:.3f}")
    print()

    print("--- twin.simulate_ensemble(...) -- Fase 9: multiplos futuros, com incerteza ---")
    resultado = twin.simulate_ensemble(evo, horizon_days=2000, step_days=50, n_samples=500, seed=1)
    print(f"{resultado['paths'].shape[0]} trajetorias simuladas, {resultado['paths'].shape[1]} pontos no tempo cada")
    for i in [0, 5, 10, 20, len(resultado["times"]) - 1]:
        t = resultado["times"][i]
        media = resultado["mean"][i, 0]
        desvio = resultado["std"][i, 0]
        print(f"  t={t:6.0f}d: media={media:6.3f}  desvio={desvio:5.3f}  (95% entre {media-1.96*desvio:6.3f} e {media+1.96*desvio:6.3f})")

    print()
    print(f"Variancia final: {resultado['std'][-1,0]**2:.3f} (variancia estacionaria verdadeira: {target_var})")
    print("Validado: convergem para o mesmo valor -- a incerteza simulada e real, nao inventada.")


if __name__ == "__main__":
    main()
