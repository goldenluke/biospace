"""
examples/09_curvature_and_metastability.py
=============================================

Fase 8 -- Geometria:

    Paciente -> Representacao -> Variedade -> Trajetoria -> Curvatura -> Estabilidade

Tres formas INDEPENDENTES de estimar curvatura neste projeto:

  1. TEMPORAL (`FeatureDynamics.curvature`, em biospace.dynamics) -- vem
     do phi ja ajustado por MeanRevertingEvolutionOperator (dinamica de
     UM paciente ao longo do tempo). k = -ln(phi).
  2. DENSIDADE POPULACIONAL (`estimate_density_curvature`,
     `detect_metastability`) -- vem da FORMA da distribuicao
     transversal (um instante, muitos pacientes) via reconstrucao de um
     potencial efetivo U(x) = -log(densidade(x)).
  3. ESTRUTURAL/GRAFO (`ollivier_ricci_curvature`, este exemplo) -- vem
     da VARIEDADE em si (o grafo k-NN de RiemannianGeometry), via
     transporte otimo entre vizinhancas.

Rode com: python3 examples/09_curvature_and_metastability.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta

import numpy as np

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.dynamics import MeanRevertingEvolutionOperator
from biospace.geometry import Euclidean, detect_metastability, graph_curvature_summary
from biospace.graph import build_cohort_similarity_graph


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


class _Dom(SemanticDomain):
    name = "d"

    def __init__(self, keys):
        self._keys = keys
        super().__init__([_FlagObservable(k) for k in keys])

    def encode(self, measurements):
        return [Feature(name=k, value=float(measurements[k].value)) for k in self._keys if k in measurements]


def main():
    print("--- 1. Curvatura TEMPORAL (via EvolutionOperator, k = -ln(phi)) ---")
    mu, dt_fixo, target_var = 20.0, 20, 4.0
    k_verdadeiro = 0.08
    phi = np.exp(-k_verdadeiro * dt_fixo)
    sigma_eps = np.sqrt(target_var * (1 - phi**2))

    representation = Representation([_Dom(["severidade"])])
    cohort = Cohort()
    rng = np.random.default_rng(0)
    t0 = datetime(2020, 1, 1)
    for i in range(100):
        x = rng.normal(mu, np.sqrt(target_var))
        system = BiologicalSystem(identifier=f"p{i}")
        t = t0
        system.observe(Observation(timestamp=t, source="t", values={"severidade": x}))
        cohort.update(system, representation, timestamp=t)
        for _ in range(4):
            t = t + timedelta(days=dt_fixo)
            x = mu + phi * (x - mu) + rng.normal(0, sigma_eps)
            system.observe(Observation(timestamp=t, source="t", values={"severidade": x}))
            cohort.update(system, representation, timestamp=t)

    evo = MeanRevertingEvolutionOperator(order=representation.domain_names())
    evo.fit(cohort)
    fd = evo.dynamics_["d.severidade"]
    print(f"k verdadeiro: {k_verdadeiro}, curvatura recuperada: {fd.curvature:.4f}, resilience_score: {fd.resilience_score:.4f}\n")

    print("--- 2. Metaestabilidade via DENSIDADE populacional (poços de potencial) ---")
    space = cohort.snapshot()
    report = detect_metastability(space, "d.severidade", min_prominence=0.3)
    print(report.summary())
    print()

    print("--- 3. Curvatura ESTRUTURAL via grafo (Ollivier-Ricci) ---")
    order = space.order()
    G = build_cohort_similarity_graph(space, Euclidean(), k=5, order=order)
    resumo = graph_curvature_summary(G, weight="weight")
    print(f"Curvatura global do grafo de similaridade: media={resumo['global_mean']:.3f}, min={resumo['global_min']:.3f}, max={resumo['global_max']:.3f}")
    print()
    print("Achado real nos dados de SAOS (355 pacientes, ver README):")
    print("  arestas DENTRO do mesmo fenotipo: media=-0.076")
    print("  arestas ENTRE fenotipos diferentes: media=-0.146 (mais negativa -- gargalo estrutural)")
    print("  Mann-Whitney p=5.7e-19 -- diferenca claramente real, nao ruido")


if __name__ == "__main__":
    main()
