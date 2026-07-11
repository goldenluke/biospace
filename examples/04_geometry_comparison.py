"""
examples/04_geometry_comparison.py
=====================================

A mesma dupla de pontos, sob geometrias diferentes, conta histórias
diferentes. Compara Euclidiana, Mahalanobis (covariância populacional),
Cosine (só direção) e uma métrica APRENDIDA via NCA sobre rótulos.

Rode com: python3 examples/04_geometry_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import numpy as np

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain
from biospace.geometry import Cosine, Euclidean, LearnedGeometry, Mahalanobis


class F1(Observable):
    key = "f1"


class F2(Observable):
    key = "f2"


class ToyDomain(SemanticDomain):
    name = "toy"

    def __init__(self):
        super().__init__([F1(), F2()])

    def encode(self, measurements):
        return [
            Feature(name="f1", value=float(measurements["f1"].value)),
            Feature(name="f2", value=float(measurements["f2"].value)),
        ]


def main():
    rng = np.random.default_rng(0)
    representation = Representation([ToyDomain()])
    cohort = Cohort()
    t0 = datetime(2023, 1, 1)

    # Duas classes com covariância bem diferente da identidade (elipses alongadas em direções opostas)
    labels = {}
    for i in range(30):
        f1, f2 = rng.normal(0, 3), rng.normal(0, 0.5)  # classe A: alongada no eixo f1
        sid = f"a{i}"
        system = BiologicalSystem(identifier=sid)
        system.observe(Observation(timestamp=t0, source="toy", values={"f1": f1, "f2": f2}))
        cohort.update(system, representation, timestamp=t0)
        labels[sid] = "A"
    for i in range(30):
        f1, f2 = rng.normal(0, 0.5), rng.normal(0, 3)  # classe B: alongada no eixo f2
        sid = f"b{i}"
        system = BiologicalSystem(identifier=sid)
        system.observe(Observation(timestamp=t0, source="toy", values={"f1": f1, "f2": f2}))
        cohort.update(system, representation, timestamp=t0)
        labels[sid] = "B"

    space = cohort.snapshot()
    order = space.order()
    matrix, ids = space.matrix()

    x = space.get("a0").as_vector(order)
    y = space.get("b0").as_vector(order)

    print("Mesma dupla de pontos (x da classe A, y da classe B), sob geometrias diferentes:\n")
    print(f"  Euclidiana:   {Euclidean().distance(x, y):.3f}")

    cov = np.cov(matrix, rowvar=False)
    print(f"  Mahalanobis:  {Mahalanobis(cov).distance(x, y):.3f}  (usa a covariância populacional)")

    print(f"  Cosine:       {Cosine().distance(x, y):.3f}  (ignora magnitude, só direção)")

    learned = LearnedGeometry()
    learned.fit(space, labels)
    print(f"  Aprendida (NCA): {learned.distance(x, y):.3f}  (otimizada para separar as classes A/B)")

    print("\nNenhuma é 'a' distância certa — cada uma responde a uma pergunta diferente (ver README).")


if __name__ == "__main__":
    main()
