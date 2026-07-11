"""
examples/01_new_disease_from_scratch.py
==========================================

Prova de que `biospace.core` é genérico de verdade: constrói uma doença
inteiramente nova (glicemia/diabetes) sem usar NADA do plugin sleep —
nem uma subclasse de BiologicalSystem, nem um domínio pré-existente.

Rode com: python3 examples/01_new_disease_from_scratch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite `import biospace` de dentro de examples/

from datetime import datetime

import numpy as np

from biospace.core import (
    BiologicalSystem,
    Cohort,
    Feature,
    Measurement,
    Observable,
    Observation,
    Representation,
    SemanticDomain,
)
from biospace.core.contracts import check_reproducibility, check_semantic_preservation
from biospace.geometry import Euclidean
from biospace.phenotyping import KMeansPhenotyper


# --- 1. Definir os Observables (o que pode ser medido) ---
class GlucoseObservable(Observable):
    key = "glicemia_mg_dl"
    unit = "mg/dL"
    description = "Glicemia de jejum"


class HbA1cObservable(Observable):
    key = "hba1c_pct"
    unit = "%"
    description = "Hemoglobina glicada"


# --- 2. Definir o SemanticDomain (como medir vira uma coordenada em X) ---
class GlycemicDomain(SemanticDomain):
    name = "glycemic"

    def __init__(self):
        super().__init__([GlucoseObservable(), HbA1cObservable()])

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        features = []
        for key, ref_mean, ref_std in [("glicemia_mg_dl", 100.0, 20.0), ("hba1c_pct", 5.5, 1.0)]:
            m = measurements.get(key)
            if m is None or m.is_missing():
                features.append(Feature(name=key, value=0.0, is_missing=True))
            else:
                z = (float(m.value) - ref_mean) / ref_std
                features.append(Feature(name=key, value=z, raw_value=float(m.value), z_score=z))
        return features


def make_patient(identifier: str, glicemia: float, hba1c: float, timestamp: datetime) -> BiologicalSystem:
    # BiologicalSystem usado DIRETAMENTE -- nem precisa de uma subclasse como SleepSystem.
    system = BiologicalSystem(identifier=identifier)
    system.observe(
        Observation(timestamp=timestamp, source="exame_laboratorial", values={"glicemia_mg_dl": glicemia, "hba1c_pct": hba1c})
    )
    return system


def main():
    representation = Representation([GlycemicDomain()])
    cohort = Cohort()
    t0 = datetime(2023, 1, 1)
    rng = np.random.default_rng(0)

    for i in range(15):
        cohort.update(make_patient(f"normal_{i}", float(rng.normal(90, 5)), float(rng.normal(5.3, 0.3)), t0), representation, timestamp=t0)
    for i in range(15):
        cohort.update(make_patient(f"diabetico_{i}", float(rng.normal(190, 10)), float(rng.normal(9.0, 0.5)), t0), representation, timestamp=t0)

    print(f"Coorte construída: {len(cohort)} pacientes, 0 dependências do plugin sleep.\n")

    # --- Geometria genérica funciona sem nenhuma alteração ---
    space = cohort.snapshot()
    order = space.order()
    euclid = Euclidean()
    d_entre_grupos = euclid.distance(space.get("normal_0").as_vector(order), space.get("diabetico_0").as_vector(order))
    d_mesmo_grupo = euclid.distance(space.get("normal_0").as_vector(order), space.get("normal_1").as_vector(order))
    print(f"Distância normal-normal:     {d_mesmo_grupo:.3f}")
    print(f"Distância normal-diabético:  {d_entre_grupos:.3f}  (deveria ser maior)\n")

    # --- Contratos formais valem para QUALQUER domínio ---
    domain = GlycemicDomain()
    normal = make_patient("n1", 90.0, 5.2, t0)
    diabetico = make_patient("d1", 190.0, 9.0, t0)
    print("Reprodutibilidade:", check_reproducibility(domain, normal))
    print("Preservação semântica:", check_semantic_preservation(domain, normal, diabetico), "\n")

    # --- Fenotipagem genérica separa os grupos sem nenhum código de doença ---
    phenotypes = KMeansPhenotyper(n_clusters=2).fit(space)
    labels = {sid: next((ph.name for ph in phenotypes if ph.contains(space.get(sid).as_vector(order))), None) for sid in space.ids()}
    normal_labels = {v for k, v in labels.items() if k.startswith("normal_")}
    diabetico_labels = {v for k, v in labels.items() if k.startswith("diabetico_")}
    print(f"Fenótipos dos 'normais':    {normal_labels}")
    print(f"Fenótipos dos 'diabéticos': {diabetico_labels}")
    print("Grupos separados corretamente:", normal_labels != diabetico_labels)


if __name__ == "__main__":
    main()
