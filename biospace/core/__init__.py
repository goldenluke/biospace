"""
biospace.core
===============

O meta-modelo dividido em uma entidade por arquivo:

    biological_system.py    BiologicalSystem       (B)
    observation.py          Observation, Observable (O)
    measurement.py          Measurement              — valor + proveniência
    feature.py               Feature, features_to_array — coordenada auditável
    domain.py                SemanticDomain          (D)
    representation.py        Representation, RepresentationVector (R)
    representation_space.py  RepresentationSpace     (X)
    geometry.py               Geometry (interface)    (G)
    trajectory.py             Trajectory              (Γ)
    phenotype.py              Phenotype               (F)
    cohort.py                 Cohort                  (C)
    operator.py               Operator                — interface p/ algoritmos
    plugin.py                  Plugin                  — interface p/ módulos de doença
    contracts.py               verificação empírica dos contratos formais (Seção 5)

Regras deste pacote (ver AGENT.md): nunca conhece uma doença específica,
nunca conhece um algoritmo específico, nunca conhece uma modalidade
específica, nunca importa pandas.
"""

from .biological_system import BiologicalSystem
from .cohort import Cohort
from .composite_representation import CompositeRepresentation
from .derived_variable import DerivedVariable, augment_with_derived_variables
from .domain import SemanticDomain
from .distribution import Distribution, Normal, PointMass
from .feature import Feature, features_to_array
from .geometry import Geometry, TrajectoryGeometry
from .latent_domain import LatentDomain
from .measurement import Measurement
from .observation import Observable, Observation
from .operator import LongitudinalOperator, Operator
from .phenotype import Phenotype
from .plugin import Plugin
from .process import PhysiologicalProcess, ProcessCoherenceReport, check_process_coherence, project_to_process_space
from .representation import Representation, RepresentationVector
from .representation_space import RepresentationSpace
from .trajectory import Trajectory

__all__ = [
    "BiologicalSystem",
    "Cohort",
    "CompositeRepresentation",
    "Feature",
    "Distribution",
    "Normal",
    "PointMass",
    "features_to_array",
    "Geometry",
    "TrajectoryGeometry",
    "Measurement",
    "Observable",
    "Observation",
    "Operator",
    "LongitudinalOperator",
    "Phenotype",
    "Plugin",
    "PhysiologicalProcess",
    "ProcessCoherenceReport",
    "check_process_coherence",
    "project_to_process_space",
    "DerivedVariable",
    "augment_with_derived_variables",
    "Representation",
    "RepresentationVector",
    "RepresentationSpace",
    "SemanticDomain",
    "LatentDomain",
    "Trajectory",
]
