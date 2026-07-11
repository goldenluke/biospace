"""
biospace.core.entities
=========================

MÓDULO DEPRECIADO — mantido apenas como shim de compatibilidade.

O núcleo foi dividido em arquivos individuais por entidade (Seção
"Arquitetura" do README): biological_system.py, observation.py,
measurement.py, feature.py, domain.py, representation.py,
representation_space.py, geometry.py, trajectory.py, phenotype.py,
cohort.py, operator.py, plugin.py.

Este módulo apenas reexporta tudo, para que código antigo que faça
`from biospace.core.entities import X` continue funcionando. Código novo
deve importar de `biospace.core` diretamente ou do submódulo específico
(ex.: `from biospace.core.feature import Feature`).
"""

from __future__ import annotations

from .biological_system import BiologicalSystem
from .cohort import Cohort
from .composite_representation import CompositeRepresentation
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
    "Representation",
    "RepresentationVector",
    "RepresentationSpace",
    "SemanticDomain",
    "LatentDomain",
    "Trajectory",
]
