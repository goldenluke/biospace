"""
biospace.topology
====================

Análise Topológica de Dados (TDA) sobre RepresentationSpace: Mapper
(resumo em grafo, sensível à lente escolhida) e homologia persistente
(números de Betti, diagramas de persistência, via ripser).
"""

from __future__ import annotations

from .mapper import MapperResult, compute_mapper_graph
from .persistence import PersistenceResult, compute_persistence

__all__ = ["MapperResult", "compute_mapper_graph", "PersistenceResult", "compute_persistence"]
