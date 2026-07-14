"""
biospace.kg_embeddings
==========================

Embeddings de grafo de conhecimento — TransE (Bordes et al., 2013),
aplicado sobre grafos de coorte construídos a partir de
`biospace.ontology` (Paciente, Fenótipo, conceito de terminologia
formal como entidades; "has_phenotype", "measured_abnormal" como
relações).
"""

from __future__ import annotations

from .transe import Triple, TransE

__all__ = ["Triple", "TransE"]
