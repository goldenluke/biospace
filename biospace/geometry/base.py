"""
biospace.geometry.base
=========================

Reexporta Geometry de biospace.core.geometry. A interface abstrata vive
no núcleo (Seção "Arquitetura" do README); este pacote (`biospace.geometry`)
contém apenas IMPLEMENTAÇÕES concretas (Euclidean, Mahalanobis, ...).
"""

from __future__ import annotations

from biospace.core.geometry import Geometry

__all__ = ["Geometry"]
