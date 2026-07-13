"""
biospace.anomaly
====================

Detecção de anomalia (não-supervisionada) sobre RepresentationSpace —
Isolation Forest, Local Outlier Factor, One-Class SVM, ou qualquer
estimador compatível com a API sklearn de detecção de anomalia.
"""

from __future__ import annotations

from .base import OutlierDetector
from .sklearn_detector import SklearnOutlierDetector

__all__ = ["OutlierDetector", "SklearnOutlierDetector"]
