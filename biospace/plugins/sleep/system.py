"""
biospace.plugins.sleep.system
================================

Especialização de BiologicalSystem para medicina do sono. Note que a
classe em si não adiciona nenhum campo — a especialização real acontece
na Representation (conjunto de domínios) e nos Observables.
"""

from __future__ import annotations

from biospace.core import BiologicalSystem

__all__ = ["SleepSystem"]


class SleepSystem(BiologicalSystem):
    """Sistema biológico especializado para pacientes em investigação/tratamento de SAOS."""
