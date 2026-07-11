"""
biospace.plugins.metabolic.builders
=====================================

exam(): conveniência para construir uma Observation de um exame
metabólico, mesma lógica de `biospace.plugins.sleep.builders.exam`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from biospace.core import Observation

__all__ = ["exam"]


def exam(values: dict[str, Any], timestamp: Optional[datetime] = None, source: str = "exame_metabolico") -> Observation:
    return Observation(timestamp=timestamp or datetime.now(), source=source, values=values)
