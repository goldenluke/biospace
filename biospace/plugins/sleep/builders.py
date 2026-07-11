"""
biospace.plugins.sleep.builders
==================================

Função de conveniência para construir uma Observation manualmente (fora do
fluxo de `loader.py`), útil para testes ou para inserir um novo exame
avulso em um SleepSystem já existente.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from biospace.core import Observation

__all__ = ["exam"]


def exam(values: dict[str, Any], timestamp: Optional[datetime] = None) -> Observation:
    """Constrói uma Observation a partir de um dicionário de valores de um exame completo."""
    return Observation(timestamp=timestamp or datetime.now(), source="exame_completo", values=values)
