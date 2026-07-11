"""
biospace.core.plugin
=======================

Plugin: interface marcadora/de registro para um módulo específico de
doença (ex.: sleep, diabetes). Também fora da lista original, mas usada
por `biospace.plugins.sleep.SleepPlugin`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Plugin"]


class Plugin(ABC):
    name: str

    @abstractmethod
    def describe(self) -> str:
        """Descrição legível do que este plugin representa."""
        raise NotImplementedError
