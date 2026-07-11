"""
biospace.plugins.metabolic.system
====================================

MetabolicSystem: especialização mínima de BiologicalSystem — o núcleo
não sabe que "sistema metabólico" existe; esta classe só dá um nome
concreto ao tipo de sistema biológico deste pacote (mesmo papel de
SleepSystem no plugin sleep). Nenhuma doença aparece aqui: um
MetabolicSystem é um sistema endócrino-metabólico, o mesmo objeto quer
o paciente tenha diabetes, síndrome metabólica, as duas coisas, ou
nenhuma — a doença é uma interpretação aplicada depois (ver
`interpretations.py`), nunca uma propriedade do sistema em si.
"""

from __future__ import annotations

from biospace.core import BiologicalSystem

__all__ = ["MetabolicSystem"]


class MetabolicSystem(BiologicalSystem):
    pass
