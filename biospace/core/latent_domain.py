"""
biospace.core.latent_domain
==============================

LatentDomain (D_L, Seção 6.6 da teoria): um SemanticDomain sem
Observables próprios — não existe operador de observação direto para
ele. Em vez de Measurements próprias, φ_L reconstrói um estado latente a
partir das Features JÁ COMPUTADAS de outros domínios ("domínios-fonte"):

    "φ_L não representa uma projeção, mas uma reconstrução computacional
    de um estado fisiológico não observável." (Seção 6.6)

DISCIPLINA OBRIGATÓRIA (para não virar um "score" com aparência de rigor
estatístico, mas sem nenhuma base real): toda subclasse concreta DEVE
declarar:

  - `hypothesis`: de qual teoria/literatura vem a ideia de que os sinais
    combinados refletem o estado latente proposto. Texto livre, mas
    nunca vazio — se não há nada citável, não é um domínio latente
    legítimo, é um índice inventado vestido de teoria.
  - `is_validated`: False por padrão, e DEVE PERMANECER False a menos
    que exista um DESFECHO INDEPENDENTE medido (não derivado dos mesmos
    sinais de entrada) que confirme a inferência. Correlação com
    severidade/fenótipo da MESMA doença NÃO CONTA como validação — é
    esperada trivialmente, já que os sinais de entrada também compõem a
    própria representação da doença que se está tentando prever.

Mecanismo: `source_domains` são os domínios dos quais o estado latente é
inferido; `infer()` é o φ_L propriamente dito, recebendo as Features já
computadas desses domínios (não Measurements brutas — não há
Observable/Measurement direto para um estado latente).
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

from .domain import SemanticDomain
from .feature import Feature
from .measurement import Measurement

if TYPE_CHECKING:
    from .biological_system import BiologicalSystem

__all__ = ["LatentDomain"]


class LatentDomain(SemanticDomain):
    """
    Ver docstring do módulo. Subclasses DEVEM sobrescrever `hypothesis`
    com uma justificativa concreta e implementar `infer()` — nunca
    `encode()` (não há Measurements próprias a codificar).
    """

    hypothesis: str = ""
    is_validated: bool = False

    def __init__(self, source_domains: Sequence[SemanticDomain]):
        super().__init__(observables=[])
        if not self.hypothesis:
            raise ValueError(
                f"{self.__class__.__name__} deve declarar `hypothesis` com uma justificativa "
                "concreta (teoria/literatura) antes de ser instanciado — um domínio latente sem "
                "hipótese declarada é um índice inventado vestido de teoria, não um φ_L legítimo."
            )
        self.source_domains: list[SemanticDomain] = list(source_domains)

    def collect_source_features(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> dict[str, list[Feature]]:
        """Recolhe as Features já computadas de cada domínio-fonte (não Measurements brutas), respeitando `as_of` (Contrato 5.7)."""
        return {domain.name: domain.transform(system, as_of=as_of) for domain in self.source_domains}

    @abstractmethod
    def infer(self, source_features: dict[str, list[Feature]]) -> list[Feature]:
        """φ_L: infere o(s) Feature(s) latente(s) a partir das Features dos domínios-fonte."""
        raise NotImplementedError

    def encode(self, measurements: dict[str, Measurement]) -> list[Feature]:
        raise NotImplementedError(
            f"{self.__class__.__name__} é um LatentDomain — não implementa encode() a partir de "
            "Measurements (não há Observables próprios). Use transform(system), que invoca "
            "infer() sobre os domínios-fonte."
        )

    def transform(self, system: "BiologicalSystem", as_of: Optional[datetime] = None) -> list[Feature]:
        return self.infer(self.collect_source_features(system, as_of=as_of))

    def __repr__(self) -> str:
        status = "VALIDADO" if self.is_validated else "NÃO VALIDADO (hipótese)"
        return f"{self.__class__.__name__}(domínio latente, {status}, fontes={[d.name for d in self.source_domains]})"
