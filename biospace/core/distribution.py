"""
biospace.core.distribution
=============================

Distribution: em vez de uma Observation carregar um valor pontual
(`values={"ido": 82}`), pode carregar uma distribuição de probabilidade
sobre o valor real (`values={"ido": Normal(82, 2)}`) — a biblioteca
passa a representar incerteza de medição explicitamente, não apenas o
valor mais provável.

Isso propaga por toda a cadeia — Measurement carrega a distribuição
(`Measurement.distribution`), e `SemanticDomain.encode()` pode propagar
a incerteza algebricamente até `Feature.uncertainty` (ver
`_zscore_features` no plugin sleep para um exemplo de propagação
através de uma transformação linear).

RETROCOMPATIBILIDADE TOTAL: uma Observation sem distribuição continua
funcionando exatamente como antes — `Measurement.distribution` é `None`,
`Measurement.uncertainty` é `0.0`, `Feature.uncertainty` é `None`. Nada
do que já existia precisa mudar para continuar funcionando.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

__all__ = ["Distribution", "Normal", "PointMass"]


class Distribution(ABC):
    """Interface mínima: qualquer distribuição expõe `mean` e `std` (desvio padrão)."""

    mean: float
    std: float

    def sample(self, rng) -> float:
        """
        Amostra um valor da distribuição. Implementação padrão assume
        Normal(mean, std); subclasses com outra forma devem sobrescrever.
        """
        if self.std <= 0:
            return self.mean
        return float(rng.normal(self.mean, self.std))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


@dataclass(frozen=True)
class Normal(Distribution):
    """Distribuição normal (Gaussiana) — o modelo padrão de incerteza de medição instrumental."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(
                f"Normal(mean={self.mean}, std={self.std}) — desvio padrão não pode ser negativo. "
                "Um std negativo não tem significado estatístico e só quebraria mais tarde, de forma "
                "confusa (ex.: dentro de numpy.random.normal), longe de onde foi realmente construído."
            )


@dataclass(frozen=True)
class PointMass(Distribution):
    """
    Distribuição degenerada (sem incerteza) — permite tratar um valor
    pontual uniformemente como uma Distribution quando conveniente
    (`std` sempre 0.0). Não é necessário usar isso para valores pontuais
    comuns — `Observation.values` aceita floats crus normalmente; esta
    classe existe só para quem quiser tratar tudo como Distribution de
    forma homogênea.
    """

    mean: float
    std: float = 0.0

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(f"PointMass(mean={self.mean}, std={self.std}) — desvio padrão não pode ser negativo.")
