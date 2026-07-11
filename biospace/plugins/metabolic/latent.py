"""
biospace.plugins.metabolic.latent
===================================

InsulinResistanceProxyDomain: primeiro domínio latente do pacote
metabólico — infere um fator a partir de GlycemicDomain + AnthropometricDomain.

HIPÓTESE DECLARADA (obrigatória por LatentDomain): resistência à insulina
está mecanisticamente associada tanto à disfunção glicêmica (glicemia de
jejum e HbA1c elevadas) quanto à adiposidade central (IMC e
circunferência abdominal) — o raciocínio por trás de índices clínicos
como o HOMA-IR, que combina glicemia e insulina de jejum. Aqui não temos
insulina de jejum medida (não incluída neste toy), então isto é, na
melhor das hipóteses, um proxy mais fraco que o HOMA-IR de verdade.

is_validated = False: sem HOMA-IR real ou clamp euglicêmico (o
padrão-ouro) nesta base sintética para confirmar que o fator extraído
mede resistência à insulina de fato.
"""

from __future__ import annotations

from biospace.latent import FactorAnalysisLatentDomain

from .domains import AnthropometricDomain, GlycemicDomain

__all__ = ["InsulinResistanceProxyDomain"]


class InsulinResistanceProxyDomain(FactorAnalysisLatentDomain):
    name = "insulin_resistance_proxy"
    description = (
        "Fator latente extraído de controle glicêmico + adiposidade central — proxy HIPOTÉTICO de "
        "resistência à insulina, não validado (sem HOMA-IR ou clamp euglicêmico nesta base)."
    )
    hypothesis = (
        "Resistência à insulina está mecanisticamente associada tanto à disfunção glicêmica "
        "(glicemia de jejum, HbA1c elevadas) quanto à adiposidade central (IMC, circunferência "
        "abdominal) — o mesmo raciocínio por trás do HOMA-IR clínico. Sem insulina de jejum medida "
        "nesta base, este fator é um proxy mais fraco que o HOMA-IR real."
    )
    is_validated = False
    n_factors = 1

    def __init__(self, glycemic: GlycemicDomain, anthropometric: AnthropometricDomain, n_factors: int = 1, random_state: int = 42):
        self.n_factors = n_factors
        super().__init__(source_domains=[glycemic, anthropometric], random_state=random_state)
