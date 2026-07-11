"""
biospace.plugins.sleep.latent
================================

InflammationProxyDomain: primeira implementação concreta de LatentDomain
neste projeto — um "domínio de inflamação sistêmica" inferido a partir
de outros três domínios já existentes (hipoxia, cardiovascular,
antropometria), NÃO observado diretamente (não há biomarcador
inflamatório — PCR, IL-6, leucócitos — nesta planilha).

HIPÓTESE DECLARADA (obrigatória por LatentDomain): a medicina do sono
associa consistentemente três mecanismos à inflamação sistêmica em SAOS:
(1) hipóxia intermitente gera estresse oxidativo e ativação de vias
inflamatórias; (2) tecido adiposo (obesidade/IMC elevado) produz
citocinas pró-inflamatórias (adipocinas); (3) ativação simpática
(frequência cardíaca elevada, menor variabilidade) modula resposta
imune. Os três domínios-fonte escolhidos refletem esses três mecanismos.
Isso é uma hipótese fisiológica plausível, não uma prova — daí a
obrigatoriedade de declará-la explicitamente.

is_validated = False, e DEVE permanecer False: não há, nesta planilha,
nenhum biomarcador inflamatório medido de forma independente contra o
qual validar o fator extraído. Testar se o fator correlaciona com
fenótipo/severidade DE SAOS não validaria nada — os mesmos domínios-fonte
(hipoxia, cardiovascular, antropometria) já alimentam a fenotipagem, então
essa correlação seria esperada por construção, não uma confirmação
independente do significado "inflamatório" do fator.

ACHADO EMPÍRICO (rodado nos dados reais, documentado no README): com
`n_factors=1` (padrão), o único fator extraído é DOMINADO pelo domínio de
hipoxemia (cargas de spo2_media/spo2_minima/tempo_spo2_90 todas > 0,6);
a contribuição cardiovascular é quase nula (|carga| < 0,2). Com
`n_factors=2`, os dois mecanismos se SEPARAM de forma limpa: um fator
dominado por frequência cardíaca, outro por hipoxemia+idade — sugerindo
que "inflamação" como um ÚNICO fator é uma simplificação questionável
nestes dados; pode ser mais honesto tratar como dois eixos latentes
distintos (ex.: "carga hipóxica" e "ativação simpática") do que um
"proxy de inflamação" unificado.

---

FrailtyProxyDomain e AutonomicBalanceProxyDomain: dois domínios latentes
adicionais, cada um com o mesmo nível de escrutínio empírico.

RECUSA DELIBERADA — "CognitiveReserveDomain" NÃO foi implementado.
Reserva cognitiva (Fried et al.; Stern, 2002) é operacionalizada na
literatura por educação, QI pré-mórbido ou neuroimagem — nada disso
existe nesta planilha. Testei, antes de recusar, se um proxy mais
MODESTO ("queixa cognitiva", via `dificuldade_concentracao` +
`perda_memoria`) sobreviveria à Análise Fatorial:

  1. Com hipoxia como domínio-fonte: os dois sintomas cognitivos ficam
     nas posições 13ª e 19ª de 22 (cargas < 0,04) — hipóxia domina
     completamente, os sintomas cognitivos não contribuem quase nada.
  2. Sem hipoxia (só sintomas + antropometria): os sintomas cognitivos
     aparecem (cargas ~0,2-0,3), mas o fator inteiro é dominado por
     `sonolencia_diurna` e `sono_nao_reparador` — ou seja, mede
     "sonolência/sono não reparador geral", não algo especificamente
     cognitivo.

Conclusão: não há, nestes dados, um sinal cognitivo isolável via este
método — nem "reserva" (dado inexistente) nem sequer "queixa cognitiva"
(indistinguível de sonolência geral). Implementar qualquer classe com
"cognitive" no nome aqui seria exatamente o "índice inventado vestido de
teoria" que `LatentDomain.hypothesis` existe para evitar.
"""

from __future__ import annotations

from biospace.latent import FactorAnalysisLatentDomain

from .domains import (
    AnthropometricDomain,
    CardiovascularDomain,
    ComorbidityDomain,
    HypoxiaDomain,
    SleepArchitectureDomain,
    SymptomsDomain,
)

__all__ = ["InflammationProxyDomain", "FrailtyProxyDomain", "AutonomicBalanceProxyDomain"]


class InflammationProxyDomain(FactorAnalysisLatentDomain):
    name = "inflammation_proxy"
    description = (
        "Fator latente extraído de hipoxia + cardiovascular + antropometria — proxy "
        "HIPOTÉTICO de inflamação sistêmica, não validado contra biomarcador independente."
    )
    hypothesis = (
        "Hipóxia intermitente (estresse oxidativo/ativação inflamatória), obesidade "
        "(adipocinas pró-inflamatórias) e ativação simpática (FC elevada) são três "
        "mecanismos consistentemente associados à inflamação sistêmica na literatura de "
        "medicina do sono em SAOS. Este fator é a variação COMPARTILHADA entre os "
        "domínios que operacionalizam esses três mecanismos nesta representação — não "
        "uma medida direta de qualquer biomarcador inflamatório real."
    )
    is_validated = False  # não alterar sem um desfecho inflamatório independente medido
    n_factors = 1

    def __init__(
        self,
        hypoxia: HypoxiaDomain,
        cardiovascular: CardiovascularDomain,
        anthropometric: AnthropometricDomain,
        n_factors: int = 1,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        super().__init__(source_domains=[hypoxia, cardiovascular, anthropometric], random_state=random_state)


class FrailtyProxyDomain(FactorAnalysisLatentDomain):
    """
    Proxy de fragilidade. O fenótipo de fragilidade clássico (Fried et
    al., 2001) tem 5 componentes: perda de peso não intencional,
    exaustão, baixa atividade física, lentidão de marcha e fraqueza de
    preensão. Esta planilha só sustenta, na melhor das hipóteses, o
    componente de EXAUSTÃO (via sintomas) e proxies fracos de
    carga/idade — marcha, preensão e perda de peso não existem aqui.

    ACHADO EMPÍRICO: com `n_factors=1`, o fator é dominado por sintomas
    de exaustão/sono não reparador (sonolencia_diurna +0.31,
    dificuldade_concentracao +0.29, sono_nao_reparador +0.27) — idade e
    comorbidades contribuem quase nada (idade: +0.06; toda comorbidade
    individual < 0.09). Ou seja, na prática, isto mede o componente de
    EXAUSTÃO do fenótipo de fragilidade, não fragilidade multissistêmica.
    """

    name = "frailty_proxy"
    description = (
        "Fator latente de exaustão/carga sintomática — proxy PARCIAL e FRACO de fragilidade "
        "(só cobre o componente de exaustão; marcha, preensão e perda de peso não existem nesta planilha)."
    )
    hypothesis = (
        "O fenótipo de fragilidade (Fried et al., 2001) inclui exaustão como um de seus 5 "
        "componentes; sono fragmentado e comorbidade acumulada são associados a fragilidade na "
        "literatura geriátrica. Aqui, a combinação de sintomas de exaustão, arquitetura do sono e "
        "carga de comorbidades é usada como proxy PARCIAL — sem marcha, preensão ou perda de peso, "
        "não cobre o construto completo."
    )
    is_validated = False
    n_factors = 1

    def __init__(
        self,
        anthropometric: AnthropometricDomain,
        comorbidity: ComorbidityDomain,
        sleep_architecture: SleepArchitectureDomain,
        symptoms: SymptomsDomain,
        n_factors: int = 1,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        super().__init__(
            source_domains=[anthropometric, comorbidity, sleep_architecture, symptoms],
            random_state=random_state,
        )


class AutonomicBalanceProxyDomain(FactorAnalysisLatentDomain):
    """
    Proxy de balanço autonômico (simpático/parassimpático). A literatura
    padrão usa variabilidade de frequência cardíaca (HRV: SDNN, RMSSD) —
    esta planilha não tem HRV real, só fc_minima/media/maxima (proxies
    grosseiros) e a hipóxia intermitente, que ativa o reflexo
    quimiorreceptor e gera surtos simpáticos agudos (mecanismo bem
    estabelecido em SAOS).

    ACHADO EMPÍRICO: com `n_factors=1`, hipoxemia domina completamente
    (mesmo padrão do InflammationProxyDomain) — a contribuição
    cardiovascular é quase nula. Por isso o padrão aqui é `n_factors=2`:
    o Fator 1 (índice 0) fica dominado por FC (média/mínima/máxima —
    o eixo "autonômico" genuíno), o Fator 2 fica dominado por hipoxemia
    (largamente REDUNDANTE com HypoxiaDomain — não traz informação nova).
    Use `top_loadings(factor_index=0)` para confirmar isso a cada ajuste.
    """

    name = "autonomic_balance_proxy"
    description = (
        "Fator(es) latente(s) de FC/hipoxemia — proxy FRACO de balanço autonômico "
        "(sem HRV real). Com n_factors=2, o Fator 1 é o eixo cardiovascular genuíno; "
        "o Fator 2 é redundante com HypoxiaDomain."
    )
    hypothesis = (
        "Hipóxia intermitente ativa o reflexo quimiorreceptor, gerando surtos simpáticos "
        "agudos (mecanismo bem estabelecido em SAOS); frequência cardíaca elevada e maior "
        "amplitude refletem, de forma grosseira, desbalanço simpatovagal. Não há HRV real "
        "(SDNN/RMSSD) nesta planilha — apenas fc_minima/media/maxima como proxies indiretos."
    )
    is_validated = False
    n_factors = 2

    def __init__(
        self,
        cardiovascular: CardiovascularDomain,
        hypoxia: HypoxiaDomain,
        n_factors: int = 2,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        super().__init__(source_domains=[cardiovascular, hypoxia], random_state=random_state)
