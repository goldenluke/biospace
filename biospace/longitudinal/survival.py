"""
biospace.longitudinal.survival
=================================

Análise de sobrevivência (tempo-até-evento) sobre trajetórias de Cohort:
quanto tempo, a partir da primeira observação de cada sistema, até que
uma condição de interesse seja satisfeita pela primeira vez (ex.: entrar
em um fenótipo grave, iniciar um tratamento). Sistemas cuja trajetória
termina sem que a condição ocorra são CENSURADOS (censura à direita) —
tratamento estatístico padrão de dados de sobrevivência, não um dado
"faltante" a ser descartado.

Implementação própria do estimador de Kaplan-Meier (sem dependência de
`lifelines`), suficiente para curvas univariadas — consistente com a
filosofia do projeto de manter o núcleo autocontido.

DISTINÇÃO DE DESIGN vs. `biospace.survival` (registrada explicitamente
numa auditoria do projeto — os dois módulos coexistem de propósito, não
por duplicação não resolvida):

- **Este módulo**: tempo = DIAS DE CALENDÁRIO reais (`timestamp` de
  cada `Observation`). Certo quando os timestamps são genuinamente
  significativos como intervalo — o caso do plugin sleep. Sem Cox
  (só Kaplan-Meier univariado), sem dependência externa.
- **`biospace.survival`** (`build_discrete_time_to_event` +
  `kaplan_meier_by_group`/`fit_cox_model`, usa `lifelines`): tempo =
  ÍNDICE ORDINAL (posição na sequência), suporta Cox com covariáveis.
  Certo quando o timestamp NÃO representa um intervalo real — o caso
  da UCI Diabetes 130-US Hospitals, onde `encounter_id` é só proxy de
  ordem cronológica, não data verdadeira. Usar este módulo (o de
  calendário real) na UCI produziria durações artificiais a partir de
  datas sintéticas (1 dia por encontro, por construção do loader) —
  um erro metodológico real que `biospace.survival` foi desenhado
  para evitar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from biospace.core import LongitudinalOperator

__all__ = ["SurvivalRecord", "KaplanMeierResult", "SurvivalAnalyzer", "SurvivalOperator"]

if TYPE_CHECKING:
    from biospace.core import Cohort, Phenotype, RepresentationVector


@dataclass
class SurvivalRecord:
    """Um registro de sobrevivência: tempo decorrido até o evento (ou até a censura)."""

    system_id: str
    time_days: float
    event_observed: bool  # True = evento ocorreu; False = censurado (não observado até o fim do acompanhamento)


@dataclass
class KaplanMeierResult:
    """Curva de sobrevivência estimada: S(t) em cada tempo de evento observado."""

    time_days: list[float] = field(default_factory=list)
    survival: list[float] = field(default_factory=list)
    at_risk: list[int] = field(default_factory=list)
    n_events: list[int] = field(default_factory=list)
    n_total: int = 0
    n_events_total: int = 0
    n_censored_total: int = 0

    def survival_at(self, t: float) -> float:
        """S(t): probabilidade estimada de o evento ainda não ter ocorrido até o instante t (em dias)."""
        s = 1.0
        for ti, si in zip(self.time_days, self.survival):
            if ti <= t:
                s = si
            else:
                break
        return s

    def median_survival_time(self) -> Optional[float]:
        """Primeiro instante em que S(t) cruza 0.5, ou None se a sobrevivência nunca cair a esse ponto."""
        for ti, si in zip(self.time_days, self.survival):
            if si <= 0.5:
                return ti
        return None


class SurvivalAnalyzer(LongitudinalOperator[KaplanMeierResult]):
    """
    event_fn: dado um RepresentationVector, retorna True se a condição de
    interesse já está satisfeita naquele instante. O "tempo zero" de cada
    sistema é sua primeira observação registrada na trajetória.

    É um `LongitudinalOperator[KaplanMeierResult]` — `fit(cohort)` é um
    alias direto de `kaplan_meier(cohort)`, satisfazendo a hierarquia de
    Operator (Seção "Hierarquia de Operator" do README) sem quebrar o
    nome mais descritivo já usado no restante do projeto/dashboard.
    """

    def __init__(self, event_fn: Callable[["RepresentationVector"], bool]):
        self.event_fn = event_fn

    @classmethod
    def for_phenotype(cls, phenotype: "Phenotype", order: Optional[Sequence[str]] = None) -> "SurvivalAnalyzer":
        """Conveniência: evento = "entrou neste fenótipo pela primeira vez"."""
        return cls(lambda vec: phenotype.contains(vec.as_vector(order)))

    @classmethod
    def for_feature_threshold(
        cls, domain_name: str, feature_name: str, threshold: float = 0.5, above: bool = True
    ) -> "SurvivalAnalyzer":
        """
        Conveniência: evento = "a Feature `feature_name` do domínio `domain_name`
        cruzou `threshold`" (ex.: início de tratamento — TreatmentDomain, feature
        binária 'aam' ou 'cpap' passando a valer 1.0).
        """

        def event_fn(vec: "RepresentationVector") -> bool:
            for f in vec.components.get(domain_name, []):
                if not hasattr(f, "name") or not hasattr(f, "value"):
                    raise TypeError(
                        f"Esperava objetos Feature em RepresentationVector.components[{domain_name!r}], "
                        f"mas encontrou {type(f).__name__} ({f!r}). Isso normalmente indica uma versão "
                        "desatualizada de biospace/plugins/sleep/domains.py: versões anteriores à "
                        "introdução de Measurement/Feature retornavam np.ndarray cru de "
                        "SemanticDomain.encode(), em vez de list[Feature]. Substitua toda a pasta "
                        "biospace/ pela versão mais recente (não copie arquivos individualmente) — "
                        "veja biospace/README.md, seção 'Núcleo dividido por entidade'."
                    )
                if f.name == feature_name:
                    return (f.value >= threshold) if above else (f.value <= threshold)
            return False

        return cls(event_fn)

    def records(self, cohort: "Cohort") -> list[SurvivalRecord]:
        """Constrói um SurvivalRecord por sistema da coorte (evento observado ou censurado)."""
        records: list[SurvivalRecord] = []
        for sid, traj in cohort.trajectories.items():
            if len(traj) == 0:
                continue
            t0 = traj.at(0).timestamp

            event_time = None
            for i in range(len(traj)):
                vec = traj.at(i)
                if self.event_fn(vec):
                    event_time = vec.timestamp
                    break

            if event_time is not None:
                days = (event_time - t0).total_seconds() / 86400
                records.append(SurvivalRecord(system_id=sid, time_days=days, event_observed=True))
            else:
                last_time = traj.latest().timestamp
                days = (last_time - t0).total_seconds() / 86400
                records.append(SurvivalRecord(system_id=sid, time_days=days, event_observed=False))
        return records

    def kaplan_meier(self, cohort: "Cohort") -> KaplanMeierResult:
        """Estima S(t) pelo método de Kaplan-Meier sobre os SurvivalRecords desta coorte."""
        records = self.records(cohort)
        result = KaplanMeierResult(
            n_total=len(records),
            n_events_total=sum(1 for r in records if r.event_observed),
            n_censored_total=sum(1 for r in records if not r.event_observed),
        )

        event_times = sorted({r.time_days for r in records if r.event_observed})
        survival = 1.0
        for t in event_times:
            at_risk = sum(1 for r in records if r.time_days >= t)
            n_events = sum(1 for r in records if r.event_observed and r.time_days == t)
            if at_risk == 0:
                continue
            survival *= 1 - (n_events / at_risk)
            result.time_days.append(t)
            result.survival.append(survival)
            result.at_risk.append(at_risk)
            result.n_events.append(n_events)

        return result

    def fit(self, cohort: "Cohort") -> KaplanMeierResult:
        """Satisfaz a interface LongitudinalOperator — alias direto de `kaplan_meier(cohort)`."""
        return self.kaplan_meier(cohort)

    def describe(self) -> str:
        return "SurvivalAnalyzer: estima S(t) por Kaplan-Meier (tempo até um evento de interesse)."


# Alias: mesma classe, nome alinhado à hierarquia de Operator (ver README).
SurvivalOperator = SurvivalAnalyzer
