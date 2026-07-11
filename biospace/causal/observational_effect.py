"""
biospace.causal.observational_effect
=======================================

ObservationalEffectEstimator: estima, a partir de transições REAIS 0->1
observadas na própria coorte (ex.: pacientes que começaram a usar AAM
entre dois exames), o vetor de mudança médio associado ao início do
tratamento.

AVISO CENTRAL (leia antes de usar): isto é uma ASSOCIAÇÃO OBSERVACIONAL,
NÃO um efeito causal validado. Sem randomização (ninguém foi sorteado
para tratar ou não) e sem uma estratégia de identificação causal (ajuste
por confundidores via grafo causal explícito), a diferença observada
antes/depois pode refletir:

  - CONFUNDIMENTO POR INDICAÇÃO: pacientes que iniciam tratamento podem
    ser sistematicamente diferentes dos que não iniciam (mais graves,
    mais motivados, com mais acesso a cuidado) — ver `check_baseline_balance()`,
    que DEVE ser rodado antes de interpretar qualquer resultado daqui.
  - REGRESSÃO À MÉDIA: pacientes tendem a procurar/iniciar tratamento
    quando estão em um pico de sintomas — parte de qualquer melhora
    subsequente pode ser só flutuação natural, não efeito do tratamento.
  - CONFUNDIDORES NÃO MEDIDOS: adesão, fatores socioeconômicos, outras
    mudanças de estilo de vida concomitantes — nada disso está nesta
    planilha.

Este estimador ainda assim tem valor: é o melhor sinal ASSOCIACIONAL
disponível nestes dados, e formaliza (com o aviso explícito) o que já
estava implícito em várias análises anteriores deste projeto. Não
implementa ajuste por escore de propensão (propensity score) — ver
`README.md` para essa limitação registrada como próximo passo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .balance import BalanceReport, check_baseline_balance

if TYPE_CHECKING:
    from biospace.core import Cohort

__all__ = ["ObservationalEffectReport", "ObservationalEffectEstimator"]


@dataclass
class ObservationalEffectReport:
    feature_names: list[str]
    delta_mean: dict[str, float] = field(default_factory=dict)  # media (depois - antes) por feature
    delta_std: dict[str, float] = field(default_factory=dict)
    n_transitions: int = 0
    balance: Optional[BalanceReport] = None

    def top_changes(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.delta_mean.items(), key=lambda kv: -abs(kv[1]))[:n]

    def summary(self) -> str:
        lines = [
            f"Efeito OBSERVACIONAL (não causal) estimado sobre {self.n_transitions} transições reais 0->1",
        ]
        if self.balance is not None:
            lines.append(f"  Balanceamento de linha de base: is_balanced={self.balance.is_balanced} "
                         f"({self.balance.n_imbalanced}/{len(self.balance.feature_names)} Features desequilibradas)")
            if not self.balance.is_balanced:
                lines.append("  *** CUIDADO: grupos desequilibrados na linha de base — efeito abaixo pode ser confundido ***")
        lines.append("Maiores mudanças médias associadas ao início do tratamento:")
        for name, delta in self.top_changes(10):
            lines.append(f"  {name}: Δ={delta:+.3f} (±{self.delta_std.get(name, 0):.3f})")
        return "\n".join(lines)


class ObservationalEffectEstimator:
    """
    `treatment_domain`/`treatment_feature`: qual Feature marca o início
    do tratamento (ex.: domain="treatment", feature="aam").
    """

    def __init__(self, treatment_domain: str, treatment_feature: str, order: Optional[Sequence[str]] = None):
        self.treatment_domain = treatment_domain
        self.treatment_feature = treatment_feature
        self.order = order

    def _find_transitions(self, cohort: "Cohort") -> list[tuple[np.ndarray, np.ndarray]]:
        """Retorna pares (vetor_antes, vetor_depois) para cada transição 0->1 real encontrada na coorte."""
        pairs = []
        for traj in cohort.trajectories.values():
            n = len(traj)
            for i in range(n - 1):
                vec_before = traj.at(i)
                vec_after = traj.at(i + 1)
                before_treated = any(
                    f.value >= 0.5
                    for f in vec_before.components.get(self.treatment_domain, [])
                    if f.name == self.treatment_feature
                )
                after_treated = any(
                    f.value >= 0.5
                    for f in vec_after.components.get(self.treatment_domain, [])
                    if f.name == self.treatment_feature
                )
                if not before_treated and after_treated:
                    pairs.append((vec_before.as_vector(self.order), vec_after.as_vector(self.order)))
        return pairs

    def estimate(self, cohort: "Cohort", run_balance_check: bool = True) -> ObservationalEffectReport:
        """
        Estima o Δ médio (depois - antes) sobre todas as transições 0->1
        reais encontradas na coorte. Roda `check_baseline_balance()`
        automaticamente (a menos que `run_balance_check=False`) e anexa
        o resultado ao relatório — não é opcional por acidente: o
        objetivo é dificultar interpretar o efeito sem ver o aviso.
        """
        pairs = self._find_transitions(cohort)
        if not pairs:
            raise ValueError(
                f"Nenhuma transição 0->1 real de '{self.treatment_domain}.{self.treatment_feature}' "
                "encontrada na coorte — não há como estimar um efeito observacional sem exemplos reais."
            )

        before = np.stack([p[0] for p in pairs])
        after = np.stack([p[1] for p in pairs])
        deltas = after - before

        any_traj = next(t for t in cohort.trajectories.values() if len(t) > 0)
        domain_order = self.order or sorted(any_traj.at(0).components.keys())
        feature_names = []
        for domain_name in domain_order:
            for f in any_traj.at(0).components[domain_name]:
                feature_names.append(f"{domain_name}.{f.name}")

        delta_mean = {name: float(np.mean(deltas[:, i])) for i, name in enumerate(feature_names)}
        delta_std = {name: float(np.std(deltas[:, i])) for i, name in enumerate(feature_names)}

        balance = None
        if run_balance_check:
            balance = check_baseline_balance(cohort, self.treatment_domain, self.treatment_feature, order=self.order)

        return ObservationalEffectReport(
            feature_names=feature_names,
            delta_mean=delta_mean,
            delta_std=delta_std,
            n_transitions=len(pairs),
            balance=balance,
        )
