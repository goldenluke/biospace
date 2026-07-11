"""
biospace.causal.balance
==========================

check_baseline_balance: o PRIMEIRO passo de qualquer análise causal
observacional séria — antes de estimar qualquer "efeito", verificar se
os grupos comparados (ex.: pacientes que iniciaram um tratamento vs. os
que não iniciaram) já eram DIFERENTES na linha de base, antes do
tratamento. Se estiverem, qualquer diferença de desfecho observada pode
ser CONFUNDIMENTO POR INDICAÇÃO (pacientes mais graves, ou mais
motivados, tendem a iniciar tratamento por razões que também afetam o
desfecho, não por causa do tratamento em si) — não um efeito causal do
tratamento.

Métrica: Differença Padronizada de Médias (Standardized Mean Difference,
SMD) por Feature, calculada nos primeiros exames de cada grupo (antes de
qualquer exposição ao tratamento). |SMD| > 0.1 é o limiar convencional
(Austin, 2009) para "desequilíbrio relevante" na literatura de
epidemiologia observacional.

`system_ids`: filtro opcional para reexecutar a MESMA checagem sobre um
SUBCONJUNTO da coorte — usado por `biospace.causal.propensity` para
comparar o balanceamento antes/depois do pareamento por escore de
propensão, sem duplicar a lógica de coleta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from biospace.core import Cohort

__all__ = ["BalanceReport", "check_baseline_balance"]


@dataclass
class BalanceReport:
    """Resultado do teste de balanceamento de linha de base entre dois grupos."""

    feature_names: list[str]
    smd: dict[str, float] = field(default_factory=dict)
    n_treated: int = 0
    n_untreated: int = 0
    imbalance_threshold: float = 0.1

    @property
    def n_imbalanced(self) -> int:
        return sum(1 for v in self.smd.values() if abs(v) > self.imbalance_threshold)

    @property
    def is_balanced(self) -> bool:
        """
        Convenção: "balanceado" se NENHUMA Feature exceder o limiar de SMD.
        Isso é uma condição necessária, mas NUNCA suficiente, para
        inferência causal válida — confundidores NÃO MEDIDOS (adesão,
        motivação, fatores socioeconômicos, gravidade não capturada pelas
        Features observadas) nunca aparecem nesta checagem, mesmo que o
        grupo esteja perfeitamente balanceado nas variáveis observadas.
        """
        return self.n_imbalanced == 0

    def most_imbalanced(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.smd.items(), key=lambda kv: -abs(kv[1]))[:n]

    def summary(self) -> str:
        lines = [
            f"Balanceamento de linha de base: grupo tratado (n={self.n_treated}) vs. não-tratado (n={self.n_untreated})",
            f"{self.n_imbalanced}/{len(self.feature_names)} Features com |SMD| > {self.imbalance_threshold} (desequilíbrio relevante)",
            f"is_balanced = {self.is_balanced}",
        ]
        if not self.is_balanced:
            lines.append("Mais desequilibradas:")
            for name, smd in self.most_imbalanced(10):
                lines.append(f"  {name}: SMD={smd:+.3f}")
            lines.append(
                "AVISO: desequilíbrio de linha de base é evidência de CONFUNDIMENTO POR INDICAÇÃO — "
                "qualquer diferença de desfecho observada entre os grupos pode refletir essas "
                "diferenças pré-existentes, não um efeito causal do tratamento."
            )
        return "\n".join(lines)


def _collect_baseline(
    cohort: "Cohort", treatment_domain: str, treatment_feature: str, order: Optional[Sequence[str]] = None
) -> tuple[dict[str, np.ndarray], set[str], list[str]]:
    """
    Núcleo de coleta compartilhado entre `check_baseline_balance()` e
    `biospace.causal.propensity` — para os dois nunca divergirem em como
    definem "tratado" ou "linha de base".

    A própria Feature de tratamento (`treatment_domain.treatment_feature`)
    é EXCLUÍDA de `feature_names` e dos vetores de linha de base —
    comparar/prever o tratamento usando o tratamento como preditor é
    autoinclusão, não confundimento. ACHADO REAL que motivou esta
    correção: em dado TRANSVERSAL (um único ponto por paciente — ex.:
    NHANES), a Feature de tratamento no "baseline" (`traj.at(0)`, a
    única observação disponível) já revela o próprio rótulo de grupo,
    fazendo `check_baseline_balance` mascarar o desequilíbrio máximo
    como SMD=0 (variância zero dentro de cada grupo — tratados=todos
    1.0, não-tratados=todos 0.0 — quebra a fórmula de SMD por
    divisão por pooled_std≈0) e fazendo `estimate_propensity` treinar
    um modelo que prevê o tratamento a partir de si mesmo (AUC=1.0
    trivial, não confundimento genuíno). Em dado LONGITUDINAL onde
    `traj.at(0)` antecede genuinamente a adoção do tratamento (ex.:
    SAOS/AAM), essa autoinclusão não causava sintoma visível porque o
    valor na linha de base já era ~0 para todo mundo — mas era
    conceitualmente incorreta de qualquer forma, agora corrigida na
    origem para as duas situações.

    Retorna (vetores de linha de base por system_id, ids que eventualmente
    são tratados, nomes de Feature — SEM a própria Feature de tratamento).
    """
    baseline_vectors: dict[str, np.ndarray] = {}
    treated_ids: set[str] = set()
    feature_names: list[str] = []
    treatment_qualified_name = f"{treatment_domain}.{treatment_feature}"

    for sid, traj in cohort.trajectories.items():
        n = len(traj)
        if n == 0:
            continue

        ever_treated = False
        for i in range(n):
            vec = traj.at(i)
            for f in vec.components.get(treatment_domain, []):
                if f.name == treatment_feature and f.value >= 0.5:
                    ever_treated = True
                    break
            if ever_treated:
                break

        baseline_vec = traj.at(0)
        if not feature_names:
            names = []
            domain_order = order or sorted(baseline_vec.components.keys())
            for domain_name in domain_order:
                for f in baseline_vec.components[domain_name]:
                    qualified = f"{domain_name}.{f.name}"
                    if qualified != treatment_qualified_name:
                        names.append(qualified)
            feature_names = names

        vetor_completo = baseline_vec.as_vector(order)
        indices_manter = [i for i, nome in enumerate(_qualified_names(baseline_vec, order)) if nome != treatment_qualified_name]
        baseline_vectors[sid] = vetor_completo[indices_manter]
        if ever_treated:
            treated_ids.add(sid)

    return baseline_vectors, treated_ids, feature_names


def _qualified_names(vector, order: Optional[Sequence[str]] = None) -> list[str]:
    """Nomes 'dominio.feature' na MESMA ordem que `RepresentationVector.as_vector(order)` produz — necessário para filtrar por índice de forma consistente."""
    domain_order = order or sorted(vector.components.keys())
    names = []
    for domain_name in domain_order:
        for f in vector.components[domain_name]:
            names.append(f"{domain_name}.{f.name}")
    return names


def check_baseline_balance(
    cohort: "Cohort",
    treatment_domain: str,
    treatment_feature: str,
    order: Optional[Sequence[str]] = None,
    imbalance_threshold: float = 0.1,
    system_ids: Optional[set[str]] = None,
) -> BalanceReport:
    """
    Compara o PRIMEIRO exame de pacientes que eventualmente iniciam o
    tratamento (`treatment_domain.treatment_feature` chega a valer >= 0.5
    em algum ponto da trajetória) contra os que NUNCA iniciam, usando
    SEMPRE o estado ANTES de qualquer exposição (o primeiro exame de cada
    paciente) — para não contaminar a comparação com o próprio efeito do
    tratamento já em curso.

    `system_ids`: se informado, restringe a comparação a este subconjunto
    de pacientes — usado para reexecutar a MESMA checagem após um
    pareamento por escore de propensão (`biospace.causal.propensity`),
    comparando o desequilíbrio antes/depois sobre a mesma métrica.
    """
    baseline_vectors, treated_ids, feature_names = _collect_baseline(cohort, treatment_domain, treatment_feature, order)

    if system_ids is not None:
        baseline_vectors = {sid: v for sid, v in baseline_vectors.items() if sid in system_ids}

    treated_baseline = [v for sid, v in baseline_vectors.items() if sid in treated_ids]
    untreated_baseline = [v for sid, v in baseline_vectors.items() if sid not in treated_ids]

    if len(treated_baseline) < 2 or len(untreated_baseline) < 2:
        raise ValueError(
            f"Poucos pacientes para comparar (tratados={len(treated_baseline)}, "
            f"não-tratados={len(untreated_baseline)}) — precisa de pelo menos 2 em cada grupo."
        )

    treated_matrix = np.stack(treated_baseline)
    untreated_matrix = np.stack(untreated_baseline)

    smd: dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        t_vals = treated_matrix[:, idx]
        u_vals = untreated_matrix[:, idx]
        pooled_std = np.sqrt((np.var(t_vals, ddof=1) + np.var(u_vals, ddof=1)) / 2)
        smd[name] = float((np.mean(t_vals) - np.mean(u_vals)) / pooled_std) if pooled_std > 1e-9 else 0.0

    return BalanceReport(
        feature_names=feature_names,
        smd=smd,
        n_treated=len(treated_baseline),
        n_untreated=len(untreated_baseline),
        imbalance_threshold=imbalance_threshold,
    )
