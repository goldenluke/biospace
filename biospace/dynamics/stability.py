"""
biospace.dynamics.stability
==============================

StabilityOperator: avalia a ESTABILIDADE da dinâmica aprendida por um
EvolutionOperator — quais Features (se alguma) divergem em vez de
reverter à média (|φ| >= 1), e qual o "tempo de meia-vida" de
recuperação para as que são estáveis.

Isso conecta diretamente com o que já existia em `early_warning`
(critical slowing down): lá, mede-se sinais de INSTABILIDADE CRESCENTE
de forma exploratória/heurística, sobre a trajetória de UM paciente por
vez, sem um modelo de dinâmica explícito. Aqui, ajustamos o modelo de
dinâmica DIRETAMENTE sobre toda a coorte, e a estabilidade é uma
propriedade MATEMÁTICA do modelo ajustado (φ >= 1), não um teste
estatístico sobre uma janela deslizante — mais direto, mas também mais
dependente da qualidade do ajuste (poucos pares por Feature => estimativa
de φ pouco confiável; ver `FeatureDynamics.n_pairs`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from biospace.core import Cohort, LongitudinalOperator

from .evolution import FeatureDynamics, MeanRevertingEvolutionOperator

__all__ = ["StabilityReport", "StabilityOperator", "RobustnessReport", "check_feature_stability_robustness"]


@dataclass
class StabilityReport:
    """Resultado da análise de estabilidade sobre a dinâmica ajustada de uma coorte."""

    n_features: int
    n_stable: int
    n_unstable: int
    least_stable: list[tuple[str, float, int]] = field(default_factory=list)  # (nome, phi, n_pairs), piores N
    dynamics: dict[str, FeatureDynamics] = field(default_factory=dict)

    @property
    def is_globally_stable(self) -> bool:
        """Convenção: estável se NENHUMA Feature tiver |φ| >= 1 (nenhum eixo diverge)."""
        return self.n_unstable == 0

    def summary(self) -> str:
        lines = [
            f"Estabilidade da dinâmica: {self.n_stable}/{self.n_features} Features estáveis (|φ| < 1)",
            f"is_globally_stable = {self.is_globally_stable}",
        ]
        if self.least_stable:
            lines.append("Menos estáveis (maior |φ|):")
            for name, phi, n_pairs in self.least_stable:
                flag = "" if abs(phi) < 1.0 else "  <-- DIVERGE"
                lines.append(f"  {name}: φ_dia={phi:.4f} (n_pares={n_pairs}){flag}")
        return "\n".join(lines)


class StabilityOperator(LongitudinalOperator[StabilityReport]):
    """
    Ajusta (ou reaproveita) um `EvolutionOperator` sobre a Cohort e
    resume a estabilidade resultante. `evolution_operator`, se informado,
    é usado (e ajustado) diretamente — permite reaproveitar um ajuste já
    feito, ou trocar a implementação (ex.: um EvolutionOperator não-linear
    futuro), em vez de sempre criar um `MeanRevertingEvolutionOperator` novo.
    """

    def __init__(self, evolution_operator: Optional[MeanRevertingEvolutionOperator] = None, order=None, n_worst: int = 5):
        self.evolution_operator = evolution_operator or MeanRevertingEvolutionOperator(order=order)
        self.n_worst = n_worst

    def analyze(self, cohort: "Cohort") -> StabilityReport:
        self.evolution_operator.fit(cohort)
        dynamics = self.evolution_operator.dynamics_

        n_stable = sum(1 for fd in dynamics.values() if fd.is_stable)
        n_unstable = len(dynamics) - n_stable

        ordenado = sorted(dynamics.items(), key=lambda kv: -abs(kv[1].phi_per_day))
        least_stable = [(name, fd.phi_per_day, fd.n_pairs) for name, fd in ordenado[: self.n_worst]]

        return StabilityReport(
            n_features=len(dynamics),
            n_stable=n_stable,
            n_unstable=n_unstable,
            least_stable=least_stable,
            dynamics=dynamics,
        )

    def describe(self) -> str:
        return "StabilityOperator: avalia estabilidade (taxa de reversão à média por Feature) da dinâmica ajustada sobre a coorte."


@dataclass
class RobustnessReport:
    """
    Resultado de `check_feature_stability_robustness()` — quão sensível
    é a conclusão de estabilidade de UMA Feature à remoção de um único
    paciente (leave-one-patient-out).

    ACHADO REAL que motivou este diagnóstico: `hypoxia.tempo_total_em_hipoxemia_min`
    parecia divergir (φ=1,0039) na coorte real de SAOS — ficou registrado
    como "mereceria investigação dedicada". Investigando: a Feature tem
    distribuição fortemente assimétrica (mediana=0, 75º percentil=0,
    máximo=135) — removendo apenas 2 pacientes (de ~296) com os valores
    mais extremos, φ caiu para 0,664 (estável). A "divergência" não era
    sinal populacional real, era sensibilidade a 2 outliers. Corrigido
    aqui como uma checagem formal, reutilizável para qualquer Feature
    suspeita — não repetir essa investigação manualmente toda vez.
    """

    feature_name: str
    phi_full: float
    is_stable_full: bool
    phi_jackknife: dict[str, float] = field(default_factory=dict)  # system_id removido -> φ reajustado sem ele
    n_patients_tested: int = 0

    @property
    def conclusion_is_robust(self) -> bool:
        """True se a conclusão de estabilidade (estável/instável) NÃO muda ao remover nenhum paciente, individualmente."""
        if not self.phi_jackknife:
            return True
        return all((abs(phi) < 1.0) == self.is_stable_full for phi in self.phi_jackknife.values())

    @property
    def most_influential_patient(self) -> Optional[tuple[str, float]]:
        """(system_id, φ resultante) do paciente cuja remoção MAIS muda φ — None se não houve nenhum teste."""
        if not self.phi_jackknife:
            return None
        sid = max(self.phi_jackknife, key=lambda s: abs(self.phi_jackknife[s] - self.phi_full))
        return sid, self.phi_jackknife[sid]

    def summary(self) -> str:
        lines = [
            f"Robustez da estabilidade de '{self.feature_name}': φ_completo={self.phi_full:.4f} "
            f"({'estável' if self.is_stable_full else 'INSTÁVEL'}), testado removendo {self.n_patients_tested} pacientes individualmente",
            f"Conclusão robusta (não muda ao remover NENHUM paciente)? {self.conclusion_is_robust}",
        ]
        if not self.conclusion_is_robust:
            influente = self.most_influential_patient
            lines.append(
                f"⚠️ Removendo o paciente '{influente[0]}' sozinho, φ vira {influente[1]:.4f} "
                f"({'estável' if abs(influente[1]) < 1.0 else 'INSTÁVEL'}) — a conclusão depende deste paciente."
            )
        return "\n".join(lines)


def check_feature_stability_robustness(
    cohort: "Cohort", feature_name: str, order: Optional[list[str]] = None, max_patients_tested: int = 60
) -> RobustnessReport:
    """
    Leave-one-patient-out sobre UMA Feature: reajusta φ removendo, um de
    cada vez, cada paciente que contribui pares para essa Feature —
    reporta se a conclusão de estabilidade é robusta a essa remoção, ou
    depende de um paciente específico (ver `RobustnessReport` para o
    achado real que motivou isto).

    `max_patients_tested`: em coortes grandes, testar TODOS os pacientes
    individualmente pode ser caro — por padrão, prioriza os pacientes com
    os valores mais EXTREMOS da Feature (mais prováveis de serem
    influentes), até este limite.
    """
    full_operator = MeanRevertingEvolutionOperator(order=order)
    full_operator.fit(cohort)
    if feature_name not in full_operator.dynamics_:
        raise KeyError(f"Feature '{feature_name}' não encontrada na dinâmica ajustada. Disponíveis: {list(full_operator.dynamics_.keys())}")

    fd_full = full_operator.dynamics_[feature_name]
    domain_name, _, short_name = feature_name.partition(".")

    candidatos: list[tuple[str, float]] = []
    for sid, traj in cohort.trajectories.items():
        if len(traj) < 2:
            continue
        valores = []
        for i in range(len(traj)):
            vec = traj.at(i)
            f = next((f for f in vec.components.get(domain_name, []) if f.name == short_name), None)
            if f is not None and not f.is_missing and f.raw_value is not None:
                valores.append(f.raw_value)
        if len(valores) >= 2:
            desvio_extremo = max(abs(v - fd_full.mu) for v in valores)
            candidatos.append((sid, desvio_extremo))

    candidatos.sort(key=lambda kv: -kv[1])
    pacientes_testados = [sid for sid, _ in candidatos[:max_patients_tested]]

    phi_jackknife: dict[str, float] = {}
    for sid_removido in pacientes_testados:
        cohort_sem = Cohort()
        for sid, traj in cohort.trajectories.items():
            if sid == sid_removido:
                continue
            cohort_sem.systems[sid] = cohort.systems[sid]
            cohort_sem.trajectories[sid] = traj

        operador_sem = MeanRevertingEvolutionOperator(order=order)
        operador_sem.fit(cohort_sem)
        if feature_name in operador_sem.dynamics_:
            phi_jackknife[sid_removido] = operador_sem.dynamics_[feature_name].phi_per_day

    return RobustnessReport(
        feature_name=feature_name,
        phi_full=fd_full.phi_per_day,
        is_stable_full=fd_full.is_stable,
        phi_jackknife=phi_jackknife,
        n_patients_tested=len(phi_jackknife),
    )
