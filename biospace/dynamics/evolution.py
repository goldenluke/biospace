"""
biospace.dynamics.evolution
==============================

EvolutionOperator: aprende como o ESTADO do sistema evolui
ESPONTANEAMENTE no tempo —

    Trajectory -> EvolutionOperator -> Future State

— a partir de pares (x_t, x_{t+Δt}) observados na própria Cohort.
Diferente de `InterventionOperator` (Seção 12.3 da teoria: efeito de UMA
intervenção terapêutica específica, τ aplicado sob demanda a um ponto),
EvolutionOperator aprende a dinâmica NATURAL — progressão espontânea,
sem assumir nenhuma intervenção — a partir de dados reais.

NOTA DE NOMENCLATURA: por que não se chama "TransitionOperator" — esse
nome já existe em `biospace.longitudinal` (`TransitionAnalyzer` /
`TransitionOperator`), para a matriz de transição entre FENÓTIPOS
DISCRETOS (P(fenótipo_j | fenótipo_i)). `EvolutionOperator` opera sobre
o ESTADO CONTÍNUO (o vetor de representação inteiro, feature a feature),
não sobre rótulos categóricos — um conceito relacionado, mas distinto;
nome diferente de propósito, para não colidir nem confundir os dois.

MeanRevertingEvolutionOperator: implementação concreta — um processo de
Ornstein-Uhlenbeck discreto (AR(1) contínuo no tempo), ajustado
INDEPENDENTEMENTE por Feature, sobre TODOS os pares consecutivos
(x_t, x_{t+Δt}, Δt) de TODAS as trajetórias da coorte:

    x_i(t+Δt) ≈ μ_i + φ_i^Δt · (x_i(t) - μ_i)

onde μ_i é a média populacional da Feature (equilíbrio de longo prazo) e
φ_i é a taxa de contração DIÁRIA: |φ_i| < 1 => a Feature reverte à média
(ESTÁVEL, perturbações se dissipam); |φ_i| >= 1 => diverge (INSTÁVEL).
φ_i é estimado por regressão log-linear através da origem sobre os pares
com desvio inicial não-trivial e do mesmo lado da média (ver
`_fit_mean_reversion`) — o MESMO princípio já usado como modelo nulo em
`early_warning.critical_slowing_down._fit_ar1`, aqui ajustado sobre
dados reais para PREVER e avaliar ESTABILIDADE, não para gerar
substitutos.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from biospace.core import LongitudinalOperator

if TYPE_CHECKING:
    from biospace.core import Cohort

__all__ = ["FeatureDynamics", "EvolutionOperator", "MeanRevertingEvolutionOperator"]


@dataclass
class FeatureDynamics:
    """Parâmetros ajustados do modelo de reversão à média para UMA Feature."""

    name: str
    mu: float  # média populacional (equilíbrio de longo prazo)
    phi_per_day: float  # taxa de contração diária
    n_pairs: int  # nº de pares (x_t, x_t+Δt) usados no ajuste — julgue a confiabilidade por isto
    residual_std: float  # desvio padrão do resíduo NA ESCALA DE mean_dt_days (não "1 dia" — ver sigma_eps_per_day)
    mean_dt_days: float = 1.0  # Δt médio dos pares usados no ajuste — necessário para escalar residual_std corretamente

    @property
    def is_stable(self) -> bool:
        return abs(self.phi_per_day) < 1.0

    @property
    def half_life_days(self) -> Optional[float]:
        """Dias para metade do desvio da média se dissipar; None se instável ou φ<=0 (sem decaimento monotônico)."""
        if self.phi_per_day <= 0 or self.phi_per_day >= 1:
            return None
        return float(np.log(0.5) / np.log(self.phi_per_day))

    @property
    def sigma_eps_per_day(self) -> float:
        """
        Ruído de UM DIA (escala "por dia", consistente com `phi_per_day`)
        — NÃO é o mesmo que `residual_std` (que está na escala de
        `mean_dt_days`, o Δt médio observado nos pares usados no ajuste,
        tipicamente dezenas a centenas de dias em dados clínicos reais).

        ACHADO REAL (Fase 9 — Simulação): a primeira versão de `sample()`
        usava `residual_std` diretamente como se já fosse a escala de 1
        dia — testado em processo de Ornstein-Uhlenbeck sintético com
        variância estacionária CONHECIDA (target_var=4.0), a simulação em
        conjunto (`DigitalTwin.simulate_ensemble`) convergiu para uma
        variância ~7x maior que a verdadeira. Corrigido invertendo a
        relação de variância estacionária:

            residual_std² = σ_dia² · (1 - φ^(2·mean_dt)) / (1 - φ²)
            =>  σ_dia = residual_std / sqrt((1 - φ^(2·mean_dt)) / (1 - φ²))

        Sem variância estacionária finita (|φ|>=1) ou mean_dt<=0, cai de
        volta para `residual_std` diretamente, documentando a
        aproximação em vez de propagar um NaN silencioso.
        """
        if self.residual_std <= 0:
            return 0.0
        phi = self.phi_per_day
        if abs(phi) >= 1.0 or self.mean_dt_days <= 0:
            return self.residual_std
        denom = (1 - phi ** (2 * self.mean_dt_days)) / (1 - phi**2)
        if denom <= 1e-12:
            return self.residual_std
        return float(self.residual_std / np.sqrt(denom))

    @property
    def curvature(self) -> Optional[float]:
        """
        Curvatura do poço de potencial ao redor do equilíbrio (Fase 8 —
        Geometria): para um processo de Ornstein-Uhlenbeck discreto
        x(t+Δt) = μ + φ^Δt·(x(t)-μ), φ_per_day = exp(-k), onde k é a
        constante de mola (curvatura) do potencial quadrático
        U(x) = (k/2)(x-μ)² cujo gradiente gera essa dinâmica. Logo,
        k = -ln(φ_per_day) — DIRETAMENTE do que `MeanRevertingEvolutionOperator`
        já ajusta, sem nenhuma estimativa nova. `None` se φ<=0 (sem
        interpretação de poço simples nesse caso — oscilação ou processo
        degenerado).

        Curvatura ALTA (k grande, φ perto de 0) = poço estreito e
        profundo = ESTÁVEL, recuperação rápida = alta resiliência.
        Curvatura BAIXA (k perto de 0, φ perto de 1) = poço raso e largo
        = recuperação lenta = "critical slowing down" = baixa resiliência.
        """
        if self.phi_per_day <= 0:
            return None
        return float(-np.log(self.phi_per_day))

    @property
    def resilience_score(self) -> Optional[float]:
        """Alias direto de `curvature` — a mesma quantidade, nomeada pelo que ela SIGNIFICA (não uma estimativa independente)."""
        return self.curvature

    def __repr__(self) -> str:
        flag = "estável" if self.is_stable else "INSTÁVEL"
        return f"FeatureDynamics({self.name}: φ={self.phi_per_day:.4f} [{flag}], n_pairs={self.n_pairs})"


def _fit_mean_reversion(pairs: list[tuple[float, float, float]]) -> tuple[float, float, int, float, float]:
    """
    Ajusta μ (média) e φ_per_day (taxa de contração diária) sobre uma
    lista de pares (x_t, x_{t+Δt}, Δt_dias), por MÍNIMOS QUADRADOS NÃO
    LINEARES conjuntos sobre (μ, log φ) — minimiza

        Σ (x_{t+Δt} - [μ + φ^Δt · (x_t - μ)])²

    diretamente, sem descartar nenhum par. Retorna (mu, phi_per_day,
    n_pares_usados, residual_std, mean_dt_days).

    NOTA HISTÓRICA (deixada de propósito — ver README): a primeira versão
    usava uma regressão log-linear através da origem, restrita a pares
    com razão positiva (mesmo lado da média). Testado em dados sintéticos
    com φ real conhecido, essa abordagem mostrou um viés SISTEMÁTICO (não
    de amostra pequena — persistia com milhares de pares) empurrando φ
    para perto de 1 mesmo quando o processo verdadeiro revertia à média
    claramente: o filtro "mesmo lado" descarta desproporcionalmente os
    pares com MAIS decaimento (onde o ruído mais facilmente inverte o
    sinal do desvio), enviesando a amostra sobrevivente para MENOS
    decaimento aparente. Mínimos quadrados não lineares sobre todos os
    pares, sem esse filtro, corrige isso.

    `mean_dt_days` (Δt médio dos pares usados neste ajuste) é retornado
    explicitamente porque `residual_std` é o desvio padrão do resíduo
    NESSA escala de Δt, não numa escala de "1 dia" — ver
    `FeatureDynamics.sigma_eps_per_day` para o porquê disso importa (Fase
    9 — Simulação estocástica).
    """
    if len(pairs) < 3:
        xs_fallback = [p[0] for p in pairs] + [p[1] for p in pairs]
        mu = float(np.mean(xs_fallback)) if xs_fallback else 0.0
        mean_dt = float(np.mean([p[2] for p in pairs])) if pairs else 1.0
        return mu, 1.0, 0, 0.0, mean_dt

    from scipy.optimize import minimize

    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    dts = np.array([p[2] for p in pairs])

    def _loss(params: np.ndarray) -> float:
        mu, log_phi = params
        exponent = np.clip(log_phi * dts, -50.0, 50.0)  # evita overflow em exp() para chutes extremos do otimizador
        predicted = mu + np.exp(exponent) * (xs - mu)
        return float(np.sum((ys - predicted) ** 2))

    mu0 = float(np.mean(np.concatenate([xs, ys])))
    # Múltiplos pontos de partida para log(φ) — evita mínimos locais em
    # amostras pequenas/ruidosas (log_phi=0 => φ=1; -0.1/+0.1 cobrem
    # decaimento/crescimento moderados como chute inicial alternativo).
    best_result = None
    for log_phi0 in (0.0, -0.1, 0.1, -0.3, 0.3):
        result = minimize(_loss, x0=np.array([mu0, log_phi0]), method="Nelder-Mead")
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    mu_fit, log_phi_fit = best_result.x
    phi_per_day = float(np.exp(log_phi_fit))

    predicted = mu_fit + (phi_per_day**dts) * (xs - mu_fit)
    residuals = ys - predicted
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    mean_dt_days = float(np.mean(dts))

    return float(mu_fit), phi_per_day, len(pairs), residual_std, mean_dt_days


class EvolutionOperator(LongitudinalOperator["dict[str, FeatureDynamics]"]):
    """Interface: aprende x(t+Δt) a partir de x(t), ajustado sobre uma Cohort inteira."""

    @abstractmethod
    def predict(self, x: np.ndarray, delta_t_days: float) -> np.ndarray:
        """Prevê o estado Δt dias no futuro, a partir do ponto atual `x` (mesmo espaço/ordem do ajuste)."""
        raise NotImplementedError


class MeanRevertingEvolutionOperator(EvolutionOperator):
    """Ver docstring do módulo para o modelo (Ornstein-Uhlenbeck discreto por Feature)."""

    def __init__(self, order: Optional[Sequence[str]] = None):
        self.order = order
        self.dynamics_: dict[str, FeatureDynamics] = {}
        self.is_fitted = False

    def _feature_names(self, cohort: "Cohort") -> list[str]:
        for traj in cohort.trajectories.values():
            if len(traj) >= 1:
                vec = traj.at(0)
                order = self.order or sorted(vec.components.keys())
                names = []
                for domain_name in order:
                    for f in vec.components[domain_name]:
                        names.append(f"{domain_name}.{f.name}")
                return names
        return []

    def _collect_pairs_by_index(self, cohort: "Cohort") -> dict[int, list[tuple[float, float, float]]]:
        pairs_by_index: dict[int, list[tuple[float, float, float]]] = {}
        for traj in cohort.trajectories.values():
            n = len(traj)
            if n < 2:
                continue
            for i in range(n - 1):
                v_t = traj.at(i)
                v_t1 = traj.at(i + 1)
                dt = (v_t1.timestamp - v_t.timestamp).total_seconds() / 86400
                if dt <= 0:
                    continue
                x_t = v_t.as_vector(self.order)
                x_t1 = v_t1.as_vector(self.order)
                for idx in range(len(x_t)):
                    pairs_by_index.setdefault(idx, []).append((float(x_t[idx]), float(x_t1[idx]), dt))
        return pairs_by_index

    def fit(self, cohort: "Cohort") -> "MeanRevertingEvolutionOperator":
        """Ajusta um modelo de reversão à média por Feature, sobre todos os pares consecutivos da coorte."""
        pairs_by_index = self._collect_pairs_by_index(cohort)
        if not pairs_by_index:
            raise ValueError(
                "Nenhum par consecutivo (paciente com >= 2 exames) encontrado na coorte para ajustar a dinâmica."
            )

        names = self._feature_names(cohort)
        self.dynamics_ = {}
        for idx, pairs in pairs_by_index.items():
            mu, phi, n_pairs, resid, mean_dt = _fit_mean_reversion(pairs)
            name = names[idx] if idx < len(names) else f"feature_{idx}"
            self.dynamics_[name] = FeatureDynamics(name=name, mu=mu, phi_per_day=phi, n_pairs=n_pairs, residual_std=resid, mean_dt_days=mean_dt)
        self.is_fitted = True
        return self

    def analyze(self, cohort: "Cohort") -> dict[str, FeatureDynamics]:
        """Alias de `fit()` para satisfazer a interface `LongitudinalOperator`."""
        self.fit(cohort)
        return self.dynamics_

    def predict(self, x: np.ndarray, delta_t_days: float) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(cohort) deve ser chamado antes de predict().")
        names = list(self.dynamics_.keys())
        result = np.empty(len(names), dtype=float)
        for i, name in enumerate(names):
            fd = self.dynamics_[name]
            result[i] = fd.mu + (fd.phi_per_day**delta_t_days) * (x[i] - fd.mu)
        return result

    def sample(self, x: np.ndarray, delta_t_days: float, rng: np.random.Generator) -> np.ndarray:
        """
        Versão ESTOCÁSTICA de `predict()` — Fase 9 (Simulação): em vez de
        um único ponto determinístico, soma ruído Gaussiano com a
        variância TEORICAMENTE CORRETA de um processo de
        Ornstein-Uhlenbeck discreto saltando `delta_t_days` de uma vez
        (não iterando dia a dia):

            Var(X_{t+Δt} | X_t) = σ_ε² · (1 - φ^(2Δt)) / (1 - φ²)

        onde σ_ε = `FeatureDynamics.residual_std` (o ruído de 1 passo já
        ajustado por `fit()`) e φ = `phi_per_day`. Para Features
        instáveis (|φ|>=1, sem variância estacionária finita) ou sem
        ruído estimado, usa `residual_std` diretamente como aproximação,
        documentando a limitação em vez de propagar um NaN silencioso.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(cohort) deve ser chamado antes de sample().")
        mean = self.predict(x, delta_t_days)
        names = list(self.dynamics_.keys())
        noise = np.empty(len(names), dtype=float)
        for i, name in enumerate(names):
            fd = self.dynamics_[name]
            phi = fd.phi_per_day
            sigma_dia = fd.sigma_eps_per_day
            if sigma_dia <= 0:
                noise[i] = 0.0
                continue
            if abs(phi) >= 1.0 or abs(1 - phi**2) < 1e-9:
                var = sigma_dia**2  # sem variancia estacionaria finita -- usa o ruido de 1 dia como aproximacao
            else:
                var = (sigma_dia**2) * (1 - phi ** (2 * delta_t_days)) / (1 - phi**2)
            noise[i] = rng.normal(0, np.sqrt(max(var, 0.0)))
        return mean + noise

    def describe(self) -> str:
        n_stable = sum(1 for fd in self.dynamics_.values() if fd.is_stable) if self.is_fitted else 0
        return (
            f"MeanRevertingEvolutionOperator(n_features={len(self.dynamics_)}, "
            f"n_stable={n_stable}/{len(self.dynamics_)})"
        )
