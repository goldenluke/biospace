"""
biospace.early_warning.critical_slowing_down
================================================

CriticalSlowingDownDetector: implementação concreta de EarlyWarningOperator,
aplicando o pipeline clássico de detecção de "critical slowing down"
(Scheffer et al. 2009, Nature; Dakos et al. 2012, PLOS ONE) sobre
trajetórias de uma Cohort.

Pipeline, por trajetória (com pelo menos `min_points` observações):

    Trajectory
        -> indicador univariado por instante (indicator_fn)
        -> detrend (linear OU kernel gaussiano — configurável)
        -> janela deslizante (por Nº DE PONTOS ou por TEMPO DECORRIDO — configurável)
        -> variância + autocorrelação lag-1 + ASSIMETRIA (skewness) por janela
        -> tendência de Kendall (τ) de cada indicador
        -> significância por DADOS SUBSTITUTOS AR(1) (Dakos et al. 2012)
        -> PESO DE EVIDÊNCIA: >= metade dos indicadores disponíveis
           precisam concordar (crescente + significativo), não todos —
           critério de "contagem de indicadores" que a própria literatura
           recomenda (Dakos et al. 2012; Kéfi et al. 2013) em vez de
           exigir unanimidade entre só 2 sinais.

MELHORIAS SOBRE A VERSÃO ANTERIOR (ver README para o histórico):

  1. Terceiro indicador — ASSIMETRIA (skewness): perto de uma transição
     crítica, o sistema tende a passar mais tempo perto de um dos dois
     atratores antes de alternar, distorcendo a distribuição das
     flutuações (Guttal & Jayaprakash, 2008). Adiciona um sinal
     independente de variância/autocorrelação.
  2. Peso de evidência em vez de unanimidade: antes exigia que AMBOS
     variância E autocorrelação concordassem. Agora, com até 3
     indicadores disponíveis, exige maioria (>= metade) — mais robusto a
     um único indicador ruidoso, e alinhado com a prática da literatura
     de "contar quantos indicadores sobem".
  3. Detrend por kernel gaussiano (`detrend_method="gaussian"`) como
     alternativa ao linear — captura tendências de longo prazo não
     lineares, reduzindo resíduo espúrio quando a trajetória real não é
     bem aproximada por uma reta.
  4. Janela por TEMPO DECORRIDO (`window_mode="days"`) além de por
     contagem de pontos — ataca diretamente a limitação de amostragem
     irregular já documentada: em vez de assumir que N observações
     consecutivas cobrem um período comparável de tempo entre
     pacientes, agrupa por uma janela de dias fixa, com nº de pontos
     variável por janela.

TESTE DE SIGNIFICÂNCIA POR DADOS SUBSTITUTOS (Dakos et al. 2012, Seção
"Significance of trends"): o p-value PARAMÉTRICO do teste de Kendall
assume observações independentes — falso aqui, já que janelas
consecutivas se sobrepõem e há poucas janelas por trajetória. Em vez
disso, ajustamos um processo AR(1) aos MESMOS resíduos detrendizados,
geramos `n_surrogates` séries desse processo nulo, rodamos o MESMO
pipeline em cada uma, e comparamos o τ observado à distribuição nula
resultante (p_surrogate = fração de substitutos com τ >= observado).

NOTA SOBRE AMOSTRAGEM IRREGULAR: mesmo com `window_mode="days"`, a
autocorrelação lag-1 dentro de uma janela ainda assume implicitamente
espaçamentos comparáveis ENTRE os pontos que caem na janela — isso é
uma limitação de fundo do método sobre dados esparsos, não totalmente
eliminada, apenas mitigada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau, skew

from biospace.geometry import Euclidean

from .base import EarlyWarningOperator

if TYPE_CHECKING:
    from biospace.core import Cohort, Geometry, RepresentationVector, Trajectory

__all__ = ["EWSSeries", "IndicatorResult", "EWSResult", "CriticalSlowingDownDetector"]

_INDICATOR_NAMES = ("variance", "autocorrelation", "skewness")


@dataclass
class EWSSeries:
    """Série de indicadores calculados por janela, para UMA trajetória (real ou substituta)."""

    times_days: list[float] = field(default_factory=list)  # tempo (dias desde a 1ª obs) no FIM de cada janela
    variance: list[float] = field(default_factory=list)
    autocorrelation: list[float] = field(default_factory=list)
    skewness: list[float] = field(default_factory=list)
    window_n_points: list[int] = field(default_factory=list)  # nº de pontos em cada janela (varia se window_mode="days")


@dataclass
class IndicatorResult:
    """Resultado de UM indicador (variância, autocorrelação ou assimetria) para uma trajetória."""

    tau: Optional[float] = None
    p_parametric: Optional[float] = None
    p_surrogate: Optional[float] = None
    n_surrogates: int = 0

    @property
    def is_rising(self) -> bool:
        return self.tau is not None and self.tau > 0

    @property
    def p_value(self) -> Optional[float]:
        """p-value preferido: substituto se disponível, senão paramétrico (fallback, menos robusto)."""
        return self.p_surrogate if self.n_surrogates > 0 else self.p_parametric

    def is_significant(self, alpha: float = 0.10) -> bool:
        p = self.p_value
        return p is not None and p < alpha

    @property
    def is_rising_and_significant(self) -> bool:
        return self.is_rising and self.is_significant()


@dataclass
class EWSResult:
    """Resultado da detecção de critical slowing down para UM sistema."""

    system_id: str
    n_points: int
    sufficient_data: bool
    series: EWSSeries = field(default_factory=EWSSeries)
    indicators: dict[str, IndicatorResult] = field(default_factory=dict)

    # Atalhos de compatibilidade com a versão anterior (mesmos nomes, agora derivados de `indicators`)
    @property
    def tau_variance(self) -> Optional[float]:
        return self.indicators.get("variance", IndicatorResult()).tau

    @property
    def tau_autocorrelation(self) -> Optional[float]:
        return self.indicators.get("autocorrelation", IndicatorResult()).tau

    @property
    def tau_skewness(self) -> Optional[float]:
        return self.indicators.get("skewness", IndicatorResult()).tau

    @property
    def p_variance_surrogate(self) -> Optional[float]:
        return self.indicators.get("variance", IndicatorResult()).p_surrogate

    @property
    def p_autocorrelation_surrogate(self) -> Optional[float]:
        return self.indicators.get("autocorrelation", IndicatorResult()).p_surrogate

    @property
    def n_surrogates(self) -> int:
        for ind in self.indicators.values():
            if ind.n_surrogates > 0:
                return ind.n_surrogates
        return 0

    @property
    def n_indicators_available(self) -> int:
        return sum(1 for ind in self.indicators.values() if ind.tau is not None)

    @property
    def n_indicators_rising_significant(self) -> int:
        """Quantos indicadores (de até 3) mostram tendência CRESCENTE e SIGNIFICATIVA (peso de evidência)."""
        return sum(1 for ind in self.indicators.values() if ind.is_rising_and_significant)

    @property
    def warning(self) -> bool:
        """
        PESO DE EVIDÊNCIA (Dakos et al. 2012; Kéfi et al. 2013): pelo
        menos METADE dos indicadores disponíveis (arredondado para cima)
        precisam mostrar tendência crescente e significativa — não
        unanimidade entre só 2 sinais como na versão anterior. Com os 3
        indicadores completos, isso equivale a exigir >= 2 de 3 (maioria).
        Requer pelo menos 2 indicadores disponíveis para emitir qualquer
        alerta (com só 1 disponível, não há "peso de evidência" possível).
        """
        n_available = self.n_indicators_available
        if n_available < 2:
            return False
        threshold = -(-n_available // 2)  # ceil(n_available / 2)
        return self.n_indicators_rising_significant >= threshold

    def summary(self) -> str:
        if not self.sufficient_data:
            return f"{self.system_id}: dados insuficientes (n={self.n_points})"
        lines = [f"{self.system_id} (n={self.n_points} exames, {self.n_indicators_available} indicadores disponíveis):"]
        for name, ind in self.indicators.items():
            if ind.tau is None:
                continue
            flag = "↑ SIGNIFICATIVO" if ind.is_rising_and_significant else ("↑" if ind.is_rising else "↓")
            p_str = f"p={ind.p_value:.3f}" if ind.p_value is not None else "p=n/a"
            lines.append(f"  {name}: τ={ind.tau:+.3f} ({p_str}) {flag}")
        lines.append(f"  => warning={self.warning} ({self.n_indicators_rising_significant}/{self.n_indicators_available} indicadores concordam)")
        return "\n".join(lines)


def _linear_detrend(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Remove a tendência linear de longo prazo; retorna os resíduos (flutuações de curto prazo)."""
    if len(times) < 2 or np.allclose(times, times[0]):
        return values - values.mean()
    coeffs = np.polyfit(times, values, deg=1)
    trend = np.polyval(coeffs, times)
    return values - trend


def _gaussian_detrend(times: np.ndarray, values: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Remove uma tendência de longo prazo suavizada por kernel gaussiano
    (Dakos et al. 2012 usam isso como alternativa ao detrend linear,
    para tendências não lineares). `bandwidth` em dias; se None, usa 1/4
    do intervalo total da trajetória.
    """
    if len(times) < 2 or np.allclose(times, times[0]):
        return values - values.mean()
    span = times.max() - times.min()
    bw = bandwidth if bandwidth is not None else max(span / 4.0, 1e-6)
    trend = np.empty_like(values)
    for i, t in enumerate(times):
        weights = np.exp(-0.5 * ((times - t) / bw) ** 2)
        weights_sum = weights.sum()
        trend[i] = float(np.sum(weights * values) / weights_sum) if weights_sum > 1e-12 else values[i]
    return values - trend


def _fit_ar1(residuals: np.ndarray) -> tuple[float, float]:
    """Ajusta r_t = φ·r_{t-1} + ε_t por mínimos quadrados. Retorna (φ, σ_ε)."""
    x, y = residuals[:-1], residuals[1:]
    if len(x) < 2 or np.std(x) < 1e-12:
        return 0.0, float(np.std(residuals)) if len(residuals) else 0.0
    phi = float(np.clip(np.sum(x * y) / np.sum(x * x), -0.999, 0.999))
    resid_ar = y - phi * x
    sigma_eps = float(np.std(resid_ar, ddof=1)) if len(resid_ar) > 1 else float(np.std(residuals))
    return phi, sigma_eps


def _generate_ar1_surrogate(n: int, phi: float, sigma_eps: float, rng: np.random.Generator) -> np.ndarray:
    """Gera uma série AR(1) estacionária de tamanho `n`, com o mesmo φ/σ_ε estimados dos dados reais."""
    series = np.zeros(n)
    stationary_std = sigma_eps / np.sqrt(1 - phi**2) if abs(phi) < 0.999 else sigma_eps
    series[0] = rng.normal(0, stationary_std)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal(0, sigma_eps)
    return series


class CriticalSlowingDownDetector(EarlyWarningOperator):
    """Ver docstring do módulo para o pipeline completo, as melhorias e as limitações."""

    def __init__(
        self,
        indicator_fn: Callable[["RepresentationVector", "RepresentationVector"], float],
        window_size: int = 4,
        min_points: int = 8,
        n_surrogates: int = 200,
        surrogate_seed: int = 42,
        detrend_method: Literal["linear", "gaussian"] = "linear",
        gaussian_bandwidth: Optional[float] = None,
        window_mode: Literal["points", "days"] = "points",
        window_days: Optional[float] = None,
        min_points_per_window: int = 3,
    ):
        """
        indicator_fn(vetor_atual, vetor_baseline) -> float: a série
        univariada sobre a qual o critical slowing down é medido.
        `window_size`: nº de observações consecutivas por janela (usado
        se `window_mode="points"`).
        `min_points`: trajetórias mais curtas são marcadas `sufficient_data=False`.
        `n_surrogates`: nº de séries substitutas AR(1) para o teste de
        significância (0 desativa, cai para o p-value paramétrico).
        `detrend_method`: "linear" (padrão, retrocompatível) ou "gaussian"
        (kernel gaussiano — melhor para tendências não lineares).
        `window_mode`: "points" (padrão, retrocompatível — janela de
        `window_size` observações consecutivas) ou "days" (janela de
        `window_days` dias decorridos, nº de pontos variável por janela
        — mitiga o problema de amostragem irregular).
        `min_points_per_window`: em `window_mode="days"`, janelas com
        menos pontos que isso são descartadas (poucos pontos tornam
        variância/autocorrelação/assimetria não confiáveis).
        """
        self.indicator_fn = indicator_fn
        self.window_size = window_size
        self.min_points = min_points
        self.n_surrogates = n_surrogates
        self.surrogate_seed = surrogate_seed
        self.detrend_method = detrend_method
        self.gaussian_bandwidth = gaussian_bandwidth
        self.window_mode = window_mode
        self.window_days = window_days
        self.min_points_per_window = min_points_per_window

        if window_mode == "days" and window_days is None:
            raise ValueError("window_mode='days' exige `window_days` explícito (ex.: 180).")

    @classmethod
    def for_feature(cls, domain_name: str, feature_name: str, **kwargs) -> "CriticalSlowingDownDetector":
        """Conveniência: indicador = valor de UMA Feature nomeada ao longo do tempo (ex.: 'ido' do domínio 'apnea')."""

        def indicator_fn(vector: "RepresentationVector", baseline: "RepresentationVector") -> float:
            for f in vector.components.get(domain_name, []):
                if f.name == feature_name:
                    return f.value
            raise KeyError(f"Feature '{feature_name}' não encontrada no domínio '{domain_name}'.")

        return cls(indicator_fn, **kwargs)

    @classmethod
    def for_distance_from_baseline(
        cls, geometry: Optional["Geometry"] = None, order: Optional[Sequence[str]] = None, **kwargs
    ) -> "CriticalSlowingDownDetector":
        """Conveniência: indicador = distância geométrica de cada instante ao PRIMEIRO ponto da própria trajetória."""
        geom = geometry or Euclidean()

        def indicator_fn(vector: "RepresentationVector", baseline: "RepresentationVector") -> float:
            return geom.distance(vector.as_vector(order), baseline.as_vector(order))

        return cls(indicator_fn, **kwargs)

    # -------------------------------------------------------------------
    # Núcleo compartilhado (usado tanto para os dados reais quanto para
    # cada série substituta — garante que exatamente o mesmo pipeline
    # seja aplicado a ambos).
    # -------------------------------------------------------------------
    def _detrend(self, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        if self.detrend_method == "gaussian":
            return _gaussian_detrend(times, values, self.gaussian_bandwidth)
        return _linear_detrend(times, values)

    def _window_indices(self, times: np.ndarray) -> list[np.ndarray]:
        """Retorna, para cada janela, os índices dos pontos que pertencem a ela."""
        n = len(times)
        if self.window_mode == "points":
            w = self.window_size
            return [np.arange(end - w, end) for end in range(w, n + 1)]

        # window_mode == "days": janela = todos os pontos com t em (t_fim - window_days, t_fim]
        windows = []
        for end_idx in range(n):
            t_end = times[end_idx]
            mask = (times > t_end - self.window_days) & (times <= t_end)
            idx = np.where(mask)[0]
            if len(idx) >= self.min_points_per_window:
                windows.append(idx)
        return windows

    def _window_series(self, times: np.ndarray, residuals: np.ndarray) -> EWSSeries:
        series = EWSSeries()
        for idx in self._window_indices(times):
            window_res = residuals[idx]
            n_pts = len(window_res)
            variance = float(np.var(window_res, ddof=1)) if n_pts >= 2 else float("nan")
            if n_pts >= 3 and np.std(window_res) > 1e-12:
                ac = float(np.corrcoef(window_res[:-1], window_res[1:])[0, 1])
                sk = float(skew(window_res))
            else:
                ac = float("nan")
                sk = float("nan")
            series.times_days.append(float(times[idx[-1]]))
            series.variance.append(variance)
            series.autocorrelation.append(ac)
            series.skewness.append(sk)
            series.window_n_points.append(n_pts)
        return series

    @staticmethod
    def _kendall_for(times_days: list[float], values: list[float]) -> tuple[Optional[float], Optional[float]]:
        valid = [(t, v) for t, v in zip(times_days, values) if not np.isnan(v)]
        if len(valid) < 4:
            return None, None
        ts, vs = zip(*valid)
        tau, p = kendalltau(ts, vs)
        return float(tau), float(p)

    def _kendall_from_series(self, series: EWSSeries) -> dict[str, tuple[Optional[float], Optional[float]]]:
        return {
            "variance": self._kendall_for(series.times_days, series.variance),
            "autocorrelation": self._kendall_for(series.times_days, series.autocorrelation),
            "skewness": self._kendall_for(series.times_days, series.skewness),
        }

    def _analyze_trajectory(self, trajectory: "Trajectory") -> EWSResult:
        n = len(trajectory)
        if n < self.min_points:
            return EWSResult(system_id=trajectory.system_id, n_points=n, sufficient_data=False)

        baseline = trajectory.at(0)
        t0 = baseline.timestamp
        times = np.array([(trajectory.at(i).timestamp - t0).total_seconds() / 86400 for i in range(n)])
        values = np.array([self.indicator_fn(trajectory.at(i), baseline) for i in range(n)])
        residuals = self._detrend(times, values)

        series = self._window_series(times, residuals)
        kendall = self._kendall_from_series(series)

        indicators = {name: IndicatorResult(tau=tau, p_parametric=p) for name, (tau, p) in kendall.items()}
        result = EWSResult(system_id=trajectory.system_id, n_points=n, sufficient_data=True, series=series, indicators=indicators)

        if self.n_surrogates > 0:
            self._run_surrogate_test(result, times, residuals)

        return result

    def _run_surrogate_test(self, result: EWSResult, times: np.ndarray, residuals: np.ndarray) -> None:
        """Preenche p_surrogate de cada indicador em `result.indicators`, in-place."""
        phi, sigma_eps = _fit_ar1(residuals)
        rng = np.random.default_rng(self.surrogate_seed)

        surrogate_taus: dict[str, list[float]] = {name: [] for name in _INDICATOR_NAMES}
        for _ in range(self.n_surrogates):
            surrogate_residuals = _generate_ar1_surrogate(len(residuals), phi, sigma_eps, rng)
            surrogate_series = self._window_series(times, surrogate_residuals)
            kendall = self._kendall_from_series(surrogate_series)
            for name, (tau, _) in kendall.items():
                if tau is not None:
                    surrogate_taus[name].append(tau)

        for name, ind in result.indicators.items():
            taus = surrogate_taus.get(name, [])
            if ind.tau is not None and taus:
                ind.p_surrogate = float(np.mean(np.array(taus) >= ind.tau))
                ind.n_surrogates = self.n_surrogates

    def fit(self, cohort: "Cohort") -> dict[str, EWSResult]:
        """Roda o pipeline completo (+ substitutos, se `n_surrogates > 0`) sobre toda a coorte."""
        return {sid: self._analyze_trajectory(traj) for sid, traj in cohort.trajectories.items()}

    def describe(self) -> str:
        return (
            f"CriticalSlowingDownDetector(window_mode={self.window_mode}, detrend={self.detrend_method}, "
            f"min_points={self.min_points}, n_surrogates={self.n_surrogates}): critical slowing down "
            f"(variância + autocorrelação + assimetria) com peso de evidência e substitutos AR(1)."
        )
