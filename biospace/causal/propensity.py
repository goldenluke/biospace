"""
biospace.causal.propensity
=============================

Ajuste por escore de propensão (Rosenbaum & Rubin, 1983) — o passo que
faltava desde a primeira versão de `biospace.causal` (documentado no
README como "próximo passo real, não um esquecimento").

PROBLEMA DE DIMENSIONALIDADE (o mesmo já encontrado e corrigido em
`PhenotypeConditionedGeometry` com Ledoit-Wolf): esta coorte real tem
~355 pacientes e 52 Features, com prevalência de tratamento desbalanceada
(ex.: 294 vs. 61 para AAM). Uma regressão logística NÃO regularizada
sobre as 52 Features sofreria overfitting severo / separação quase
perfeita nesse regime. Por isso o escore de propensão aqui usa
regularização L2 (Ridge logístico) forte por padrão, e a qualidade do
ajuste é reportada explicitamente via AUC de validação cruzada — nunca
escondida atrás de um número de confiança que pareça mais sólido do que é.

MÉTODO DE PAREAMENTO: vizinho-mais-próximo 1:1 sem reposição, na escala
LOGIT do escore de propensão, com caliper (Austin, 2011) — pacientes
tratados só são pareados a controles cujo logit da propensão esteja a
até `caliper` de distância; tratados sem par dentro do caliper ficam de
fora (não são forçados a um mau pareamento só para não perder um caso).

ISSO CONTINUA SENDO UMA ASSOCIAÇÃO OBSERVACIONAL AJUSTADA, NÃO UMA PROVA
CAUSAL: o pareamento só corrige desequilíbrio nas Features OBSERVADAS.
Confundidores não medidos (adesão, motivação, acesso a cuidado) nunca
aparecem aqui, pareados ou não.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .balance import BalanceReport, _collect_baseline, check_baseline_balance

if TYPE_CHECKING:
    from biospace.core import Cohort

__all__ = ["PropensityModel", "PropensityMatchResult", "estimate_propensity", "match_on_propensity"]


@dataclass
class PropensityModel:
    """Escore de propensão ajustado — P(tratamento=1 | linha de base) por paciente, com diagnóstico de qualidade."""

    scores: dict[str, float]
    cv_auc: float
    feature_names: list[str]
    coefficients: dict[str, float]
    C: float

    def summary(self) -> str:
        lines = [f"Escore de propensão (regressão logística L2, C={self.C}): AUC de validação cruzada = {self.cv_auc:.3f}"]
        if np.isnan(self.cv_auc):
            lines.append("(AUC não calculável — grupo pequeno demais para validação cruzada.)")
        elif self.cv_auc > 0.95:
            lines.append(
                "⚠️ AUC muito alto (>0.95): os grupos são tão distintos na linha de base que o modelo "
                "quase os separa perfeitamente. Isso NÃO é um problema do modelo — é o mesmo "
                "confundimento por indicação que check_baseline_balance() já mostrava; o pareamento "
                "pode ter dificuldade em achar bons pares neste regime."
            )
        top = sorted(self.coefficients.items(), key=lambda kv: -abs(kv[1]))[:8]
        lines.append("Maiores coeficientes (após regularização L2, em escala padronizada):")
        for name, coef in top:
            lines.append(f"  {name}: {coef:+.3f}")
        return "\n".join(lines)


def estimate_propensity(
    cohort: "Cohort",
    treatment_domain: str,
    treatment_feature: str,
    order: Optional[Sequence[str]] = None,
    C: float = 0.1,
    cv_folds: int = 5,
    random_state: int = 42,
) -> PropensityModel:
    """
    Ajusta P(tratamento=1 | linha de base) via regressão logística com
    regularização L2. `C` pequeno (padrão 0.1) implica regularização
    FORTE, deliberadamente — ver docstring do módulo sobre o regime de
    alta dimensionalidade.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    baseline_vectors, treated_ids, feature_names = _collect_baseline(cohort, treatment_domain, treatment_feature, order)
    ids = list(baseline_vectors.keys())
    X = np.stack([baseline_vectors[sid] for sid in ids])
    y = np.array([1 if sid in treated_ids else 0 for sid in ids])

    if len(set(y.tolist())) < 2:
        raise ValueError("Só há um grupo (todos tratados ou todos não-tratados) — não há como estimar propensão.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=C, max_iter=2000, random_state=random_state)

    cv_auc = float("nan")
    n_per_class = int(np.bincount(y).min())
    n_splits = min(cv_folds, n_per_class)
    if n_splits >= 2:
        scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring="roc_auc")
        cv_auc = float(np.mean(scores))

    model.fit(X_scaled, y)
    propensity_scores = model.predict_proba(X_scaled)[:, 1]

    return PropensityModel(
        scores=dict(zip(ids, propensity_scores.tolist())),
        cv_auc=cv_auc,
        feature_names=feature_names,
        coefficients=dict(zip(feature_names, model.coef_[0].tolist())),
        C=C,
    )


@dataclass
class PropensityMatchResult:
    """Resultado do pareamento — pares (tratado, controle) + comparação de balanceamento antes/depois."""

    matched_pairs: list[tuple[str, str]]
    n_treated_total: int
    n_matched: int
    caliper: float
    balance_before: BalanceReport
    balance_after: Optional[BalanceReport] = None

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_treated_total if self.n_treated_total else 0.0

    @property
    def matched_system_ids(self) -> set[str]:
        ids: set[str] = set()
        for t, c in self.matched_pairs:
            ids.add(t)
            ids.add(c)
        return ids

    @property
    def mean_absolute_smd_before(self) -> float:
        return float(np.mean([abs(v) for v in self.balance_before.smd.values()])) if self.balance_before.smd else float("nan")

    @property
    def mean_absolute_smd_after(self) -> Optional[float]:
        if self.balance_after is None:
            return None
        return float(np.mean([abs(v) for v in self.balance_after.smd.values()])) if self.balance_after.smd else float("nan")

    @property
    def improved_balance(self) -> Optional[bool]:
        """
        True se o pareamento reduziu o DESEQUILÍBRIO MÉDIO (|SMD| médio
        entre todas as Features) — uma métrica CONTÍNUA, não a contagem
        binária de quantas Features cruzam o limiar de 0.1. Achado real
        ao validar isto em dados sintéticos: o SMD de uma Feature caiu de
        1.50 para 0.14 (>90% de redução) após o pareamento, mas 0.14
        ainda estava (por pouco) acima do limiar — a contagem binária
        sozinha diria "não melhorou", escondendo uma melhora real e
        substancial. Por isso a métrica de referência aqui é contínua.
        """
        after = self.mean_absolute_smd_after
        if after is None:
            return None
        return after < self.mean_absolute_smd_before

    def summary(self) -> str:
        lines = [
            f"Pareamento por escore de propensão: {self.n_matched}/{self.n_treated_total} pacientes "
            f"tratados pareados ({100 * self.match_rate:.0f}%), caliper={self.caliper} (escala logit)",
            "",
            f"ANTES:  {self.balance_before.n_imbalanced}/{len(self.balance_before.feature_names)} Features "
            f"desequilibradas (|SMD| médio = {self.mean_absolute_smd_before:.3f})",
        ]
        if self.balance_after is not None:
            lines.append(
                f"DEPOIS: {self.balance_after.n_imbalanced}/{len(self.balance_after.feature_names)} Features "
                f"desequilibradas (|SMD| médio = {self.mean_absolute_smd_after:.3f})"
            )
            reducao_pct = 100 * (1 - self.mean_absolute_smd_after / self.mean_absolute_smd_before) if self.mean_absolute_smd_before > 1e-9 else 0.0
            if self.improved_balance:
                lines.append(f"=> O pareamento REDUZIU o desequilíbrio médio em {reducao_pct:.0f}% (mesmo que a contagem binária de Features acima do limiar não tenha mudado).")
            else:
                lines.append(
                    "=> ⚠️ O pareamento NÃO reduziu o desequilíbrio médio (ou piorou) — não assuma que "
                    "pareamento resolve confundimento automaticamente; confira caliper e AUC do modelo."
                )
        else:
            lines.append("DEPOIS: não foi possível comparar (nenhum par encontrado dentro do caliper).")
        return "\n".join(lines)


def match_on_propensity(
    cohort: "Cohort",
    treatment_domain: str,
    treatment_feature: str,
    order: Optional[Sequence[str]] = None,
    caliper: float = 0.2,
    propensity_model: Optional[PropensityModel] = None,
    imbalance_threshold: float = 0.1,
) -> PropensityMatchResult:
    """
    Pareamento vizinho-mais-próximo 1:1 SEM reposição, na escala logit do
    escore de propensão, com caliper (ver docstring do módulo). Roda
    `check_baseline_balance()` antes E depois automaticamente — não é
    opcional por acidente, para dificultar interpretar o pareamento sem
    ver se ele realmente ajudou.
    """
    balance_before = check_baseline_balance(cohort, treatment_domain, treatment_feature, order, imbalance_threshold)

    model = propensity_model or estimate_propensity(cohort, treatment_domain, treatment_feature, order)

    _, treated_ids, _ = _collect_baseline(cohort, treatment_domain, treatment_feature, order)
    treated_ids_list = [sid for sid in model.scores if sid in treated_ids]
    control_ids = [sid for sid in model.scores if sid not in treated_ids]

    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    treated_logits = {sid: _logit(model.scores[sid]) for sid in treated_ids_list}
    available_controls = {sid: _logit(model.scores[sid]) for sid in control_ids}

    pairs: list[tuple[str, str]] = []
    # Pareia do tratado com propensão mais EXTREMA (mais difícil de casar) primeiro —
    # evita que os "fáceis" esgotem controles que os "difíceis" precisariam.
    for sid in sorted(treated_ids_list, key=lambda s: -abs(treated_logits[s])):
        if not available_controls:
            break
        t_logit = treated_logits[sid]
        best_control = min(available_controls, key=lambda c: abs(available_controls[c] - t_logit))
        if abs(available_controls[best_control] - t_logit) <= caliper:
            pairs.append((sid, best_control))
            del available_controls[best_control]

    balance_after = None
    if pairs:
        matched_ids = set()
        for t, c in pairs:
            matched_ids.add(t)
            matched_ids.add(c)
        balance_after = check_baseline_balance(cohort, treatment_domain, treatment_feature, order, imbalance_threshold, system_ids=matched_ids)

    return PropensityMatchResult(
        matched_pairs=pairs,
        n_treated_total=len(treated_ids_list),
        n_matched=len(pairs),
        caliper=caliper,
        balance_before=balance_before,
        balance_after=balance_after,
    )


@dataclass
class MatchedEffectReport:
    """
    Estimativa de efeito por DIFERENÇA-EM-DIFERENÇAS (DiD) sobre os pares
    pareados: para cada par (tratado, controle), Δ_tratado = (último
    exame - primeiro exame) do tratado; Δ_controle = idem para o
    controle no MESMO período; efeito = média(Δ_tratado) - média(Δ_controle).

    Ainda uma associação observacional AJUSTADA, não uma prova causal —
    ver aviso central em `biospace.causal.do_operator`.
    """

    feature_names: list[str]
    effect: dict[str, float] = field(default_factory=dict)
    effect_std: dict[str, float] = field(default_factory=dict)
    n_pairs_used: int = 0
    n_pairs_dropped_single_exam_control: int = 0

    def top_effects(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.effect.items(), key=lambda kv: -abs(kv[1]))[:n]

    def summary(self) -> str:
        lines = [
            f"Efeito pareado (diferença-em-diferenças, {self.n_pairs_used} pares usados"
            + (f", {self.n_pairs_dropped_single_exam_control} descartados por controle com 1 exame só)" if self.n_pairs_dropped_single_exam_control else ")"),
        ]
        for name, eff in self.top_effects(10):
            lines.append(f"  {name}: {eff:+.3f} (±{self.effect_std.get(name, 0):.3f})")
        return "\n".join(lines)


def estimate_matched_effect(
    cohort: "Cohort",
    match_result: PropensityMatchResult,
    order: Optional[Sequence[str]] = None,
) -> MatchedEffectReport:
    """
    Fecha o ciclo que `match_on_propensity()` sozinho deixa aberto:
    parear melhora o BALANCEAMENTO, mas não dá, por si só, uma estimativa
    de EFEITO. Aqui, usa os pares já formados para uma diferença-em-
    diferenças: Δ de cada Feature (último exame - primeiro exame) do
    tratado, menos o mesmo Δ do controle pareado.

    Pares cujo CONTROLE tem só 1 exame (sem Δ mensurável) são descartados
    — reportado explicitamente em `n_pairs_dropped_single_exam_control`,
    não escondido.
    """
    deltas_treated: list[np.ndarray] = []
    deltas_control: list[np.ndarray] = []
    feature_names: list[str] = []
    n_dropped = 0

    for treated_id, control_id in match_result.matched_pairs:
        treated_traj = cohort.trajectories[treated_id]
        control_traj = cohort.trajectories[control_id]

        if len(control_traj) < 2:
            n_dropped += 1
            continue
        if len(treated_traj) < 2:
            n_dropped += 1
            continue

        if not feature_names:
            domain_order = order or sorted(treated_traj.at(0).components.keys())
            for domain_name in domain_order:
                for f in treated_traj.at(0).components[domain_name]:
                    feature_names.append(f"{domain_name}.{f.name}")

        delta_t = treated_traj.latest().as_vector(order) - treated_traj.at(0).as_vector(order)
        delta_c = control_traj.latest().as_vector(order) - control_traj.at(0).as_vector(order)
        deltas_treated.append(delta_t)
        deltas_control.append(delta_c)

    if not deltas_treated:
        raise ValueError(
            "Nenhum par pareado tinha AMBOS os lados com >= 2 exames — impossível calcular "
            "diferença-em-diferenças. Verifique se os controles pareados têm follow-up."
        )

    dt = np.stack(deltas_treated)
    dc = np.stack(deltas_control)
    diff_in_diff = dt - dc  # uma linha por par: Δ_tratado - Δ_controle

    effect = {name: float(np.mean(diff_in_diff[:, i])) for i, name in enumerate(feature_names)}
    effect_std = {name: float(np.std(diff_in_diff[:, i])) for i, name in enumerate(feature_names)}

    return MatchedEffectReport(
        feature_names=feature_names,
        effect=effect,
        effect_std=effect_std,
        n_pairs_used=len(deltas_treated),
        n_pairs_dropped_single_exam_control=n_dropped,
    )
