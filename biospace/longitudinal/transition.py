"""
biospace.longitudinal.transition
===================================

TransitionAnalyzer generaliza `Cohort.transition_matrix()` (que continua
existindo, simples e sem dependências extras) com duas capacidades que
não fazem sentido morar no núcleo:

  1. Filtrar transições por intervalo de tempo entre exames consecutivos
     (`min_gap`/`max_gap`) — evita confundir um reexame no mesmo dia com
     uma verdadeira transição de acompanhamento de 6 meses.
  2. `time_to_transition()` / `summary()` — quanto tempo, em dias, cada
     tipo de transição de fenótipo observada de fato levou (não apenas
     SE ocorreu, mas EM QUANTO TEMPO).
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from biospace.core import LongitudinalOperator

if TYPE_CHECKING:
    from biospace.core import Cohort, Phenotype

__all__ = ["TransitionAnalyzer", "TransitionOperator"]


class TransitionAnalyzer(LongitudinalOperator[tuple]):
    def __init__(self, phenotypes: Sequence["Phenotype"], order: Optional[Sequence[str]] = None):
        self.phenotypes = list(phenotypes)
        self.order = order

    def _sequence_with_time(self, trajectory) -> list[tuple]:
        seq = []
        for i in range(len(trajectory)):
            vec = trajectory.at(i)
            x = vec.as_vector(self.order)
            name = next((ph.name for ph in self.phenotypes if ph.contains(x)), None)
            seq.append((vec.timestamp, name))
        return seq

    def _valid_transitions(self, cohort: "Cohort", min_gap: Optional[timedelta], max_gap: Optional[timedelta]):
        for traj in cohort.trajectories.values():
            seq = self._sequence_with_time(traj)
            for (t_a, a), (t_b, b) in zip(seq, seq[1:]):
                if a is None or b is None:
                    continue
                gap = t_b - t_a
                if min_gap is not None and gap < min_gap:
                    continue
                if max_gap is not None and gap > max_gap:
                    continue
                yield a, b, gap

    def matrix(
        self,
        cohort: "Cohort",
        min_gap: Optional[timedelta] = None,
        max_gap: Optional[timedelta] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """P(F_j | F_i), apenas sobre transições cujo intervalo de tempo esteja em [min_gap, max_gap]."""
        names = [ph.name for ph in self.phenotypes]
        idx = {n: i for i, n in enumerate(names)}
        counts = np.zeros((len(names), len(names)))
        for a, b, _ in self._valid_transitions(cohort, min_gap, max_gap):
            counts[idx[a], idx[b]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return counts / row_sums, names

    def time_to_transition(
        self,
        cohort: "Cohort",
        min_gap: Optional[timedelta] = None,
        max_gap: Optional[timedelta] = None,
    ) -> dict[tuple[str, str], list[timedelta]]:
        """Para cada par (fenótipo_origem, fenótipo_destino) observado, a lista de intervalos de tempo decorridos."""
        result: dict[tuple[str, str], list[timedelta]] = {}
        for a, b, gap in self._valid_transitions(cohort, min_gap, max_gap):
            result.setdefault((a, b), []).append(gap)
        return result

    def summary(
        self,
        cohort: "Cohort",
        min_gap: Optional[timedelta] = None,
        max_gap: Optional[timedelta] = None,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Estatísticas resumidas (n, média e mediana em dias) de cada tipo de transição observada."""
        raw = self.time_to_transition(cohort, min_gap, max_gap)
        summary: dict[tuple[str, str], dict[str, float]] = {}
        for key, gaps in raw.items():
            days = [g.total_seconds() / 86400 for g in gaps]
            summary[key] = {
                "n": len(days),
                "media_dias": float(np.mean(days)),
                "mediana_dias": float(np.median(days)),
            }
        return summary

    def fit(self, cohort: "Cohort") -> tuple[np.ndarray, list[str]]:
        """Satisfaz a interface LongitudinalOperator — alias direto de `matrix(cohort)` (sem filtro de tempo)."""
        return self.matrix(cohort)

    def describe(self) -> str:
        return f"TransitionAnalyzer: matriz de transição entre {len(self.phenotypes)} fenótipos."


# Alias: mesma classe, nome alinhado à hierarquia de Operator (ver README).
TransitionOperator = TransitionAnalyzer
