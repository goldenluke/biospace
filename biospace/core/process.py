"""
biospace.core.process
========================

PhysiologicalProcess (P): um mecanismo biológico nomeado que um ou mais
Observables medem, direta ou indiretamente — ex.: "Homeostase
Glicêmica", "Filtração Renal", "Regulação Cardiovascular".

Distinto de SemanticDomain: um domínio agrupa Features para fins de
REPRESENTAÇÃO (o que entra no vetor, com que normalização); um processo
nomeia o MECANISMO subjacente que um Observable mede, independente de
qual(is) domínio(s) acabam consumindo essa medida. O mesmo processo pode
alimentar vários domínios (ex.: Metabolismo Lipídico pode alimentar
tanto um domínio Metabólico quanto um domínio Cardiovascular); o mesmo
domínio pode se apoiar em vários processos.

CAMADA OPCIONAL, NÃO OBRIGATÓRIA: um Observable que nunca declara
`process` continua funcionando exatamente como antes —
`Observable.process` tem default `None`, e todo o código existente
(núcleo, plugin sleep, testes) nunca precisa saber que esta classe
existe. Isto é aditivo por construção, não uma migração.

Ver `SemanticDomain.processes()` (domain.py) e
`Representation.processes()` / `Representation.features_by_process()`
(representation.py) para as consultas que esta camada habilita.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .representation import Representation
    from .representation_space import RepresentationSpace

__all__ = ["PhysiologicalProcess", "ProcessCoherenceReport", "check_process_coherence"]


@dataclass(frozen=True)
class PhysiologicalProcess:
    """
    name: identificador curto (ex.: "glucose_homeostasis") — é este
    valor que `Observable.process` referencia como string, não uma
    instância desta classe (mantém `Observable` livre para declarar seu
    processo como um literal de classe simples, sem importar este
    módulo — o mesmo padrão já usado por `Observable.key`).

    description: explicação em linguagem natural do mecanismo — para
    documentação e para relatórios, nunca usada em lógica de código.
    """

    name: str
    description: str = ""

    def __repr__(self) -> str:
        return f"PhysiologicalProcess({self.name!r})"


@dataclass
class ProcessCoherenceReport:
    """
    Resultado de `check_process_coherence` — validação EMPÍRICA (não
    apenas a alegação) de que Features rotuladas sob o mesmo
    PhysiologicalProcess correlacionam mais entre si, através da
    população, do que Features de processos diferentes. Um rótulo de
    processo é uma hipótese sobre mecanismo compartilhado; este
    contrato testa se os dados sustentam essa hipótese, na mesma
    disciplina de `phenotyping.contracts.check_phenotype_stability`.
    """

    same_process_pairs: list[tuple[str, str, float]] = field(default_factory=list)  # (feature_a, feature_b, |r|)
    different_process_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    mean_same_process: float = float("nan")
    mean_different_process: float = float("nan")
    mannwhitney_p: Optional[float] = None

    @property
    def is_coherent(self) -> bool:
        """True se pares do MESMO processo têm |correlação| média maior que pares de processos DIFERENTES, com p<0.05."""
        if not self.same_process_pairs or not self.different_process_pairs:
            return False
        if self.mannwhitney_p is None:
            return False
        return self.mean_same_process > self.mean_different_process and self.mannwhitney_p < 0.05

    def summary(self) -> str:
        status = "OK" if self.is_coherent else "NÃO CONFIRMADO"
        return (
            f"Coerência de processo: {status}\n"
            f"  |r| médio, mesmo processo:     {self.mean_same_process:.3f}  (n={len(self.same_process_pairs)} pares)\n"
            f"  |r| médio, processos diferentes: {self.mean_different_process:.3f}  (n={len(self.different_process_pairs)} pares)\n"
            f"  Mann-Whitney p={self.mannwhitney_p:.2e}" if self.mannwhitney_p is not None else "  (pares insuficientes para o teste)"
        )


def check_process_coherence(
    representation: "Representation", space: "RepresentationSpace", order: Optional[Sequence[str]] = None
) -> ProcessCoherenceReport:
    """
    Para cada par de Features com PROCESSO DECLARADO (via
    `Observable.process`), calcula a correlação de Pearson (valores
    BRUTOS, pairwise-complete — ignora pacientes com algum dos dois
    ausentes) através de toda a população em `space`. Compara |r| médio
    entre pares do MESMO processo contra pares de processos
    DIFERENTES, via teste de Mann-Whitney.

    Features sem processo declarado (a maioria, em plugins que nunca
    usaram esta camada) são simplesmente ignoradas — não entram em
    nenhum par.
    """
    from scipy import stats as scipy_stats

    domain_order = list(order) if order is not None else representation.domain_names()

    # feature_id = (domain_name, feature_name) -> (processo, array de valores brutos alinhados à população)
    ids = space.ids()
    valores_por_feature: dict[tuple[str, str], list[Optional[float]]] = {}
    processo_por_feature: dict[tuple[str, str], str] = {}

    for domain in representation.domains:
        key_to_process = {obs.key: obs.process for obs in domain.observables if obs.process is not None}
        if not key_to_process:
            continue
        for system_id in ids:
            vector = space.get(system_id)
            for feature in vector.components.get(domain.name, []):
                if feature.name not in key_to_process:
                    continue
                feature_id = (domain.name, feature.name)
                processo_por_feature[feature_id] = key_to_process[feature.name]
                valores_por_feature.setdefault(feature_id, []).append(
                    feature.raw_value if (feature.raw_value is not None and not feature.is_missing) else None
                )

    feature_ids = list(valores_por_feature.keys())
    same_process_pairs: list[tuple[str, str, float]] = []
    different_process_pairs: list[tuple[str, str, float]] = []

    for i in range(len(feature_ids)):
        for j in range(i + 1, len(feature_ids)):
            fid_a, fid_b = feature_ids[i], feature_ids[j]
            a = valores_por_feature[fid_a]
            b = valores_por_feature[fid_b]
            pares_completos = [(va, vb) for va, vb in zip(a, b) if va is not None and vb is not None]
            if len(pares_completos) < 5:
                continue
            arr_a = np.array([p[0] for p in pares_completos])
            arr_b = np.array([p[1] for p in pares_completos])
            if np.std(arr_a) < 1e-12 or np.std(arr_b) < 1e-12:
                continue
            r = float(np.corrcoef(arr_a, arr_b)[0, 1])
            nome_a, nome_b = f"{fid_a[0]}.{fid_a[1]}", f"{fid_b[0]}.{fid_b[1]}"
            if processo_por_feature[fid_a] == processo_por_feature[fid_b]:
                same_process_pairs.append((nome_a, nome_b, abs(r)))
            else:
                different_process_pairs.append((nome_a, nome_b, abs(r)))

    report = ProcessCoherenceReport(same_process_pairs=same_process_pairs, different_process_pairs=different_process_pairs)
    if same_process_pairs:
        report.mean_same_process = float(np.mean([p[2] for p in same_process_pairs]))
    if different_process_pairs:
        report.mean_different_process = float(np.mean([p[2] for p in different_process_pairs]))
    if same_process_pairs and different_process_pairs:
        stat, p = scipy_stats.mannwhitneyu(
            [p[2] for p in same_process_pairs], [p[2] for p in different_process_pairs], alternative="greater"
        )
        report.mannwhitney_p = float(p)
    return report
