"""
biospace.ontology.verification
=================================

ContractSuiteReport / run_contract_suite: consolida TODOS os contratos
formais implementados (núcleo + fenotipagem) em uma única checagem,
produzindo um relatório de conformidade para uma Representation (e,
opcionalmente, uma Cohort/RepresentationSpace e um PhenotypingOperator)
— a forma mais direta de tornar a teoria "verificável" na prática.

Cada contrato só é checado se os dados necessários para ele forem
informados; omitir um argumento pula aquele contrato específico (listado
em `report.skipped`), sem falhar a suíte inteira.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from biospace.core.contracts import (
    AlgorithmicIndependenceReport,
    CompositionalityReport,
    InjectivityReport,
    InteroperabilityReport,
    TemporalityReport,
    VersionCompatibilityReport,
    check_algorithmic_independence,
    check_compositionality,
    check_extensibility,
    check_injectivity,
    check_interoperability,
    check_lipschitz_continuity,
    check_reproducibility,
    check_representation_schema_compatibility,
    check_semantic_preservation,
    check_temporality,
)
from biospace.core.feature import features_to_array

if TYPE_CHECKING:
    from biospace.core import BiologicalSystem, CompositeRepresentation, Observation, Representation, RepresentationSpace, SemanticDomain
    from biospace.phenotyping import PhenotypingOperator
    from biospace.phenotyping.contracts import StabilityReport

__all__ = ["ContractSuiteReport", "run_contract_suite"]


@dataclass
class ContractSuiteReport:
    reproducibility: Optional[bool] = None
    semantic_preservation: Optional[bool] = None
    lipschitz_constant: Optional[float] = None
    extensibility: Optional[bool] = None
    injectivity: Optional[InjectivityReport] = None
    schema_compatibility: Optional[dict[str, VersionCompatibilityReport]] = None
    phenotype_stability: Optional["StabilityReport"] = None
    temporality: Optional[TemporalityReport] = None
    compositionality: Optional[CompositionalityReport] = None
    algorithmic_independence: Optional[AlgorithmicIndependenceReport] = None
    interoperability: Optional[InteroperabilityReport] = None
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Relatório de Conformidade com os Contratos Formais", "=" * 55]

        if self.reproducibility is not None:
            lines.append(f"[5.8 Reprodutibilidade]        {'OK' if self.reproducibility else 'FALHOU'}")
        if self.semantic_preservation is not None:
            lines.append(f"[5.2 Preservação Semântica]    {'OK' if self.semantic_preservation else 'FALHOU'}")
        if self.compositionality is not None:
            lines.append(f"[5.3 Composicionalidade]       {'OK' if self.compositionality.is_compositional else 'FALHOU'}")
        if self.lipschitz_constant is not None:
            lines.append(f"[5.4 Continuidade]             L={self.lipschitz_constant:.3f} (finito = OK)")
        if self.extensibility is not None:
            lines.append(f"[5.5 Extensibilidade]          {'OK' if self.extensibility else 'FALHOU'}")
        if self.algorithmic_independence is not None:
            lines.append(f"[5.6 Independência Algorítmica] {'OK' if self.algorithmic_independence.is_independent else 'FALHOU'}")
        if self.injectivity is not None:
            status = "OK" if self.injectivity.is_injective else f"FALHOU ({len(self.injectivity.violations)} colisões)"
            lines.append(f"[Injetividade]                 {status} ({self.injectivity.n_pairs_checked} pares checados)")
        if self.schema_compatibility is not None:
            incompativeis = [n for n, r in self.schema_compatibility.items() if not r.is_compatible]
            status = "OK" if not incompativeis else f"FALHOU ({incompativeis})"
            lines.append(f"[5.9 Versionabilidade]         {status}")
        if self.interoperability is not None:
            lines.append(f"[5.10 Interoperabilidade]      {'OK' if self.interoperability.is_interoperable else 'FALHOU'}")
        if self.phenotype_stability is not None:
            s = self.phenotype_stability
            status = "OK" if s.is_stable else "INSTÁVEL"
            lines.append(f"[8.5 Estabilidade Fenotípica]  ARI={s.adjusted_rand_index:.3f} ({status})")
        if self.temporality is not None:
            status = "OK" if self.temporality.is_compliant else "FALHOU"
            lines.append(f"[5.7 Temporalidade]            {status}")

        if self.skipped:
            lines.append("")
            lines.append("Contratos não checados (dados insuficientes fornecidos):")
            for s in self.skipped:
                lines.append(f"  - {s}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def run_contract_suite(
    representation: "Representation",
    domain: Optional["SemanticDomain"] = None,
    systems: Optional[Sequence["BiologicalSystem"]] = None,
    pairs_for_continuity: Optional[Sequence[tuple["BiologicalSystem", "BiologicalSystem"]]] = None,
    raw_distance_fn: Optional[Callable[["BiologicalSystem", "BiologicalSystem"], float]] = None,
    new_domain_for_extensibility: Optional["SemanticDomain"] = None,
    representation_v2: Optional["Representation"] = None,
    space: Optional["RepresentationSpace"] = None,
    phenotyping_operator_factory: Optional[Callable[[], "PhenotypingOperator"]] = None,
    system_factory: Optional[Callable[[], "BiologicalSystem"]] = None,
    observations_for_temporality: Optional[Sequence["Observation"]] = None,
    composite_for_compositionality: Optional["CompositeRepresentation"] = None,
    algorithm_factory: Optional[Callable[[], Any]] = None,
    algorithm_apply_fn: Optional[Callable[[Any, Any], Any]] = None,
    space_b_for_independence: Optional[Any] = None,
    vector_for_interoperability: Optional[Any] = None,
) -> ContractSuiteReport:
    """Roda todos os contratos aplicáveis, dado o que foi fornecido. Ver docstring do módulo."""
    report = ContractSuiteReport()

    ref_domain = domain or (representation.domains[0] if representation.domains else None)
    ref_system = systems[0] if systems else None

    if ref_domain and ref_system:
        report.reproducibility = check_reproducibility(ref_domain, ref_system)
    else:
        report.skipped.append("reprodutibilidade (informe domain e systems)")

    if ref_domain and systems and len(systems) >= 2:
        report.semantic_preservation = check_semantic_preservation(ref_domain, systems[0], systems[1])
    else:
        report.skipped.append("preservação semântica (informe domain e >=2 systems)")

    if composite_for_compositionality and ref_system:
        report.compositionality = check_compositionality(composite_for_compositionality, ref_system)
    else:
        report.skipped.append("composicionalidade (informe composite_for_compositionality, um CompositeRepresentation)")

    if ref_domain and pairs_for_continuity and raw_distance_fn:
        report.lipschitz_constant = check_lipschitz_continuity(ref_domain, pairs_for_continuity, raw_distance_fn)
    else:
        report.skipped.append("continuidade (informe pairs_for_continuity e raw_distance_fn)")

    if new_domain_for_extensibility and ref_system:
        report.extensibility = check_extensibility(representation, new_domain_for_extensibility, ref_system)
    else:
        report.skipped.append("extensibilidade (informe new_domain_for_extensibility)")

    if algorithm_factory and algorithm_apply_fn and space and space_b_for_independence is not None:
        report.algorithmic_independence = check_algorithmic_independence(
            algorithm_factory, space, space_b_for_independence, algorithm_apply_fn
        )
    else:
        report.skipped.append("independência algorítmica (informe algorithm_factory, algorithm_apply_fn e space_b_for_independence)")

    if ref_domain and systems and len(systems) >= 2:
        report.injectivity = check_injectivity(
            transform=lambda s: features_to_array(ref_domain.transform(s)),
            systems=systems,
        )
    else:
        report.skipped.append("injetividade (informe domain e systems)")

    if representation_v2 and ref_system:
        report.schema_compatibility = check_representation_schema_compatibility(representation, representation_v2, ref_system)
    else:
        report.skipped.append("versionabilidade / compatibilidade de esquema (informe representation_v2)")

    if vector_for_interoperability is not None:
        report.interoperability = check_interoperability(vector_for_interoperability)
    else:
        report.skipped.append("interoperabilidade (informe vector_for_interoperability, um RepresentationVector)")

    if phenotyping_operator_factory and space:
        from biospace.phenotyping.contracts import check_phenotype_stability

        report.phenotype_stability = check_phenotype_stability(phenotyping_operator_factory, space)
    else:
        report.skipped.append("estabilidade fenotípica (informe phenotyping_operator_factory e space)")

    if system_factory and observations_for_temporality:
        report.temporality = check_temporality(representation, system_factory, observations_for_temporality)
    else:
        report.skipped.append("temporalidade (informe system_factory e observations_for_temporality)")

    return report
