"""
biospace.ontology.ontology
=============================

Ontology: une DomainRegistry + ObservableRegistry em um catálogo único,
construído automaticamente a partir de uma Representation (ou registrado
manualmente, domínio a domínio). Serve como vocabulário central de uma
implantação do biospace — e gera um dicionário de dados em Markdown,
útil para documentação automática de um plugin de doença.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .observable import ObservableRegistry
from .semantic_domain import DomainRegistry

if TYPE_CHECKING:
    from biospace.core import Representation, SemanticDomain

__all__ = ["Ontology"]


class Ontology:
    def __init__(self, name: str = "biospace"):
        self.name = name
        self.domains = DomainRegistry()
        self.observables = ObservableRegistry()

    def register_domain(self, domain: "SemanticDomain") -> None:
        self.domains.register(domain, registered_by=self.name)
        for observable in domain.observables:
            self.observables.register(observable, registered_by=domain.name)

    @classmethod
    def from_representation(cls, representation: "Representation", name: str = "") -> "Ontology":
        """Constrói uma Ontology varrendo todos os domínios de uma Representation já composta."""
        ontology = cls(name=name or "representation")
        for domain in representation.domains:
            ontology.register_domain(domain)
        return ontology

    def to_markdown(self) -> str:
        """Gera um dicionário de dados em Markdown — um domínio por seção, um Observable por linha."""
        lines = [f"# Ontologia: {self.name}", "", f"{len(self.domains)} domínios, {len(self.observables)} observables.", ""]
        for domain in self.domains.all():
            lines.append(f"## {domain.name}")
            if domain.description:
                lines.append(f"_{domain.description}_")
            lines.append("")
            lines.append("| Observable | Unidade | Descrição |")
            lines.append("|---|---|---|")
            for obs in domain.observables:
                lines.append(f"| `{obs.key}` | {obs.unit or '—'} | {obs.description or '—'} |")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Ontology({self.name!r}, n_domains={len(self.domains)}, n_observables={len(self.observables)})"
