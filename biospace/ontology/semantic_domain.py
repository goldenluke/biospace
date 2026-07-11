"""
biospace.ontology.semantic_domain
=====================================

DomainRegistry: catálogo dos SemanticDomains conhecidos pelo sistema —
análogo a ObservableRegistry, mas no nível dos domínios. Não redefine
`SemanticDomain` (isso vive em `biospace.core.domain`).
"""

from __future__ import annotations

from biospace.core import SemanticDomain

__all__ = ["DomainRegistry"]


class DomainRegistry:
    def __init__(self):
        self._by_name: dict[str, SemanticDomain] = {}
        self._registered_by: dict[str, str] = {}

    def register(self, domain: SemanticDomain, registered_by: str = "") -> None:
        existing = self._by_name.get(domain.name)
        if existing is not None and existing.description != domain.description:
            raise ValueError(
                f"Domínio '{domain.name}' já registrado com description={existing.description!r} "
                f"(por {self._registered_by.get(domain.name)!r}), mas uma nova definição com "
                f"description={domain.description!r} foi encontrada em {registered_by!r}."
            )
        self._by_name[domain.name] = domain
        if registered_by:
            self._registered_by[domain.name] = registered_by

    def get(self, name: str) -> SemanticDomain:
        return self._by_name[name]

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def all(self) -> list[SemanticDomain]:
        return list(self._by_name.values())

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        return f"DomainRegistry(n={len(self)})"
