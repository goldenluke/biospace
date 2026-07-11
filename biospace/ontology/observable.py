"""
biospace.ontology.observable
===============================

ObservableRegistry: um catálogo de Observables conhecidos pelo sistema —
não redefine `Observable` (isso já existe em `biospace.core.observation`),
apenas organiza instâncias já criadas por plugins, detectando conflitos
(a mesma chave registrada com unidade/descrição diferentes em lugares
diferentes é, quase sempre, um erro de modelagem, não uma coincidência).
"""

from __future__ import annotations

from biospace.core import Observable

__all__ = ["ObservableRegistry", "OntologyConflictError"]


class OntologyConflictError(ValueError):
    """Levantado quando o mesmo Observable.key é registrado com semântica divergente."""


class ObservableRegistry:
    def __init__(self):
        self._by_key: dict[str, Observable] = {}
        self._registered_by: dict[str, list[str]] = {}

    def register(self, observable: Observable, registered_by: str = "") -> None:
        existing = self._by_key.get(observable.key)
        if existing is not None:
            if existing.unit != observable.unit or existing.description != observable.description:
                raise OntologyConflictError(
                    f"Observable '{observable.key}' já registrado com "
                    f"unit={existing.unit!r}, description={existing.description!r} "
                    f"(por {self._registered_by.get(observable.key)}), mas uma nova "
                    f"definição com unit={observable.unit!r}, description={observable.description!r} "
                    f"foi encontrada em {registered_by!r}. Se são de fato a mesma grandeza, "
                    "unifique a definição; se não são, use uma chave diferente."
                )
        else:
            self._by_key[observable.key] = observable
        self._registered_by.setdefault(observable.key, [])
        if registered_by and registered_by not in self._registered_by[observable.key]:
            self._registered_by[observable.key].append(registered_by)

    def get(self, key: str) -> Observable:
        return self._by_key[key]

    def registered_by(self, key: str) -> list[str]:
        return list(self._registered_by.get(key, []))

    def keys(self) -> list[str]:
        return list(self._by_key.keys())

    def all(self) -> list[Observable]:
        return list(self._by_key.values())

    def __contains__(self, key: str) -> bool:
        return key in self._by_key

    def __len__(self) -> int:
        return len(self._by_key)

    def __repr__(self) -> str:
        return f"ObservableRegistry(n={len(self)})"
