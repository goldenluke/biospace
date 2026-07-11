"""
biospace.core.representation
===============================

Representation (R): compõe os φ_i individuais de cada domínio em uma
representação única (Princípio da Composicionalidade, Seção 3.2). Não
implementa nenhum algoritmo de aprendizado — apenas orquestra domínios.

RepresentationVector: um ponto x = R(B) no espaço de representação, em um
instante t, guardado como {nome_do_domínio: list[Feature]} — não como um
array plano — para preservar a proveniência de cada coordenada.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .feature import Feature, features_to_array

if TYPE_CHECKING:
    from .biological_system import BiologicalSystem
    from .domain import SemanticDomain

__all__ = ["RepresentationVector", "Representation"]


@dataclass
class RepresentationVector:
    system_id: str
    timestamp: datetime
    components: dict[str, list[Feature]]

    def as_vector(self, domain_order: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatena os componentes em um único vetor numérico (projeção conveniente para X = R^n)."""
        names = domain_order or sorted(self.components.keys())
        if not names:
            return np.array([], dtype=float)
        missing = [n for n in names if n not in self.components]
        if missing:
            raise KeyError(
                f"Domínio(s) {missing} não encontrado(s) em `components` (disponíveis: "
                f"{sorted(self.components.keys())}) — `domain_order` deve corresponder aos domínios "
                "realmente presentes neste RepresentationVector."
            )
        return np.concatenate([features_to_array(self.components[n]) for n in names])

    def features_flat(self, domain_order: Optional[Sequence[str]] = None) -> list[Feature]:
        """Todas as Features em ordem — útil para auditoria (nome, valor, proveniência)."""
        names = domain_order or sorted(self.components.keys())
        missing = [n for n in names if n not in self.components]
        if missing:
            raise KeyError(
                f"Domínio(s) {missing} não encontrado(s) em `components` (disponíveis: "
                f"{sorted(self.components.keys())})."
            )
        flat: list[Feature] = []
        for n in names:
            flat.extend(self.components[n])
        return flat

    def __repr__(self) -> str:
        dims = {k: len(v) for k, v in self.components.items()}
        return f"RepresentationVector(system={self.system_id}, t={self.timestamp}, dims={dims})"


class Representation:
    """
    R : B -> X

    R(B) = (φ_1(D_1), φ_2(D_2), ..., φ_n(D_n))
    """

    def __init__(self, domains: Sequence["SemanticDomain"]):
        domains = list(domains)
        seen: dict[str, int] = {}
        for domain in domains:
            seen[domain.name] = seen.get(domain.name, 0) + 1
        colliding = {name: count for name, count in seen.items() if count > 1}
        if colliding:
            raise ValueError(
                f"Domínios com nomes colidentes na mesma Representation: {colliding} — "
                "isso faria um domínio SOBRESCREVER SILENCIOSAMENTE o outro no dicionário de "
                "componentes (mesmo bug encontrado manualmente ao testar check_extensibility(); "
                "ver README). Dê nomes distintos a cada domínio."
            )
        self.domains: list["SemanticDomain"] = domains

    def transform(self, system: "BiologicalSystem", timestamp: Optional[datetime] = None) -> RepresentationVector:
        """
        Contrato 5.7 (Temporalidade): quando `timestamp` é informado, ele
        é usado NÃO SÓ como rótulo do ponto resultante, mas também como
        corte temporal (`as_of`) repassado a cada domínio — garantindo
        que Observations POSTERIORES a `timestamp` (ex.: um
        preenchimento retroativo de histórico fora de ordem) não
        contaminem um ponto que deveria refletir apenas o que era
        conhecido até aquele instante. Sem `timestamp` (None), usa TODAS
        as observações disponíveis no sistema — comportamento anterior,
        preservado para retrocompatibilidade.
        """
        components = {domain.name: domain.transform(system, as_of=timestamp) for domain in self.domains}
        return RepresentationVector(
            system_id=system.id,
            timestamp=timestamp or datetime.now(),
            components=components,
        )

    def domain_names(self) -> list[str]:
        return [d.name for d in self.domains]

    def processes(self) -> set[str]:
        """União dos PhysiologicalProcess declarados por todos os domínios desta Representation (ver `SemanticDomain.processes()`)."""
        result: set[str] = set()
        for domain in self.domains:
            result |= domain.processes()
        return result

    def features_by_process(self, vector: "RepresentationVector") -> dict[str, list[Feature]]:
        """
        Agrupa as Features de `vector` pelo PhysiologicalProcess que seu
        Observable de origem declara — cruzando FRONTEIRAS DE DOMÍNIO:
        duas Features de domínios diferentes que declaram o mesmo
        processo aparecem juntas aqui, mesmo vindo de `components`
        diferentes dentro do vetor (o cenário que motivou esta camada:
        uma medida de colesterol pode alimentar tanto um domínio
        metabólico quanto um cardiovascular através do mesmo processo
        "lipid_metabolism"). Features cujo Observable não declara
        processo (o caso comum — plugins que nunca usaram esta camada,
        como o sleep) não aparecem aqui, não sob uma chave "None" espúria.
        """
        result: dict[str, list[Feature]] = {}
        for domain in self.domains:
            key_to_process = {obs.key: obs.process for obs in domain.observables if obs.process is not None}
            for feature in vector.components.get(domain.name, []):
                process = key_to_process.get(feature.name)
                if process is not None:
                    result.setdefault(process, []).append(feature)
        return result

    def extend(self, domain: "SemanticDomain") -> "Representation":
        """Contrato 5.5 (Extensibilidade): R' = R ⊕ R_{n+1}, sem alterar `self`."""
        return Representation(self.domains + [domain])

    def __repr__(self) -> str:
        return f"Representation(domains={self.domain_names()})"
