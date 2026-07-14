"""
biospace.ontology.knowledge_graph
=====================================

Constrói o grafo Paciente → Domínio → Observable → Measurement →
Ontologia (a estrutura sugerida no item #35 da lista de técnicas de
IA aplicáveis sobre BioSpace) para um sistema biológico real, usando
`networkx` (já dependência do projeto, via `biospace.graph`).

Este é um grafo POR PACIENTE — uma "carta de apresentação" navegável
do que se sabe sobre um sistema biológico e a que terminologia formal
cada medição corresponde —, não o grafo de similaridade entre
pacientes já construído em `biospace.graph` (esse é populacional, este
é individual).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biospace.core import BiologicalSystem, Representation

from .terminology import lookup_loinc, lookup_snomed

__all__ = ["build_patient_knowledge_graph"]


def build_patient_knowledge_graph(system: "BiologicalSystem", representation: "Representation", at=None):
    """Devolve um `networkx.DiGraph` com nós tipados (patient, domain,
    observable, measurement, ontology_term) e arestas rotuladas
    ('has_domain', 'has_observable', 'has_measurement', 'maps_to').
    Observables sem correspondência conhecida em LOINC/SNOMED simplesmente
    não recebem nó de ontologia -- reportado como está, não inventado."""
    import networkx as nx

    g = nx.DiGraph()
    no_paciente = f"patient:{system.id}"
    g.add_node(no_paciente, tipo="patient")

    for dominio in representation.domains:
        no_dominio = f"domain:{dominio.name}"
        g.add_node(no_dominio, tipo="domain")
        g.add_edge(no_paciente, no_dominio, relacao="has_domain")

        for observable in dominio.observables:
            no_observable = f"observable:{observable.key}"
            g.add_node(no_observable, tipo="observable", unit=observable.unit, process=observable.process)
            g.add_edge(no_dominio, no_observable, relacao="has_observable")

            medicao = observable.extract(system, as_of=at)
            if medicao is not None:
                no_medicao = f"measurement:{system.id}:{observable.key}:{medicao.timestamp.isoformat()}"
                g.add_node(no_medicao, tipo="measurement", valor=medicao.value, fonte=medicao.source, timestamp=medicao.timestamp)
                g.add_edge(no_observable, no_medicao, relacao="has_measurement")

            termo = lookup_loinc(observable.key) or lookup_snomed(observable.key)
            if termo is not None:
                no_ontologia = f"ontology:{termo.system}:{termo.code}"
                g.add_node(no_ontologia, tipo="ontology_term", sistema=termo.system, codigo=termo.code, nome=termo.display)
                g.add_edge(no_observable, no_ontologia, relacao="maps_to")

    return g
