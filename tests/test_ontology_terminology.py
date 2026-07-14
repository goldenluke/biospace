"""
tests.test_ontology_terminology
===================================

`biospace.ontology.terminology` (LOINC/SNOMED) e
`biospace.ontology.knowledge_graph` — todo código listado foi
verificado por busca contra fonte pública (loinc.org, SNOMED CT)
antes de ser incluído no módulo, não lembrado de memória. Estes
testes confirmam a MECÂNICA do módulo (busca correta, grafo bem
formado, cobertura honesta), não os códigos em si — isso já foi
verificado externamente, uma vez, no momento da escrita.
"""

from __future__ import annotations

from datetime import datetime

from biospace.core import Observation
from biospace.ontology import (
    build_patient_knowledge_graph,
    coverage_report,
    lookup_loinc,
    lookup_snomed,
)
from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem


def test_lookup_loinc_returns_known_code_for_hba1c():
    resultado = lookup_loinc("hba1c_pct")
    assert resultado is not None
    assert resultado.system == "LOINC"
    assert resultado.code == "4548-4"


def test_lookup_loinc_returns_none_for_unmapped_key():
    """Uma chave sem codigo verificado deveria devolver None -- nao um palpite, nao um erro."""
    assert lookup_loinc("chave_que_nao_existe_em_lugar_nenhum") is None


def test_lookup_snomed_returns_known_code_for_diabetes():
    resultado = lookup_snomed("diabetes_tipo_2")
    assert resultado is not None
    assert resultado.system == "SNOMED CT"
    assert resultado.code == "44054006"


def test_coverage_report_is_honest_about_unmapped_observables():
    """TESTE DECISIVO: o relatorio de cobertura deveria reportar EXPLICITAMENTE quais chaves nao tem codigo, nao esconder ou inventar. 'sexo' e' o caso real ainda sem codigo verificado no registro."""
    relatorio = coverage_report(["hba1c_pct", "idade", "sexo", "hipertensao"])
    assert relatorio["hba1c_pct"] is not None
    assert relatorio["idade"] is not None  # LOINC 30525-0, verificado
    assert relatorio["sexo"] is None  # nenhum codigo administrativo de sexo verificado com confianca ainda
    assert relatorio["hipertensao"] is not None  # SNOMED CT 38341003, verificado


def test_knowledge_graph_has_correct_node_type_counts():
    system = MetabolicSystem(identifier="paciente_teste")
    system.observe(Observation(
        timestamp=datetime(2024, 1, 15), source="lab",
        values={"hba1c_pct": 7.2, "glicemia_jejum_mg_dl": 145, "idade": 58},
    ))
    representation = MetabolicRepresentation()
    g = build_patient_knowledge_graph(system, representation)

    tipos = {}
    for _, dados in g.nodes(data=True):
        tipos[dados["tipo"]] = tipos.get(dados["tipo"], 0) + 1

    assert tipos["patient"] == 1
    assert tipos["domain"] == len(representation.domains)
    assert tipos["measurement"] == 3
    assert tipos["ontology_term"] >= 10


def test_knowledge_graph_links_observable_to_both_measurement_and_ontology_term():
    """TESTE DECISIVO: hba1c_pct, que TEM medicao real E codigo de terminologia verificado, deveria ter arestas pras duas coisas -- nao uma ou outra."""
    system = MetabolicSystem(identifier="paciente_teste2")
    system.observe(Observation(timestamp=datetime(2024, 1, 15), source="lab", values={"hba1c_pct": 6.8}))
    representation = MetabolicRepresentation()
    g = build_patient_knowledge_graph(system, representation)

    vizinhos = list(g.successors("observable:hba1c_pct"))
    tipos_vizinhos = {g.nodes[v]["tipo"] for v in vizinhos}
    assert "measurement" in tipos_vizinhos
    assert "ontology_term" in tipos_vizinhos


def test_knowledge_graph_observable_without_measurement_has_no_measurement_node():
    """Um Observable nunca observado nao deveria ter no de measurement -- nao inventar um valor."""
    system = MetabolicSystem(identifier="paciente_vazio")
    representation = MetabolicRepresentation()
    g = build_patient_knowledge_graph(system, representation)

    vizinhos = list(g.successors("observable:hba1c_pct"))
    tipos_vizinhos = {g.nodes[v]["tipo"] for v in vizinhos}
    assert "measurement" not in tipos_vizinhos
