"""
tests.test_contracts_composition_independence_interop
=========================================================

Os 3 contratos formais que faltavam (ver README, seção "Contratos
formais"): 5.3 Composicionalidade, 5.6 Independência Algorítmica, 5.10
Interoperabilidade. (5.9 Versionabilidade já estava coberta por
`check_representation_compatibility`, só não estava rotulada assim.)

Cada um testado tanto no caso POSITIVO (deveria passar) quanto num caso
NEGATIVO decisivo (deveria falhar) — provar que o contrato discrimina de
verdade, não que só roda sem erro.
"""

from __future__ import annotations

from datetime import datetime

from biospace.core import BiologicalSystem, CompositeRepresentation, Feature, Observable, Observation, SemanticDomain
from biospace.core.contracts import check_algorithmic_independence, check_compositionality, check_interoperability


class _FlagObservable(Observable):
    def __init__(self, key):
        self.key = key


class _SimpleDomain(SemanticDomain):
    def __init__(self, name: str, keys: list[str]):
        self.name = name
        self._keys = keys
        super().__init__([_FlagObservable(k) for k in keys])

    def encode(self, measurements):
        return [Feature(name=k, value=float(measurements[k].value), raw_value=float(measurements[k].value)) for k in self._keys if k in measurements]


def _make_system():
    system = BiologicalSystem(identifier="p1")
    system.observe(
        Observation(
            timestamp=datetime(2024, 1, 1),
            source="teste",
            values={"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
        )
    )
    return system


# =============================================================================
# 5.3 Composicionalidade
# =============================================================================
def test_compositionality_holds_for_simple_grouping():
    domain_x = _SimpleDomain("x", ["a", "b"])
    domain_y = _SimpleDomain("y", ["c", "d"])
    grupo = CompositeRepresentation("grupo", [domain_x, domain_y])

    relatorio = check_compositionality(grupo, _make_system())
    assert relatorio.is_compositional
    assert relatorio.n_features_composite == 4
    assert relatorio.n_features_leaves_sum == 4


def test_compositionality_holds_recursively_for_nested_groups():
    """O caso não trivial: aninhamento de 2 níveis (grupo dentro de grupo) deveria continuar batendo, não só o caso raso."""
    domain_x = _SimpleDomain("x", ["a"])
    domain_y = _SimpleDomain("y", ["b"])
    domain_z = _SimpleDomain("z", ["c", "d"])

    subgrupo = CompositeRepresentation("subgrupo", [domain_x, domain_y])
    grupo_externo = CompositeRepresentation("externo", [subgrupo, domain_z])

    relatorio = check_compositionality(grupo_externo, _make_system())
    assert relatorio.is_compositional
    assert relatorio.n_features_composite == 4


def test_compositionality_report_counts_all_leaves_even_with_deep_nesting():
    domain_a = _SimpleDomain("da", ["a"])
    domain_b = _SimpleDomain("db", ["b"])
    domain_c = _SimpleDomain("dc", ["c"])
    domain_d = _SimpleDomain("dd", ["d"])

    nivel1 = CompositeRepresentation("nivel1", [domain_a, domain_b])
    nivel2 = CompositeRepresentation("nivel2", [nivel1, domain_c])
    nivel3 = CompositeRepresentation("nivel3", [nivel2, domain_d])

    relatorio = check_compositionality(nivel3, _make_system())
    assert relatorio.is_compositional
    assert relatorio.n_features_composite == 4
    assert len(nivel3.leaf_domains()) == 4


# =============================================================================
# 5.6 Independência Algorítmica
# =============================================================================
def test_algorithmic_independence_holds_for_a_generic_algorithm():
    """Um algoritmo genuinamente genérico (só olha para o vetor, nunca para nomes de domínio) deveria funcionar em espaços de origens diferentes."""

    class DistanciaGenerica:
        def calcular(self, x, y):
            return float(sum((a - b) ** 2 for a, b in zip(x, y)) ** 0.5)

    def aplicar(algoritmo, vetores_par):
        x, y = vetores_par
        return algoritmo.calcular(x, y)

    espaco_a = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    espaco_b = ([1.0, 2.0], [3.0, 4.0])

    relatorio = check_algorithmic_independence(DistanciaGenerica, espaco_a, espaco_b, aplicar)
    assert relatorio.is_independent


def test_algorithmic_independence_fails_for_domain_specific_algorithm():
    """
    O TESTE DECISIVO: um algoritmo com dependência HARDCODED de um nome
    de domínio específico deveria FALHAR o contrato quando aplicado a um
    espaço que não tem esse domínio -- prova que o contrato discrimina
    de verdade, não que só roda sem erro em qualquer coisa.
    """

    class AlgoritmoEspecificoDeGlicemia:
        def calcular(self, dados_com_glicemia):
            return dados_com_glicemia["glycemic.hba1c_pct"]

    def aplicar(algoritmo, dados):
        return algoritmo.calcular(dados)

    espaco_com_glicemia = {"glycemic.hba1c_pct": 7.0}
    espaco_sem_glicemia = {"apnea.ido": 20.0}

    relatorio = check_algorithmic_independence(AlgoritmoEspecificoDeGlicemia, espaco_com_glicemia, espaco_sem_glicemia, aplicar)
    assert not relatorio.is_independent
    assert relatorio.ran_successfully_on_a
    assert not relatorio.ran_successfully_on_b
    assert "glycemic.hba1c_pct" in (relatorio.error_b or "")


# =============================================================================
# 5.10 Interoperabilidade
# =============================================================================
def test_interoperability_holds_for_a_normal_representation_vector():
    from biospace.core import Cohort, Representation

    domain = _SimpleDomain("d", ["a", "b"])
    representation = Representation([domain])
    cohort = Cohort()
    system = _make_system()
    cohort.update(system, representation, timestamp=datetime(2024, 1, 1))

    vetor = cohort.trajectories[system.id].latest()
    relatorio = check_interoperability(vetor)
    assert relatorio.is_interoperable
    assert relatorio.max_absolute_error == 0.0
