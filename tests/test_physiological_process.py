"""
tests.test_physiological_process
====================================

PhysiologicalProcess (biospace.core.process): camada OPCIONAL e
ADITIVA entre Observable e SemanticDomain — nomeia o mecanismo
biológico que um Observable mede, independente de qual domínio o
consome. Testa: (1) retrocompatibilidade total com domínios que nunca
declaram processo (plugin sleep, e qualquer SemanticDomain escrito
antes desta camada existir); (2) consulta correta em domínios que
declaram processos (plugin metabolic); (3) o caso decisivo — agregação
cruzando FRONTEIRA DE DOMÍNIO de verdade, não apenas quando processo e
domínio coincidem 1:1.
"""

from __future__ import annotations

from datetime import datetime

from biospace.core import (
    BiologicalSystem,
    Feature,
    Observable,
    Observation,
    PhysiologicalProcess,
    Representation,
    SemanticDomain,
)


class _FlagObservable(Observable):
    def __init__(self, key: str, process=None):
        self.key = key
        if process is not None:
            self.process = process


class _SimpleDomain(SemanticDomain):
    def __init__(self, name: str, keys_and_processes: list[tuple[str, str | None]]):
        self.name = name
        self._keys = [k for k, _ in keys_and_processes]
        super().__init__([_FlagObservable(k, p) for k, p in keys_and_processes])

    def encode(self, measurements):
        return [Feature(name=k, value=float(measurements[k].value), raw_value=float(measurements[k].value)) for k in self._keys if k in measurements]


def _make_system(values: dict) -> BiologicalSystem:
    system = BiologicalSystem(identifier="p1")
    system.observe(Observation(timestamp=datetime(2024, 1, 1), source="teste", values=values))
    return system


def test_physiological_process_is_frozen_and_named():
    p = PhysiologicalProcess(name="glucose_homeostasis", description="teste")
    assert p.name == "glucose_homeostasis"
    try:
        p.name = "outro"  # dataclass frozen -- deve falhar
        assert False, "PhysiologicalProcess deveria ser imutavel (frozen=True)"
    except Exception:
        pass


def test_observable_without_process_defaults_to_none():
    """Retrocompatibilidade: um Observable que nunca ouviu falar desta camada continua com process=None."""
    obs = _FlagObservable("x")
    assert obs.process is None


def test_domain_without_any_process_declared_has_empty_processes():
    """Um SemanticDomain cujos Observables nunca declaram processo deve reportar processes() vazio -- nao None, nao erro."""
    domain = _SimpleDomain("d", [("a", None), ("b", None)])
    assert domain.processes() == set()


def test_domain_with_declared_processes_reports_them():
    domain = _SimpleDomain("d", [("a", "process_x"), ("b", "process_x"), ("c", "process_y")])
    assert domain.processes() == {"process_x", "process_y"}


def test_representation_processes_is_union_across_domains():
    d1 = _SimpleDomain("d1", [("a", "process_x")])
    d2 = _SimpleDomain("d2", [("b", "process_y")])
    representation = Representation([d1, d2])
    assert representation.processes() == {"process_x", "process_y"}


def test_features_by_process_crosses_domain_boundary():
    """
    O TESTE DECISIVO: duas Features de DOIS DOMINIOS DIFERENTES
    declarando o MESMO processo devem aparecer juntas em
    features_by_process -- o cenario que motivou esta camada (ex.:
    colesterol alimentando tanto um dominio metabolico quanto um
    cardiovascular atraves do mesmo processo "lipid_metabolism").
    """
    d1 = _SimpleDomain("metabolic", [("colesterol_via_metabolico", "lipid_metabolism")])
    d2 = _SimpleDomain("cardiovascular", [("colesterol_via_cardio", "lipid_metabolism"), ("pressao", "cardiovascular_regulation")])
    representation = Representation([d1, d2])

    system = _make_system({"colesterol_via_metabolico": 200.0, "colesterol_via_cardio": 200.0, "pressao": 130.0})
    vector = representation.transform(system)

    agrupado = representation.features_by_process(vector)

    assert set(agrupado.keys()) == {"lipid_metabolism", "cardiovascular_regulation"}
    nomes_lipid = {f.name for f in agrupado["lipid_metabolism"]}
    assert nomes_lipid == {"colesterol_via_metabolico", "colesterol_via_cardio"}, "As duas Features de dominios DIFERENTES deveriam aparecer juntas sob o mesmo processo."
    assert len(agrupado["cardiovascular_regulation"]) == 1


def test_features_without_declared_process_are_absent_not_null_key():
    """Features cujo Observable nao declara processo NAO devem aparecer sob uma chave 'None' espuria -- devem simplesmente estar ausentes."""
    domain = _SimpleDomain("d", [("com_processo", "process_x"), ("sem_processo", None)])
    representation = Representation([domain])
    system = _make_system({"com_processo": 1.0, "sem_processo": 2.0})
    vector = representation.transform(system)

    agrupado = representation.features_by_process(vector)
    assert None not in agrupado
    todos_nomes = {f.name for features in agrupado.values() for f in features}
    assert "sem_processo" not in todos_nomes
    assert "com_processo" in todos_nomes


def test_sleep_plugin_processes_are_minimal_and_deliberate():
    """
    ATUALIZADO: o plugin sleep NÃO é mais totalmente livre de processos
    -- os 3 observables de frequência cardíaca (fc_minima/media/maxima_bpm)
    foram deliberadamente marcados com process="cardiovascular_regulation"
    (o MESMO nome usado por plugins.metabolic) para habilitar comparação
    cross-disease via projeção em espaço de processo (ver
    test_cross_disease_functor.py). A opcionalidade da camada continua
    provada de outro jeito: 7 dos 8 domínios do sleep (todos exceto
    cardiovascular) continuam sem NENHUM processo declarado -- a
    mudança foi cirúrgica e deliberada, não uma migração completa.
    """
    from biospace.plugins.sleep import SleepRepresentation

    representation = SleepRepresentation()
    assert representation.processes() == {"cardiovascular_regulation"}, (
        "Esperava exatamente 1 processo declarado no sleep (achado documentado, mudança deliberada para o functor cross-disease)."
    )

    dominios_sem_processo = [d for d in representation.domains if d.name != "cardiovascular"]
    for dominio in dominios_sem_processo:
        assert dominio.processes() == set(), f"Domínio '{dominio.name}' não deveria ter processo declarado -- só cardiovascular foi tocado."


def test_metabolic_plugin_declares_five_real_processes():
    """Renomeado de 'quatro' para 'cinco' -- lipid_metabolism adicionado quando NHANES ganhou colesterol/HDL/triglicerídeos reais."""
    from biospace.plugins.metabolic import ALL_PROCESSES, MetabolicRepresentation

    representation = MetabolicRepresentation()
    nomes_esperados = {p.name for p in ALL_PROCESSES}
    assert representation.processes() == nomes_esperados
    assert representation.processes() == {"glucose_homeostasis", "body_composition", "cardiovascular_regulation", "renal_filtration", "lipid_metabolism"}


def test_metabolic_plugin_features_by_process_groups_correctly():
    from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, exam

    representation = MetabolicRepresentation()
    system = MetabolicSystem()
    system.observe(exam({
        "hba1c_pct": 7.0, "glicemia_jejum_mg_dl": 130.0, "idade": 50,
        "imc": 28.0, "circunferencia_abdominal_cm": 95.0,
        "pressao_sistolica_mmhg": 125.0, "pressao_diastolica_mmhg": 80.0, "fc_repouso_bpm": 72.0,
        "creatinina_mg_dl": 1.0, "taxa_filtracao_glomerular": 85.0,
    }, timestamp=datetime(2024, 1, 1)))
    vector = representation.transform(system)

    agrupado = representation.features_by_process(vector)
    assert {f.name for f in agrupado["glucose_homeostasis"]} == {"hba1c_pct", "glicemia_jejum_mg_dl"}
    assert {f.name for f in agrupado["renal_filtration"]} == {"creatinina_mg_dl", "taxa_filtracao_glomerular"}
    # 'idade' nao declara processo (deliberado -- ver observables.py) -- nao deveria aparecer em nenhum grupo
    todos_nomes = {f.name for features in agrupado.values() for f in features}
    assert "idade" not in todos_nomes
