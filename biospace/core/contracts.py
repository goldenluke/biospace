"""
biospace.core.contracts
==========================

Verificações empíricas (testes de sanidade, não provas matemáticas) para
os contratos formais definidos na teoria:

  5.1  Rastreabilidade (indiretamente, via VersionCompatibilityReport)
  5.2  Preservação Semântica
  5.3  Composicionalidade
  5.4  Continuidade (Lipschitz)
  5.5  Extensibilidade
  5.6  Independência Algorítmica
  5.7  Temporalidade (comprimento da trajetória, unicidade do sistema,
       ordem cronológica, ausência de "espiar o futuro")
  5.8  Reprodutibilidade
  5.9  Versionabilidade (via VersionCompatibilityReport — mesmo
       verificador de 5.1, o esquema é o que precisa ser estável entre
       versões)
  5.10 Interoperabilidade
  —    Injetividade (varredura populacional, não apenas pares)

Operam sobre SemanticDomain / Representation / BiologicalSystem do
núcleo — nunca sobre conceitos de um plugin específico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .biological_system import BiologicalSystem
from .cohort import Cohort
from .composite_representation import CompositeRepresentation
from .domain import SemanticDomain
from .feature import features_to_array
from .observation import Observation
from .representation import Representation

__all__ = [
    "check_reproducibility",
    "check_semantic_preservation",
    "check_lipschitz_continuity",
    "check_extensibility",
    "check_injectivity",
    "check_representation_compatibility",
    "check_representation_schema_compatibility",
    "check_temporality",
    "check_compositionality",
    "check_algorithmic_independence",
    "check_interoperability",
    "check_domain_update_independence",
    "InjectivityViolation",
    "InjectivityReport",
    "VersionCompatibilityReport",
    "TemporalityReport",
    "CompositionalityReport",
    "AlgorithmicIndependenceReport",
    "InteroperabilityReport",
    "DomainIndependenceReport",
]


def check_reproducibility(domain: SemanticDomain, system: BiologicalSystem) -> bool:
    """Contrato 5.8: R(O1) = R(O2) se O1 = O2 (determinismo de φ_i)."""
    x1 = features_to_array(domain.transform(system))
    x2 = features_to_array(domain.transform(system))
    return bool(np.allclose(x1, x2))


def check_semantic_preservation(
    domain: SemanticDomain, system_a: BiologicalSystem, system_b: BiologicalSystem, tol: float = 1e-9
) -> bool:
    """Contrato 5.2: estados fisiológicos diferentes -> representações diferentes."""
    values_a = system_a.latest_values()
    values_b = system_b.latest_values()
    if values_a == values_b:
        return True  # nada a preservar
    x_a = features_to_array(domain.transform(system_a))
    x_b = features_to_array(domain.transform(system_b))
    return not bool(np.allclose(x_a, x_b, atol=tol))


def check_lipschitz_continuity(
    domain: SemanticDomain,
    pairs: Sequence[tuple[BiologicalSystem, BiologicalSystem]],
    raw_distance_fn: Callable[[BiologicalSystem, BiologicalSystem], float],
) -> float:
    """
    Contrato 5.4: estima empiricamente
        L = max_i d_X(R(B1), R(B2)) / d_B(B1, B2)
    sobre um conjunto de pares. Um valor finito e estável sugere
    continuidade adequada.
    """
    ratios = []
    for system_a, system_b in pairs:
        d_bio = raw_distance_fn(system_a, system_b)
        if d_bio == 0:
            continue
        x_a = features_to_array(domain.transform(system_a))
        x_b = features_to_array(domain.transform(system_b))
        d_repr = float(np.linalg.norm(x_a - x_b))
        ratios.append(d_repr / d_bio)
    return max(ratios) if ratios else float("nan")


def check_extensibility(representation: Representation, new_domain: SemanticDomain, system: BiologicalSystem) -> bool:
    """Contrato 5.5: R' = R ⊕ R_{n+1} deve preservar todos os componentes anteriores."""
    old = representation.transform(system)
    extended = representation.extend(new_domain)
    new = extended.transform(system)
    for name, features in old.components.items():
        if name not in new.components:
            return False
        if not np.allclose(features_to_array(features), features_to_array(new.components[name])):
            return False
    return True


# =============================================================================
# Injetividade — varredura populacional (não apenas um par de sistemas)
# =============================================================================
@dataclass
class InjectivityViolation:
    """Um par de sistemas fisiologicamente distintos que colidiram na mesma representação."""

    system_id_a: str
    system_id_b: str
    raw_distance: float  # distância nos valores brutos observados (contextualiza o quão "diferentes" eram)


@dataclass
class InjectivityReport:
    """
    Resultado da varredura de injetividade sobre uma população.

    Diferente de `check_semantic_preservation` (que testa UM par por
    chamada), este contrato varre TODOS os pares de uma população de uma
    vez — colisões que só emergem em escala (ex.: duas configurações
    fisiológicas raras, mas distintas, que por acidente numérico caem no
    mesmo ponto do espaço) não aparecem em testes de par único.
    """

    n_systems: int
    n_pairs_checked: int
    violations: list[InjectivityViolation] = field(default_factory=list)

    @property
    def is_injective(self) -> bool:
        return len(self.violations) == 0


def check_injectivity(
    transform: Callable[[BiologicalSystem], np.ndarray],
    systems: Sequence[BiologicalSystem],
    raw_distance_fn: Optional[Callable[[BiologicalSystem, BiologicalSystem], float]] = None,
    tol: float = 1e-9,
) -> InjectivityReport:
    """
    Contrato de Injetividade: dois sistemas biológicos fisiologicamente
    distintos NUNCA devem produzir a mesma representação (dentro de `tol`),
    verificado sobre TODOS os pares de `systems` — não apenas um par
    isolado.

    `transform` é qualquer função `BiologicalSystem -> np.ndarray` — tipicamente
    `lambda s: features_to_array(domain.transform(s))` para um domínio, ou
    `lambda s: representation.transform(s).as_vector(order)` para uma
    Representation completa.

    `raw_distance_fn`, se informado, decide quando dois sistemas são
    "fisiologicamente distintos" (distância > 0); se omitido, usa
    `system.latest_values()` diferentes como critério de distinção.

    Custo O(n²) pares — adequado para populações de até alguns milhares
    de sistemas; para coortes maiores, amostre um subconjunto.
    """
    n = len(systems)
    vectors = [transform(s) for s in systems]
    violations: list[InjectivityViolation] = []
    n_pairs_checked = 0

    for i in range(n):
        for j in range(i + 1, n):
            system_a, system_b = systems[i], systems[j]

            if raw_distance_fn is not None:
                d_raw = raw_distance_fn(system_a, system_b)
                distinct = d_raw > 0
            else:
                distinct = system_a.latest_values() != system_b.latest_values()
                d_raw = float("nan")

            if not distinct:
                continue

            n_pairs_checked += 1
            if np.allclose(vectors[i], vectors[j], atol=tol):
                violations.append(
                    InjectivityViolation(system_id_a=system_a.id, system_id_b=system_b.id, raw_distance=d_raw)
                )

    return InjectivityReport(n_systems=n, n_pairs_checked=n_pairs_checked, violations=violations)


# =============================================================================
# Compatibilidade entre versões de representação — esquema estável
# =============================================================================
@dataclass
class VersionCompatibilityReport:
    """
    Compara o ESQUEMA (quais eixos/Features existem, em que ordem) entre
    duas versões do mesmo domínio — não os valores numéricos. Duas
    versões podem divergir em valor (ex.: `Reference` reajustada sobre
    uma nova população) e ainda assim serem compatíveis, desde que os
    mesmos eixos existam na mesma ordem relativa.
    """

    domain_name: str
    same_feature_names: bool
    same_dimension: bool
    common_features: list[str]
    only_in_v1: list[str]
    only_in_v2: list[str]

    @property
    def is_compatible(self) -> bool:
        return self.same_feature_names and self.same_dimension


def check_representation_compatibility(
    domain_v1: SemanticDomain, domain_v2: SemanticDomain, system: BiologicalSystem
) -> VersionCompatibilityReport:
    """
    Contrato de Compatibilidade entre Versões: duas versões do MESMO
    domínio (mesmo `.name`, mas talvez `Reference` reajustada ou uma
    Feature nova adicionada) devem produzir Features com os MESMOS NOMES
    na mesma ordem — para que R_v1(B) e R_v2(B) continuem comparáveis
    coordenada a coordenada onde se sobrepõem, mesmo que os valores
    numéricos mudem porque a referência estatística mudou.

    Isto NÃO testa se os valores são iguais (eles podem legitimamente
    não ser) — testa se o esquema (quais eixos existem) é estável.
    """
    if domain_v1.name != domain_v2.name:
        raise ValueError(
            f"Domínios com nomes diferentes não são comparáveis como versões: "
            f"{domain_v1.name!r} vs {domain_v2.name!r}."
        )

    names_v1 = [f.name for f in domain_v1.transform(system)]
    names_v2 = [f.name for f in domain_v2.transform(system)]
    set_v1, set_v2 = set(names_v1), set(names_v2)

    return VersionCompatibilityReport(
        domain_name=domain_v1.name,
        same_feature_names=(names_v1 == names_v2),
        same_dimension=(len(names_v1) == len(names_v2)),
        common_features=sorted(set_v1 & set_v2),
        only_in_v1=sorted(set_v1 - set_v2),
        only_in_v2=sorted(set_v2 - set_v1),
    )


def check_representation_schema_compatibility(
    representation_v1: Representation, representation_v2: Representation, system: BiologicalSystem
) -> dict[str, VersionCompatibilityReport]:
    """
    Aplica `check_representation_compatibility` a cada domínio em comum
    entre duas Representations completas (ex.: a mesma `SleepRepresentation`
    antes e depois de `fit_reference()` ser reajustada sobre uma nova
    população, ou antes/depois de `representation.extend(novo_dominio)`).
    """
    domains_v1 = {d.name: d for d in representation_v1.domains}
    domains_v2 = {d.name: d for d in representation_v2.domains}
    common_names = sorted(set(domains_v1) & set(domains_v2))
    return {
        name: check_representation_compatibility(domains_v1[name], domains_v2[name], system)
        for name in common_names
    }


# =============================================================================
# Temporalidade (Seção 5.7 da teoria) — trajetória, não representações isoladas
# =============================================================================
@dataclass
class TemporalityReport:
    """
    Resultado da verificação do Contrato de Temporalidade (5.7):

        "Sistemas biológicos evoluem continuamente... Cada observação
        produz uma ATUALIZAÇÃO da trajetória, e não uma nova
        representação independente. Formalmente, para uma sequência de
        observações O(t1), O(t2),…, O(tn), obtém-se
        Γ = {R(B,t1), R(B,t2),…, R(B,tn)}."

    Quatro propriedades verificadas, cada uma correspondendo a uma parte
    específica do enunciado acima:
    """

    n_observations: int
    n_trajectory_points: int
    n_distinct_systems: int  # deveria ser sempre 1 — "não uma nova representação independente"
    is_chronologically_sorted: bool
    timestamps_match_observations: bool
    no_lookahead: bool  # nenhum ponto da trajetória usa informação de observações futuras

    @property
    def correct_length(self) -> bool:
        """Γ tem exatamente um ponto por observação — nenhuma se perde, nenhuma é duplicada."""
        return self.n_trajectory_points == self.n_observations

    @property
    def single_system(self) -> bool:
        """Todas as observações atualizaram o MESMO sistema — nunca criaram um paciente novo."""
        return self.n_distinct_systems == 1

    @property
    def is_compliant(self) -> bool:
        return (
            self.correct_length
            and self.single_system
            and self.is_chronologically_sorted
            and self.timestamps_match_observations
            and self.no_lookahead
        )

    def summary(self) -> str:
        checks = [
            ("Comprimento correto (1 ponto por observação)", self.correct_length),
            ("Sistema único (nunca criou paciente novo)", self.single_system),
            ("Ordem cronológica preservada", self.is_chronologically_sorted),
            ("Timestamps correspondem às observações", self.timestamps_match_observations),
            ("Sem 'espiar o futuro' (cada ponto usa só o passado)", self.no_lookahead),
        ]
        lines = ["Contrato 5.7 — Temporalidade:"]
        for label, ok in checks:
            lines.append(f"  [{'OK' if ok else 'FALHOU'}] {label}")
        lines.append(f"  => is_compliant = {self.is_compliant}")
        return "\n".join(lines)


def check_temporality(
    representation: Representation,
    system_factory: Callable[[], BiologicalSystem],
    observations: Sequence[Observation],
) -> TemporalityReport:
    """
    Aplica `observations` (em qualquer ordem — a própria trajetória deve
    reordená-las cronologicamente) a um sistema construído por
    `system_factory()`, uma observação de cada vez, através do MESMO
    mecanismo usado em produção (`Cohort.update()` — não uma
    reimplementação paralela), e verifica as 4 propriedades do Contrato
    5.7 descritas em `TemporalityReport`.

    `system_factory`: função sem argumentos que retorna uma NOVA
    instância vazia do tipo de sistema em uso (ex.: `lambda: SleepSystem()`)
    — usada tanto para o sistema principal quanto para reconstruir
    snapshots históricos ao verificar `no_lookahead`.

    IMPORTANTE: `observations` é alimentado na ORDEM DADA pelo chamador
    (não pré-ordenada por este contrato) — propositalmente, para testar
    de verdade se o próprio sistema/trajetória reordena corretamente
    quando uma observação chega fora de ordem (ex.: reprocessamento
    retroativo de histórico). Passe observações fora de ordem para
    exercitar essa capacidade.

    Também mutará o BiologicalSystem retornado por `system_factory()`
    internamente (chama `.observe()` nele) — não passe um sistema já em
    uso por outro lugar; deixe a fábrica criar um novo a cada chamada.
    """
    main_system = system_factory()
    cohort = Cohort()

    for obs in observations:  # ordem DADA, propositalmente não ordenada aqui
        main_system.observe(obs)
        cohort.update(main_system, representation, timestamp=obs.timestamp)

    trajectory = cohort.trajectories.get(main_system.id)
    n_points = len(trajectory) if trajectory is not None else 0

    n_distinct_systems = len(cohort.trajectories)

    timestamps = [trajectory.at(i).timestamp for i in range(n_points)] if trajectory is not None else []
    is_sorted = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
    expected_timestamps = sorted(o.timestamp for o in observations)
    timestamps_match = timestamps == expected_timestamps

    order = representation.domain_names()
    no_lookahead = True
    if trajectory is not None:
        ordered_observations = sorted(observations, key=lambda o: o.timestamp)
        for i in range(n_points):
            snapshot_system = system_factory()
            for obs in ordered_observations[: i + 1]:
                snapshot_system.observe(obs)
            expected_vector = representation.transform(snapshot_system)
            actual_vector = trajectory.at(i)
            if not np.allclose(expected_vector.as_vector(order), actual_vector.as_vector(order)):
                no_lookahead = False
                break

    return TemporalityReport(
        n_observations=len(observations),
        n_trajectory_points=n_points,
        n_distinct_systems=n_distinct_systems,
        is_chronologically_sorted=is_sorted,
        timestamps_match_observations=timestamps_match,
        no_lookahead=no_lookahead,
    )


# =============================================================================
# Composicionalidade (Seção 5.3 da teoria)
# =============================================================================
@dataclass
class CompositionalityReport:
    """
    Compara o vetor de UM `CompositeRepresentation` contra a concatenação
    independente de cada domínio FOLHA (recursivamente, via
    `leaf_domains()`) — compor não pode perder, duplicar, nem distorcer
    nenhuma Feature em relação a transformar cada folha separadamente.
    """

    composite_name: str
    n_features_composite: int
    n_features_leaves_sum: int
    all_leaf_values_present: bool

    @property
    def is_compositional(self) -> bool:
        return self.n_features_composite == self.n_features_leaves_sum and self.all_leaf_values_present

    def summary(self) -> str:
        status = "OK" if self.is_compositional else "FALHOU"
        return (
            f"Composicionalidade de '{self.composite_name}': {status} "
            f"(composite={self.n_features_composite} Features, soma das folhas={self.n_features_leaves_sum})"
        )


def check_compositionality(composite: CompositeRepresentation, system: BiologicalSystem) -> CompositionalityReport:
    """
    Contrato de Composicionalidade (5.3): R(A ∪ B) deve corresponder
    exatamente a R(A) ∪ R(B) — a Representation de um grupo é a
    concatenação sem perdas das Representations de suas partes,
    independentemente de quantos níveis de aninhamento existam entre o
    grupo e cada folha (`CompositeRepresentation` permite grupos dentro
    de grupos — ver `composite_representation.py`).

    Verifica CONTAGEM (nenhuma Feature perdida ou duplicada ao compor) e
    o CONJUNTO DE VALORES (nenhuma Feature distorcida no processo de
    prefixar nomes / achatar hierarquia) — não apenas "não deu erro".
    """
    composite_features = composite.transform(system)
    composite_values = sorted(f.value for f in composite_features)

    leaf_values: list[float] = []
    for leaf in composite.leaf_domains():
        leaf_values.extend(f.value for f in leaf.transform(system))
    leaf_values_sorted = sorted(leaf_values)

    return CompositionalityReport(
        composite_name=composite.name,
        n_features_composite=len(composite_features),
        n_features_leaves_sum=len(leaf_values),
        all_leaf_values_present=(composite_values == leaf_values_sorted),
    )


# =============================================================================
# Independência Algorítmica (Seção 5.6 da teoria)
# =============================================================================
@dataclass
class AlgorithmicIndependenceReport:
    """
    Testa empiricamente se um algoritmo (Geometry, Operator, Phenotyper,
    ...) funciona sobre `RepresentationSpace`s de ORIGENS DIFERENTES
    (dimensões diferentes, domínios diferentes — ex.: sleep vs diabetes)
    sem nenhuma modificação — a forma testável de "o algoritmo não
    depende de conhecimento específico de doença".
    """

    algorithm_name: str
    ran_successfully_on_a: bool
    ran_successfully_on_b: bool
    error_a: Optional[str] = None
    error_b: Optional[str] = None

    @property
    def is_independent(self) -> bool:
        return self.ran_successfully_on_a and self.ran_successfully_on_b

    def summary(self) -> str:
        status = "OK" if self.is_independent else "FALHOU"
        lines = [f"Independência Algorítmica de '{self.algorithm_name}': {status}"]
        if not self.ran_successfully_on_a:
            lines.append(f"  Falhou no espaço A: {self.error_a}")
        if not self.ran_successfully_on_b:
            lines.append(f"  Falhou no espaço B: {self.error_b}")
        return "\n".join(lines)


def check_algorithmic_independence(
    algorithm_factory: Callable[[], Any],
    space_a: Any,
    space_b: Any,
    apply_fn: Callable[[Any, Any], Any],
    algorithm_name: Optional[str] = None,
) -> AlgorithmicIndependenceReport:
    """
    `algorithm_factory()`: cria uma instância NOVA do algoritmo (nunca
    reaproveitada entre os dois testes — evita que estado interno de um
    ajuste vaze para o outro). `apply_fn(algoritmo, space)`: roda o
    algoritmo sobre um `RepresentationSpace` (assinatura varia por tipo
    de algoritmo — Geometry precisa de 2 vetores, Phenotyper precisa de
    `.fit(space)`, então quem chama decide como aplicar). `space_a` e
    `space_b` devem vir de PLUGINS/DOMÍNIOS DIFERENTES (dimensões e
    nomes de Feature diferentes) — o teste só é significativo se as
    duas origens forem genuinamente distintas.
    """
    nome = algorithm_name or algorithm_factory().__class__.__name__

    try:
        apply_fn(algorithm_factory(), space_a)
        ok_a, err_a = True, None
    except Exception as e:  # noqa: BLE001 — deliberado: qualquer exceção conta como falha do contrato, não deve propagar
        ok_a, err_a = False, f"{type(e).__name__}: {e}"

    try:
        apply_fn(algorithm_factory(), space_b)
        ok_b, err_b = True, None
    except Exception as e:  # noqa: BLE001
        ok_b, err_b = False, f"{type(e).__name__}: {e}"

    return AlgorithmicIndependenceReport(
        algorithm_name=nome, ran_successfully_on_a=ok_a, ran_successfully_on_b=ok_b, error_a=err_a, error_b=err_b
    )


# =============================================================================
# Interoperabilidade (Seção 5.10 da teoria)
# =============================================================================
@dataclass
class InteroperabilityReport:
    """
    Serializa um `RepresentationVector` para um formato de intercâmbio
    padrão (dict compatível com JSON — não um formato proprietário do
    BioSpace) e reconstrói, conferindo fidelidade sem perda — a forma
    testável de "os dados podem sair do sistema e voltar sem distorção",
    em vez de uma alegação vaga de "é interoperável".
    """

    n_features: int
    roundtrip_successful: bool
    max_absolute_error: float

    @property
    def is_interoperable(self) -> bool:
        return self.roundtrip_successful and self.max_absolute_error < 1e-9

    def summary(self) -> str:
        status = "OK" if self.is_interoperable else "FALHOU"
        return f"Interoperabilidade: {status} ({self.n_features} Features, erro máximo={self.max_absolute_error:.2e})"


def check_interoperability(vector) -> InteroperabilityReport:  # vector: RepresentationVector (import evitado para não criar ciclo)
    """
    Contrato de Interoperabilidade (5.10): serializa `vector` para JSON
    (via `json.dumps`/`json.loads` — o formato de intercâmbio mais
    universal disponível, não um pickle específico do Python) e
    reconstrói, comparando valor a valor. `RepresentationVector` em si
    NÃO precisa saber serializar-se (não é responsabilidade do núcleo);
    este contrato testa que a INFORMAÇÃO que ele carrega (Features com
    nome/valor/domínio) sobrevive a uma volta completa por um formato
    universal, o requisito real por trás de "interoperabilidade".
    """
    import json

    serializable = {
        "system_id": vector.system_id,
        "timestamp": vector.timestamp.isoformat(),
        "components": {
            domain_name: [{"name": f.name, "value": f.value} for f in features]
            for domain_name, features in vector.components.items()
        },
    }

    try:
        json_str = json.dumps(serializable)
        reconstructed = json.loads(json_str)
    except (TypeError, ValueError):
        return InteroperabilityReport(n_features=0, roundtrip_successful=False, max_absolute_error=float("inf"))

    errors: list[float] = []
    for domain_name, features in vector.components.items():
        recon_features = {f["name"]: f["value"] for f in reconstructed["components"][domain_name]}
        for f in features:
            errors.append(abs(f.value - recon_features[f.name]))

    n_features = sum(len(features) for features in vector.components.values())
    return InteroperabilityReport(
        n_features=n_features,
        roundtrip_successful=True,
        max_absolute_error=max(errors) if errors else 0.0,
    )


# =============================================================================
# Independência de Atualização entre Domínios (proposta em revisão externa,
# conectada ao bug real do parâmetro `as_of` já documentado no README)
# =============================================================================
@dataclass
class DomainIndependenceReport:
    """
    Resultado de `check_domain_update_independence` para UM domínio:
    domínios que não compartilham nenhuma variável com uma nova
    observação não deveriam ter nenhuma de suas Features alteradas por
    ela — R(t) -> Observation -> R(t+1) deve atualizar só o que a nova
    observação realmente toca.
    """

    domain_name: str
    changed_feature_names: list[str] = field(default_factory=list)

    @property
    def is_independent(self) -> bool:
        return len(self.changed_feature_names) == 0

    def summary(self) -> str:
        status = "OK (inalterado)" if self.is_independent else f"MUDOU: {self.changed_feature_names}"
        return f"Domínio '{self.domain_name}': {status}"


def check_domain_update_independence(
    system: BiologicalSystem,
    representation: Representation,
    unrelated_observation: Observation,
    protected_domain_names: Sequence[str],
) -> dict[str, DomainIndependenceReport]:
    """
    Contrato: domínios cujas variáveis não são tocadas por
    `unrelated_observation` não devem ter NENHUMA de suas Features
    alteradas depois que ela é adicionada ao sistema. Computa a
    representação ANTES, adiciona a observação (efeito colateral real em
    `system` — mesmo padrão de `check_temporality`), computa DEPOIS, e
    compara Features de cada domínio em `protected_domain_names` — devem
    ser bit-idênticas.

    Esta é uma checagem PASSIVA: pelo desenho atual de
    `Observable.extract()`/`BiologicalSystem.latest_measurement()`, o
    sistema já não deveria mudar Features de domínios não relacionados —
    o valor deste contrato é detectar uma REGRESSÃO futura que quebre
    essa garantia (ex.: um cache mal invalidado, uma normalização que
    passasse a depender de estatísticas globais recalculadas a cada
    observação), não corrigir um bug conhecido hoje.
    """
    vector_antes = representation.transform(system)
    system.observe(unrelated_observation)
    vector_depois = representation.transform(system)

    resultado: dict[str, DomainIndependenceReport] = {}
    for domain_name in protected_domain_names:
        features_antes = {f.name: f.value for f in vector_antes.components.get(domain_name, [])}
        features_depois = {f.name: f.value for f in vector_depois.components.get(domain_name, [])}
        mudou = [
            nome
            for nome in features_antes
            if nome not in features_depois or not np.isclose(features_antes[nome], features_depois[nome], equal_nan=True)
        ]
        resultado[domain_name] = DomainIndependenceReport(domain_name=domain_name, changed_feature_names=mudou)
    return resultado
