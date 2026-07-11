"""
tests.test_process_coherence
================================

check_process_coherence: validação EMPÍRICA (não apenas a alegação) de
que Features rotuladas sob o mesmo PhysiologicalProcess correlacionam
mais entre si, através da população, do que Features de processos
diferentes. Testado em duas condições sintéticas com resultado
CONHECIDO antes de confiar em qualquer dado real: correlação real
dentro do processo (deveria confirmar) e ruído puro independente
(deveria NÃO confirmar, contraprova).

ACHADO REAL ao aplicar no plugin metabolic (dados sintéticos de
diabetes já existentes): a coerência NÃO se confirma
(`test_metabolic_synthetic_data_does_not_confirm_coherence`) — investigado,
não escondido: o gerador sintético (`plugins/diabetes/synthetic.py`)
sorteia cada variável INDEPENDENTEMENTE dentro da classe de
severidade (`patient_level = {key: rng.normal(...) for key in profile}`,
uma chamada por chave, sem nenhum fator latente compartilhado) —
a correlação populacional que existe vem do rótulo de classe e de
efeitos de tratamento, não de um mecanismo fisiológico real ligando
HbA1c e glicemia além disso. É uma limitação real do GERADOR, não um
bug no contrato -- que se comporta corretamente nos dois testes
sintéticos de verdade conhecida abaixo.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from biospace.core import BiologicalSystem, Cohort, Feature, Observable, Observation, Representation, SemanticDomain, check_process_coherence


class _ProcessObservable(Observable):
    def __init__(self, key: str, process: str):
        self.key = key
        self.process = process


class _SimpleDomain(SemanticDomain):
    def __init__(self, name: str, keys_and_processes: list[tuple[str, str]]):
        self.name = name
        self._keys = [k for k, _ in keys_and_processes]
        super().__init__([_ProcessObservable(k, p) for k, p in keys_and_processes])

    def encode(self, measurements):
        return [Feature(name=k, value=float(measurements[k].value), raw_value=float(measurements[k].value)) for k in self._keys if k in measurements]


def _build_space(valores: dict[str, np.ndarray], keys_and_processes: list[tuple[str, str]], n: int):
    domain = _SimpleDomain("d", keys_and_processes)
    representation = Representation([domain])
    cohort = Cohort()
    for i in range(n):
        system = BiologicalSystem(identifier=f"p{i}")
        vals = {k: valores[k][i] for k, _ in keys_and_processes}
        system.observe(Observation(timestamp=datetime(2024, 1, 1), source="t", values=vals))
        cohort.update(system, representation, timestamp=datetime(2024, 1, 1))
    return representation, cohort.snapshot()


def _correlated_scenario(seed=0, n=200):
    """3 processos, 3 features cada -- cada trinca compartilha um fator latente + ruido pequeno (correlacao REAL dentro do processo)."""
    rng = np.random.default_rng(seed)
    keys_and_processes = []
    valores = {}
    for p_idx in range(3):
        latente = rng.normal(0, 1, n)
        for f_idx in range(3):
            k = f"f{p_idx}_{f_idx}"
            valores[k] = latente + rng.normal(0, 0.3, n)
            keys_and_processes.append((k, f"process_{p_idx}"))
    return _build_space(valores, keys_and_processes, n)


def _pure_noise_scenario(seed=1, n=200):
    """Mesma estrutura de rotulos, mas TODAS as features sao ruido puro independente -- sem estrutura real nenhuma."""
    rng = np.random.default_rng(seed)
    keys_and_processes = []
    valores = {}
    for p_idx in range(3):
        for f_idx in range(3):
            k = f"f{p_idx}_{f_idx}"
            valores[k] = rng.normal(0, 1, n)
            keys_and_processes.append((k, f"process_{p_idx}"))
    return _build_space(valores, keys_and_processes, n)


def test_coherence_confirmed_when_same_process_features_share_a_latent_factor():
    """O TESTE DECISIVO: quando features do mesmo processo genuinamente compartilham um mecanismo (fator latente), o contrato deve confirmar coerencia."""
    representation, space = _correlated_scenario()
    relatorio = check_process_coherence(representation, space)

    assert relatorio.is_coherent is True
    assert relatorio.mean_same_process > relatorio.mean_different_process
    assert relatorio.mannwhitney_p < 0.05


def test_coherence_not_confirmed_for_pure_noise():
    """CONTRAPROVA: sem nenhuma estrutura real (ruido puro independente), o contrato NAO deve confirmar coerencia -- prova que o teste discrimina de verdade."""
    representation, space = _pure_noise_scenario()
    relatorio = check_process_coherence(representation, space)

    assert relatorio.is_coherent is False


def test_features_without_process_are_excluded_from_all_pairs():
    """Features sem processo declarado nao devem aparecer em NENHUM par (nem mesmo-processo, nem diferente-processo)."""
    rng = np.random.default_rng(2)
    n = 100
    keys_and_processes = [("com_proc_a", "x"), ("com_proc_b", "x"), ("sem_processo", None)]
    domain = _SimpleDomain("d", [(k, p) for k, p in keys_and_processes if p is not None])
    # sem_processo precisa ser adicionado sem process declarado -- construir manualmente
    obs_sem_processo = _ProcessObservable("sem_processo", None)
    obs_sem_processo.process = None
    domain.observables.append(obs_sem_processo)
    domain._keys.append("sem_processo")

    representation = Representation([domain])
    cohort = Cohort()
    for i in range(n):
        system = BiologicalSystem(identifier=f"p{i}")
        vals = {"com_proc_a": rng.normal(), "com_proc_b": rng.normal(), "sem_processo": rng.normal()}
        system.observe(Observation(timestamp=datetime(2024, 1, 1), source="t", values=vals))
        cohort.update(system, representation, timestamp=datetime(2024, 1, 1))

    space = cohort.snapshot()
    relatorio = check_process_coherence(representation, space)

    todos_nomes = set()
    for a, b, _ in relatorio.same_process_pairs + relatorio.different_process_pairs:
        todos_nomes.add(a)
        todos_nomes.add(b)
    assert "d.sem_processo" not in todos_nomes


def test_metabolic_synthetic_data_does_not_confirm_coherence():
    """
    ACHADO REAL, documentado como regressao: o gerador sintetico de
    diabetes sorteia cada variavel independentemente dentro da classe
    de severidade -- nao ha fator latente compartilhado por processo,
    entao a coerencia nao deveria se confirmar. Se este teste
    eventualmente falhar (coerencia passar a se confirmar), significa
    que o gerador mudou para incluir correlacao intra-processo real --
    atualizar este teste E a documentacao correspondente juntos.
    """
    from biospace.plugins.diabetes import generate_synthetic_dataframe, load_from_dataframe

    df = generate_synthetic_dataframe(n_per_group=100, seed=1)
    cohort, representation = load_from_dataframe(df)
    space = cohort.snapshot()

    relatorio = check_process_coherence(representation, space)
    assert relatorio.is_coherent is False, (
        "Esperava que a coerencia NAO se confirmasse no gerador sintetico atual (achado documentado) -- "
        "se isto passou a ser True, o gerador mudou e a documentacao precisa ser atualizada."
    )
