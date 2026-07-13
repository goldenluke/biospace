"""
tests.test_topology
=======================

`biospace.topology` (Mapper via kmapper, homologia persistente via
ripser) — validado contra um círculo sintético com topologia
CONHECIDA (1 componente conexa, exatamente 1 "buraco") antes de
qualquer aplicação real.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.topology import compute_mapper_graph, compute_persistence


def _vetor(system_id: str, x: float, y: float) -> RepresentationVector:
    comps = {"d": [Feature(name="x", value=x, raw_value=x), Feature(name="y", value=y, raw_value=y)]}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def _space_circulo(n=200, seed=0, ruido=0.05):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    space = RepresentationSpace(domain_order=["d"])
    for i in range(n):
        x = np.cos(theta[i]) + rng.normal(0, ruido)
        y = np.sin(theta[i]) + rng.normal(0, ruido)
        space.add(_vetor(f"p{i}", x, y))
    return space


def _space_dois_blobs(seed=1):
    """2 clusters bem separados -- topologia conhecida: H_0 deveria achar 2 componentes significativas, H_1 deveria ser vazio (sem buraco)."""
    rng = np.random.default_rng(seed)
    space = RepresentationSpace(domain_order=["d"])
    for i in range(60):
        x, y = rng.normal(0, 0.2, 2)
        space.add(_vetor(f"a{i}", x, y))
    for i in range(60):
        x, y = rng.normal(10, 0.2, 2)
        space.add(_vetor(f"b{i}", x, y))
    return space


def test_persistence_correctly_detects_the_single_hole_in_a_circle():
    """TESTE DECISIVO: um circulo tem exatamente 1 "buraco" topologico por construcao -- H_1 deveria mostrar 1 persistencia muito maior que as demais (ruido)."""
    space = _space_circulo()
    resultado = compute_persistence(space, order=["d"])

    persistencias_h1 = resultado.persistences(1)
    assert len(persistencias_h1) > 0
    assert persistencias_h1[0] > 5 * persistencias_h1[1], (
        f"Esperava a maior persistencia H1 dominar as demais por uma margem grande -- obteve {persistencias_h1[:3]}"
    )
    assert resultado.betti_number(1, min_persistence=0.3) == 1


def test_persistence_finds_two_significant_components_in_two_separated_blobs():
    """2 clusters bem separados: H_0 deveria mostrar exatamente 1 persistencia grande (a fusao final dos 2 blobs, tardia no filtro) e nenhum buraco significativo em H_1."""
    space = _space_dois_blobs()
    resultado = compute_persistence(space, order=["d"])

    persistencias_h0 = resultado.persistences(0)
    assert persistencias_h0[0] > 5 * persistencias_h0[1], "Esperava 1 persistencia H0 dominante (a fusao tardia dos 2 blobs bem separados)."
    assert resultado.betti_number(1, min_persistence=0.5) == 0, "Esperava nenhum buraco significativo em 2 blobs sem estrutura ciclica."


def test_mapper_graph_forms_a_cycle_on_circle_data_with_angle_sensitive_lens():
    """
    TESTE DECISIVO: com uma lente que varia ao redor do circulo (a
    coordenada x, nao a norma L2 que fica quase constante num circulo
    centrado), o grafo de Mapper deveria formar um CICLO -- numero de
    arestas >= numero de nos (uma arvore teria n-1 arestas; um ciclo
    tem exatamente n arestas para n nos).

    ACHADO REAL no caminho: com perc_overlap=0.3 (valor testado
    inicialmente), o grafo formou uma ARVORE quase-ciclo (16 arestas,
    17 nos -- faltando exatamente 1 aresta pra fechar o loop), nao um
    ciclo genuino. Investigado variando perc_overlap: 0.4-0.5 fecham o
    ciclo perfeitamente (nos=arestas); 0.6 sobre-conecta (arestas >>
    nos, uniao espuria de regioes nao adjacentes). O modulo usa 0.4
    como padrao, nao o 0.3 originalmente testado, por causa deste
    achado -- registrado aqui, nao escondido atras de um valor padrao
    ajustado silenciosamente.
    """
    space = _space_circulo(seed=2)
    grafo = compute_mapper_graph(space, order=["d"], lens=lambda m: m[:, 0], lens_description="coordenada x")

    assert grafo.n_nodes() > 3
    assert grafo.n_edges() >= grafo.n_nodes(), (
        f"Esperava pelo menos tantas arestas quanto nos (estrutura ciclica, consistente com H_1=1) -- "
        f"obteve nos={grafo.n_nodes()}, arestas={grafo.n_edges()}"
    )


def test_mapper_default_lens_runs_without_error_on_real_shaped_data():
    """A lente padrao (norma L2) deveria rodar sem erro mesmo quando nao e' a lente ideal para revelar uma estrutura especifica -- confirma que o valor padrao e' pelo menos um fallback razoavel, nao quebrado."""
    space = _space_dois_blobs(seed=3)
    grafo = compute_mapper_graph(space, order=["d"])
    assert grafo.n_nodes() > 0
    assert "L2" in grafo.lens_description or "l2" in grafo.lens_description.lower()


def test_persistence_result_summary_mentions_the_threshold_used():
    space = _space_circulo(seed=4, ruido=0.02)
    resultado = compute_persistence(space, order=["d"])
    texto = resultado.summary(min_persistence=0.5)
    assert "0.5" in texto


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/P_DEMO.xpt"),
    reason="Requer os arquivos reais do NHANES.",
)
def test_no_significant_topological_loop_found_in_real_nhanes_sample():
    """
    ACHADO REAL, negativo: diferente do circulo sintetico (onde H_1
    mostra 1 persistencia dominante, >5x maior que a segunda), uma
    amostra real do NHANES (600 pacientes) nao mostra NENHUM buraco
    topologico significativo -- a maior persistencia H_1 fica bem
    abaixo de 1.0, com decaimento suave entre as top-10, sem o "salto"
    caracteristico de estrutura ciclica genuina. Interpretacao honesta:
    a populacao metabolica do NHANES, nesta representacao, parece mais
    proxima de uma "nuvem" continua do que de uma estrutura com
    ciclos genuinos -- consistente com (nao prova de) a populacao ser
    mais um continuo de gravidade metabolica do que subgrupos
    discretos com estrutura de retorno (ciclo).
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.nhanes import NHANES_PREPANDEMIC_FILES, load_nhanes_metabolic_cohort
    from biospace.plugins.metabolic import load_from_dataframe

    df = load_nhanes_metabolic_cohort("/mnt/user-data/uploads", files=NHANES_PREPANDEMIC_FILES)
    df_adultos = df[df["idade"] >= 20].copy()
    cohort, representation = load_from_dataframe(df_adultos)
    order = representation.domain_names()
    space_completo = cohort.snapshot()

    import random

    random.seed(0)
    ids_amostra = random.sample(space_completo.ids(), 600)
    space_amostra = RepresentationSpace(domain_order=order)
    for sid in ids_amostra:
        space_amostra.add(space_completo.get(sid))

    resultado = compute_persistence(space_amostra, order=order, max_dimension=1)
    persistencias_h1 = resultado.persistences(1)
    assert len(persistencias_h1) > 0, "Esperava pelo menos alguns ciclos de ruido em H_1 (mesmo que nenhum significativo)."
    assert persistencias_h1[0] < 1.0, f"Esperava a maior persistencia H1 bem abaixo de 1.0 (achado documentado: ausencia de ciclo genuino) -- obteve {persistencias_h1[0]:.3f}"
    assert resultado.betti_number(1, min_persistence=1.0) == 0
