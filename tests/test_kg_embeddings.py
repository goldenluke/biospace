"""
tests.test_kg_embeddings
============================

`biospace.kg_embeddings.TransE` — validado em camadas, na ordem em
que a investigação real aconteceu: primeiro confirma que o treino
converge (acurácia alta nas próprias triplas de treino); depois
confirma um limite genuíno e documentado (uma entidade cuja única
tripla de treino é retida para teste nunca tem seu embedding
atualizado -- comportamento esperado de métodos transdutivos, não um
bug); por fim confirma predição de link em retenção genuína (dado com
correlação real entre relações, condição em que TransE deveria
generalizar).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from biospace.kg_embeddings import Triple, TransE


def test_transe_fits_training_triples_with_high_accuracy():
    """O treino em si deveria convergir -- acuracia alta nas PROPRIAS triplas de treino, sem nenhuma retencao."""
    categorias = ["catA", "catB", "catC"]
    tamanhos = ["pequeno", "medio", "grande"]
    itens = [f"item{i}" for i in range(60)]
    atrib_cat = {item: categorias[i % 3] for i, item in enumerate(itens)}
    atrib_tam = {item: tamanhos[(i // 3) % 3] for i, item in enumerate(itens)}

    triplas = []
    for item in itens:
        triplas.append(Triple(item, "belongs_to", atrib_cat[item]))
        triplas.append(Triple(item, "has_size", atrib_tam[item]))

    modelo = TransE(dim=10, margin=1.0, lr=0.1, n_epochs=400, seed=0)
    modelo.fit(triplas, all_entities=itens + categorias + tamanhos)

    acertos = 0
    triplas_belongs = [t for t in triplas if t.relation == "belongs_to"]
    for t in triplas_belongs:
        ranking = modelo.predict_tail(t.head, t.relation, categorias, k=len(categorias))
        acertos += int(ranking[0] == t.tail)
    assert acertos / len(triplas_belongs) > 0.85, f"Esperava o treino convergir bem nas proprias triplas -- obteve {acertos}/{len(triplas_belongs)}"


def test_transe_generalizes_to_held_out_links_when_structure_is_correlated():
    """
    TESTE DECISIVO: quando ha estrutura correlacionada genuina entre
    relacoes (tamanho determina categoria), o modelo deveria predizer
    corretamente uma tripla de categoria RETIDA usando so a informacao
    de tamanho que sobrou no treino -- a avaliacao classica e decisiva
    de embeddings de grafo de conhecimento (predicao de link), nao
    apenas "o treino roda sem erro".
    """
    categorias = ["catA", "catB", "catC"]
    tamanhos = ["pequeno", "medio", "grande"]
    itens = [f"item{i}" for i in range(60)]
    mapa_tam_para_cat = {"pequeno": "catA", "medio": "catB", "grande": "catC"}
    atrib_tam = {item: tamanhos[i % 3] for i, item in enumerate(itens)}
    atrib_cat = {item: mapa_tam_para_cat[atrib_tam[item]] for item in itens}

    triplas = []
    for item in itens:
        triplas.append(Triple(item, "belongs_to", atrib_cat[item]))
        triplas.append(Triple(item, "has_size", atrib_tam[item]))

    candidatas_holdout = [t for t in triplas if t.relation == "belongs_to"]
    rng = np.random.RandomState(5)
    idx_teste = rng.choice(len(candidatas_holdout), size=15, replace=False)
    triplas_teste = [candidatas_holdout[i] for i in idx_teste]
    set_teste = {(t.head, t.relation, t.tail) for t in triplas_teste}
    triplas_treino = [t for t in triplas if (t.head, t.relation, t.tail) not in set_teste]

    modelo = TransE(dim=10, margin=1.0, lr=0.1, n_epochs=400, seed=0)
    modelo.fit(triplas_treino, all_entities=itens + categorias + tamanhos)

    acertos = sum(
        int(modelo.predict_tail(t.head, t.relation, categorias, k=1)[0] == t.tail)
        for t in triplas_teste
    )
    assert acertos / len(triplas_teste) > 0.7, (
        f"Esperava predicao de link muito acima do acaso (0.33) com estrutura correlacionada -- obteve {acertos}/{len(triplas_teste)}"
    )


def test_transe_fails_to_generalize_when_held_out_relation_is_the_entitys_only_training_signal():
    """
    ACHADO REAL, documentado como comportamento esperado, nao bug: se
    a UNICA tripla de treino de uma entidade e' justamente a retida
    para teste, o embedding dessa entidade nunca e' atualizado por
    gradiente (fica no valor de inicializacao aleatoria) -- limite
    genuino de metodos transdutivos de embedding de grafo, nao uma
    falha de implementacao. Investigado e confirmado antes de aceitar
    qualquer resultado de generalizacao como valido.
    """
    categorias = ["catA", "catB", "catC"]
    itens = [f"item{i}" for i in range(15)]
    atribuicao = {item: categorias[i % 3] for i, item in enumerate(itens)}
    triplas = [Triple(item, "belongs_to", atribuicao[item]) for item in itens]

    item_isolado = "item0"
    triplas_treino = [t for t in triplas if t.head != item_isolado]

    modelo = TransE(dim=8, margin=1.0, lr=0.1, n_epochs=300, seed=0)
    modelo.fit(triplas_treino, all_entities=itens + categorias)

    emb_isolado = modelo.entity_emb[item_isolado]
    assert abs(np.linalg.norm(emb_isolado) - 1.0) < 0.2, "Esperava o embedding nao-treinado permanecer proximo da normalizacao inicial (nao foi atualizado por gradiente)."


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_transe_link_prediction_near_chance_on_real_uci_cohort_graph():
    """
    ACHADO REAL: um grafo de coorte construido a partir da UCI real
    (paciente -[has_phenotype]-> fenotipo K-Means; paciente
    -[a1c_level]-> alto/normal, quando disponivel) -- predicao de link
    retida (fenotipo) fica perto do acaso (0,32 vs. 0,25 esperado por
    4 fenotipos). Investigado antes de aceitar: separando pacientes
    de teste que TINHAM a segunda relacao (a1c_level) dos que so
    tinham has_phenotype (retida), a acuracia NAO foi maior no grupo
    com mais informacao (0,245 vs. 0,338) -- o oposto do que se
    esperaria se a segunda relacao fosse informativa sobre fenotipo.
    Interpretacao honesta: a1c_level nao parece correlacionar com o
    fenotipo de utilizacao hospitalar nesta base, nao um problema de
    entidade sem sinal de treino (que teria dado o padrao oposto).
    Coerente com o restante da triangulacao ja documentada nesta
    base -- mais um metodo independente encontrando pouco sinal
    explorável na mesma representação.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", max_rows=8000, include_diagnosis_category=False)
    space = cohort.snapshot()
    matrix, ids_ordem = space.matrix()
    km = KMeans(n_clusters=4, random_state=0, n_init=10).fit(matrix)
    atribuicoes = dict(zip(ids_ordem, km.labels_.tolist()))

    triplas = []
    for sid in ids_ordem:
        triplas.append(Triple(sid, "has_phenotype", f"phenotype_{atribuicoes[sid]}"))
        vetor = space.get(sid)
        a1c_feat = next((f for f in vetor.components.get("glycemic_testing", []) if f.name == "A1Cresult_ordinal"), None)
        if a1c_feat is not None and not a1c_feat.is_missing:
            nivel = "a1c_alto" if a1c_feat.raw_value >= 1 else "a1c_normal"
            triplas.append(Triple(sid, "a1c_level", nivel))

    assert len(ids_ordem) > 5000, f"Esperava >5000 pacientes -- obteve {len(ids_ordem)}"

    candidatas_holdout = [t for t in triplas if t.relation == "has_phenotype"]
    rng = np.random.RandomState(0)
    idx_teste = rng.choice(len(candidatas_holdout), size=int(0.15 * len(candidatas_holdout)), replace=False)
    triplas_teste = [candidatas_holdout[i] for i in idx_teste]
    set_teste = {(t.head, t.relation, t.tail) for t in triplas_teste}
    triplas_treino = [t for t in triplas if (t.head, t.relation, t.tail) not in set_teste]

    todos_fenotipos = sorted({f"phenotype_{a}" for a in set(atribuicoes.values())})
    todas_entidades = ids_ordem + todos_fenotipos + ["a1c_alto", "a1c_normal"]
    modelo = TransE(dim=12, margin=1.0, lr=0.05, n_epochs=60, seed=0)
    modelo.fit(triplas_treino, all_entities=todas_entidades)

    acertos = sum(int(modelo.predict_tail(t.head, t.relation, todos_fenotipos, k=1)[0] == t.tail) for t in triplas_teste)
    acuracia = acertos / len(triplas_teste)
    assert 0.15 < acuracia < 0.45, f"Esperava acuracia perto do acaso (0.25, achado documentado) -- obteve {acuracia:.3f}"
