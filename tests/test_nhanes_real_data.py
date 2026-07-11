"""
tests.test_nhanes_real_data
===============================

Testes contra os arquivos NHANES REAIS (ciclo Pre-pandemic ago/2017-mar/2020),
enviados pelo usuário — pytest.mark.skipif quando ausentes (mesmo
padrão de tests/test_stability_robustness.py e
tests/test_phenotype_stability_sweep.py).

MOTIVAÇÃO REAL para o primeiro teste: os testes em test_nhanes_loader.py
mockam `_read_xpt` inteiramente (testam `_merge_nhanes_frames` com
DataFrames já carregados) — nunca exercitam `pandas.read_sas` de
verdade. Foi exatamente aí que um bug real (`format="xpt"` em vez de
`format="xport"`) ficou escondido até os arquivos reais chegarem.
`test_read_xpt_works_against_real_files` fecha essa lacuna.
"""

from __future__ import annotations

import os

import pytest

CAMINHO_UPLOADS = "/mnt/user-data/uploads"
ARQUIVOS_NHANES = {"demo": "P_DEMO.xpt", "ghb": "P_GHB.xpt", "glu": "P_GLU.xpt", "bmx": "P_BMX.xpt", "bpxo": "P_BPXO.xpt", "diq": "P_DIQ.xpt"}

_todos_presentes = all(os.path.exists(os.path.join(CAMINHO_UPLOADS, f)) for f in ARQUIVOS_NHANES.values())
pytestmark = pytest.mark.skipif(not _todos_presentes, reason="Arquivos NHANES reais não disponíveis neste ambiente (esperado em CI).")


@pytest.fixture(scope="module")
def real_nhanes_dataframe():
    from biospace.datasets.nhanes import load_nhanes_metabolic_cohort

    return load_nhanes_metabolic_cohort(CAMINHO_UPLOADS, files=ARQUIVOS_NHANES)


@pytest.fixture(scope="module")
def real_nhanes_adults(real_nhanes_dataframe):
    return real_nhanes_dataframe[real_nhanes_dataframe["idade"] >= 20].copy()


def test_read_xpt_works_against_real_files():
    """
    Fecha a lacuna real: testes com dado fabricado mockam _read_xpt
    inteiramente e nunca teriam pego o bug format='xpt' vs 'xport'.
    Este teste exercita pandas.read_sas de verdade.
    """
    from biospace.datasets.nhanes import _read_xpt

    df = _read_xpt(os.path.join(CAMINHO_UPLOADS, "P_DEMO.xpt"))
    assert "SEQN" in df.columns
    assert "RIDAGEYR" in df.columns
    assert len(df) > 1000


def test_real_nhanes_loads_with_plausible_shape(real_nhanes_dataframe):
    assert len(real_nhanes_dataframe) > 10000
    assert set(real_nhanes_dataframe.columns) >= {
        "paciente", "idade", "hba1c_pct", "glicemia_jejum_mg_dl", "imc",
        "circunferencia_abdominal_cm", "pressao_sistolica_mmhg", "pressao_diastolica_mmhg",
    }


def test_real_nhanes_includes_children_by_design(real_nhanes_dataframe):
    """NHANES amostra toda a populacao, incluindo criancas -- achado real, nao um bug (idade minima proxima de 0)."""
    assert real_nhanes_dataframe["idade"].min() < 1.0


def test_derived_variables_return_none_on_cross_sectional_nhanes_data(real_nhanes_dataframe):
    """NHANES e transversal -- qualquer DerivedVariable que precise de >=2 pontos deve devolver None, nunca inventar uma trajetoria que nao existe."""
    from biospace.plugins.metabolic import HbA1cSlopeVariable, load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_dataframe.head(50))
    sid = next(iter(cohort.trajectories))
    traj = cohort.trajectories[sid]
    assert len(traj) == 1
    assert HbA1cSlopeVariable().compute(traj) is None


def test_classify_diabetes_status_against_real_self_reported_diagnosis(real_nhanes_adults):
    """
    O TESTE DECISIVO com dado real: classify_diabetes_status (criterio
    ADA sobre HbA1c/glicemia) comparado contra diagnostico
    AUTORREFERIDO (DIQ010) em ~9 mil adultos reais. Resultado
    documentado como regressao -- sensibilidade/especificidade
    plausiveis clinicamente (subdiagnostico de diabetes e
    pre-diabetes e um fenomeno real e bem documentado, nao um erro
    do classificador).
    """
    from biospace.plugins.metabolic import classify_diabetes_status, load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)

    tp = fn = fp = tn = 0
    for sid, traj in cohort.trajectories.items():
        paciente_original = cohort.systems[sid].metadata.get("paciente_original")
        linha = real_nhanes_adults[real_nhanes_adults["paciente"] == paciente_original]
        if linha.empty:
            continue
        autorreferido = linha.iloc[0]["diabetes_autorreferido"]
        if autorreferido not in (1.0, 2.0):
            continue
        status = classify_diabetes_status(traj.latest())
        if status == "indeterminado":
            continue
        predisse_diabetes = status == "diabetes"
        tem_diabetes_real = autorreferido == 1.0
        if predisse_diabetes and tem_diabetes_real:
            tp += 1
        elif not predisse_diabetes and tem_diabetes_real:
            fn += 1
        elif predisse_diabetes and not tem_diabetes_real:
            fp += 1
        else:
            tn += 1

    sensibilidade = tp / (tp + fn)
    especificidade = tn / (tn + fp)

    assert 0.55 < sensibilidade < 0.80, f"Sensibilidade {sensibilidade:.3f} fora da faixa clinicamente plausivel."
    assert especificidade > 0.90, f"Especificidade {especificidade:.3f} abaixo do esperado."


def test_process_coherence_confirmed_on_real_population_unlike_synthetic_data(real_nhanes_adults):
    """
    ACHADO REAL, contrastando com test_metabolic_synthetic_data_does_not_confirm_coherence
    (tests/test_process_coherence.py): em populacao REAL, a coerencia
    de processo SE CONFIRMA -- HbA1c e glicemia (glucose_homeostasis)
    correlacionam genuinamente mais entre si que com pressao arterial/IMC
    (outros processos). O gerador sintetico de diabetes NAO tinha essa
    propriedade (variaveis sorteadas independentemente); dados reais tem.
    """
    from biospace.core import check_process_coherence
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()

    relatorio = check_process_coherence(representation, space)
    assert relatorio.is_coherent is True, (
        "Esperava coerencia de processo confirmada em populacao real -- se isto falhar, "
        "investigar se algo mudou na forma como os dados sao carregados ou normalizados."
    )


def test_phenotype_stability_is_high_unlike_saos(real_nhanes_adults):
    """
    ACHADO REAL: o oposto exato do achado em SAOS (tests/test_phenotype_stability_sweep.py --
    nenhuma configuracao estavel). No NHANES, K-Means com K=2 e' altamente estavel
    (ARI > 0.85 -- bem acima do limiar convencional de 0.7, Hennig 2007).
    """
    from biospace.phenotyping import KMeansPhenotyper
    from biospace.phenotyping.contracts import check_phenotype_stability
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()

    relatorio = check_phenotype_stability(lambda: KMeansPhenotyper(n_clusters=2), space)
    assert relatorio.adjusted_rand_index > 0.85, (
        f"Esperava alta estabilidade em K=2 (achado documentado, oposto de SAOS) -- obteve ARI={relatorio.adjusted_rand_index:.3f}"
    )


def test_k3_phenotype_stability_is_far_less_robust_without_age_unlike_k2(real_nhanes_adults):
    """
    ACHADO REAL, corrigido após verificação com amostra maior de seeds:
    uma primeira exploração com apenas 3 seeds sugeriu que K=3 SEMPRE
    desestabiliza sem idade -- FALSO, uma reexecução independente
    encontrou uma seed (dentre as mesmas 3) dando ARI=0.953 (estável),
    contradizendo a alegação original. Investigado com 20 seeds antes de
    corrigir a alegação: o padrão real é de VARIABILIDADE muito maior,
    não instabilidade universal -- com idade, ARI e' consistentemente
    alto e quase sem variancia (media=0.938, desvio=0.018, nunca abaixo
    de 0.7 em 20 seeds); sem idade, a media cai para ~0.53 com desvio
    ~0.20 (10x maior) e 18 de 20 seeds ficam abaixo do limiar de
    estabilidade -- mas nao 20 de 20. K=2 permanece robustamente
    estavel com OU sem idade (ver teste companheiro).
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()
    order = representation.domain_names()
    ids = space.ids()

    matriz = np.stack([space.get(sid).as_vector(order) for sid in ids])
    nomes_features = []
    for dom in order:
        exemplo = space.get(ids[0])
        for f in exemplo.components.get(dom, []):
            nomes_features.append(f"{dom}.{f.name}")
    idx_idade = nomes_features.index("anthropometric.idade")
    matriz_sem_idade = np.delete(matriz, idx_idade, axis=1)

    def stability_ari(m, k, seed):
        n = m.shape[0]
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(n)
        split = n // 2
        idx1, idx2 = shuffled[:split], shuffled[split:]
        km1 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(m[idx1])
        km2 = KMeans(n_clusters=k, n_init=10, random_state=0).fit(m[idx2])
        return adjusted_rand_score(km1.predict(m), km2.predict(m))

    seeds = list(range(20))
    aris_com = [stability_ari(matriz, 3, s) for s in seeds]
    aris_sem = [stability_ari(matriz_sem_idade, 3, s) for s in seeds]

    assert np.mean(aris_com) > 0.85, f"Esperava K=3 COM idade consistentemente estavel -- media={np.mean(aris_com):.3f}"
    assert np.std(aris_com) < 0.05, f"Esperava baixa variancia COM idade -- desvio={np.std(aris_com):.3f}"

    assert np.mean(aris_sem) < 0.70, f"Esperava media reduzida SEM idade (achado real, nao 'sempre instavel') -- media={np.mean(aris_sem):.3f}"
    assert np.std(aris_sem) > 0.10, f"Esperava variancia MUITO maior SEM idade -- desvio={np.std(aris_sem):.3f}"
    n_instaveis_sem_idade = sum(1 for a in aris_sem if a < 0.7)
    assert n_instaveis_sem_idade >= 12, f"Esperava maioria das seeds instavel SEM idade (nao todas) -- obteve {n_instaveis_sem_idade}/20"

    # K=2, em contraste, permanece robustamente estavel SEM idade (varias seeds, nao so uma)
    aris_k2_sem_idade = [stability_ari(matriz_sem_idade, 2, s) for s in seeds[:10]]
    assert all(a > 0.7 for a in aris_k2_sem_idade), (
        f"Esperava K=2 continuar estavel SEM idade em TODAS as 10 seeds testadas (achado documentado, contraste com K=3) -- obteve {aris_k2_sem_idade}"
    )


def test_structural_curvature_does_not_differ_between_phenotypes_unlike_saos(real_nhanes_adults):
    """
    ACHADO REAL, NEGATIVO: diferente de SAOS (onde arestas entre fenotipos
    tem curvatura significativamente mais negativa, p=5.7e-19), no NHANES
    a curvatura estrutural NAO difere significativamente entre dentro/entre
    fenotipo -- interpretacao: a assinatura de curvatura parece marcar
    fronteiras frageis num continuo mal separado; com fenotipos ja bem
    separados e estaveis (Secao anterior), sobra pouca tensao estrutural
    na fronteira para detectar.
    """
    import random

    from scipy import stats as scipy_stats

    from biospace.core import RepresentationSpace
    from biospace.geometry import Euclidean, graph_curvature_summary
    from biospace.graph import build_cohort_similarity_graph
    from biospace.phenotyping import KMeansPhenotyper
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()
    order = representation.domain_names()

    random.seed(0)
    ids_amostra = random.sample(space.ids(), min(1500, len(space.ids())))
    space_amostra = RepresentationSpace(domain_order=order)
    for sid in ids_amostra:
        space_amostra.add(space.get(sid))

    grafo = build_cohort_similarity_graph(space_amostra, Euclidean(), k=8, order=order)
    resumo = graph_curvature_summary(grafo, weight="weight")

    phenotyper = KMeansPhenotyper(n_clusters=2)
    phenotypes = phenotyper.fit(space_amostra)
    labels = {}
    for sid in ids_amostra:
        vec = space_amostra.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

    dentro, entre = [], []
    for (u, v), k in resumo["edge_curvatures"].items():
        if labels.get(u) and labels.get(v):
            (dentro if labels[u] == labels[v] else entre).append(k)

    assert len(dentro) > 100 and len(entre) > 20, "Amostra insuficiente de arestas para o teste estatistico ter sentido."
    _, p = scipy_stats.mannwhitneyu(dentro, entre, alternative="greater")
    assert p > 0.05, f"Esperava NAO significativo (achado documentado, oposto de SAOS) -- obteve p={p:.2e}"


def test_autoencoder_beats_pca_at_low_dimension_unlike_saos(real_nhanes_adults):
    """
    ACHADO REAL: diferente de SAOS (n=355, PCA venceu em TODA dimensao
    testada), no NHANES (n=9.232) o autoencoder VENCE em dim=2 --
    confirmado robusto em 3 hidden_dim x 3 seeds (9 configuracoes).
    Confirma diretamente a hipotese de tamanho de amostra: com dado
    suficiente, o metodo nao linear encontra uma solucao melhor que a
    otima linear do PCA, mesmo nesta dimensao baixa onde PCA e' mais
    competitivo.
    """
    from biospace.plugins.metabolic import load_from_dataframe
    from biospace.representation_learning import compare_reconstruction_error

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()

    resultado = compare_reconstruction_error(space, embedding_dim=2, hidden_dim=16, random_state=0)
    assert resultado["autoencoder_melhor"] is True, (
        f"Esperava autoencoder vencer em dim=2 (achado documentado, oposto de SAOS) -- "
        f"PCA={resultado['erro_pca_linear']:.4f}, AE={resultado['erro_autoencoder_nao_linear']:.4f}"
    )


def test_gnn_graph_hurts_even_at_low_label_fractions_unlike_saos(real_nhanes_adults):
    """
    ACHADO REAL, oposto do padrao de SAOS: em SAOS, o grafo ajudava
    MUITO com poucos rotulos (+17.8pp em 5%). No NHANES, o grafo
    ATRAPALHA em toda fracao testada de 5% a 50% -- consistente com os
    achados anteriores (fenotipos ja extremamente bem separados: alta
    estabilidade, curvatura nao discrimina fronteira) -- ha pouco
    espaco para a estrutura relacional ajudar quando a fronteira
    baseada em Features ja e' quase perfeita sozinha.
    """
    import random

    import numpy as np

    from biospace.core import Cohort
    from biospace.geometry import Euclidean
    from biospace.gnn import SimpleGCN, prepare_node_classification_data
    from biospace.graph import build_cohort_similarity_graph
    from biospace.phenotyping import KMeansPhenotyper
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    order = representation.domain_names()

    random.seed(0)
    ids_amostra = random.sample(list(cohort.trajectories.keys()), min(1500, len(cohort.trajectories)))
    cohort_amostra = Cohort()
    for sid in ids_amostra:
        cohort_amostra.systems[sid] = cohort.systems[sid]
        cohort_amostra.trajectories[sid] = cohort.trajectories[sid]
    space = cohort_amostra.snapshot()

    phenotyper = KMeansPhenotyper(n_clusters=2)
    phenotypes = phenotyper.fit(space)
    labels = {}
    for sid in space.ids():
        vec = space.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

    grafo = build_cohort_similarity_graph(space, Euclidean(), k=8, order=order)

    def acc_com_e_sem_grafo(fracao, seed=0):
        rng = np.random.default_rng(seed)
        todos_ids = list(labels.keys())
        rng.shuffle(todos_ids)
        n_treino = max(4, int(fracao * len(todos_ids)))
        treino_ids = todos_ids[:n_treino]
        teste_ids = set(todos_ids[n_treino:])
        dados = prepare_node_classification_data(space, grafo, labels, labeled_ids=treino_ids, order=order)
        A, X, y, labeled_mask, node_ids = dados["A"], dados["X"], dados["y"], dados["labeled_mask"], dados["node_ids"]
        test_mask = np.array([nid in teste_ids for nid in node_ids])
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        gcn = SimpleGCN(hidden_dim=16, learning_rate=0.05, random_state=0)
        gcn.fit(A, X_std, y, labeled_mask, epochs=300)
        acc_com = float((gcn.predict(A, X_std)[test_mask] == y[test_mask]).mean())

        A_id = np.eye(A.shape[0])
        gcn2 = SimpleGCN(hidden_dim=16, learning_rate=0.05, random_state=0)
        gcn2.fit(A_id, X_std, y, labeled_mask, epochs=300)
        acc_sem = float((gcn2.predict(A_id, X_std)[test_mask] == y[test_mask]).mean())
        return acc_com, acc_sem

    for fracao in [0.5, 0.1, 0.05]:
        acc_com, acc_sem = acc_com_e_sem_grafo(fracao)
        assert acc_com <= acc_sem + 0.01, (
            f"Esperava grafo NAO ajudar (ou atrapalhar) em fracao={fracao} (achado documentado) -- "
            f"com_grafo={acc_com:.3f}, sem_grafo={acc_sem:.3f}"
        )


def _diabeticos_com_insulina(real_nhanes_adults):
    return real_nhanes_adults[real_nhanes_adults["diabetes_autorreferido"] == 1.0].copy()


def test_insulin_baseline_imbalance_shows_confounding_by_indication(real_nhanes_adults):
    """
    ACHADO REAL, primeira aplicacao do modulo causal (check_baseline_balance)
    em dado real fora de SAOS: entre diabeticos diagnosticados, quem usa
    insulina tem HbA1c de linha de base MUITO mais alto que quem nao usa
    (SMD=+0.619, o maior desequilibrio entre as 15 Features) --
    confundimento por indicacao classico (insulina prescrita para
    diabetes mais dificil de controlar, nao atribuida ao acaso). Isso so
    funciona porque a Feature de tratamento e' excluida do proprio
    baseline (correcao aplicada anteriormente para dado transversal --
    ver docstring de `causal.balance._collect_baseline`).
    """
    from biospace.causal import check_baseline_balance
    from biospace.plugins.metabolic import load_from_dataframe

    df_diabeticos = _diabeticos_com_insulina(real_nhanes_adults)
    cohort, representation = load_from_dataframe(df_diabeticos)
    order = representation.domain_names()

    relatorio = check_baseline_balance(cohort, "treatment", "insulina", order=order)
    assert relatorio.is_balanced is False
    smd_hba1c = relatorio.smd["glycemic.hba1c_pct"]
    assert smd_hba1c > 0.5, f"Esperava desequilibrio grande em HbA1c (achado documentado, confundimento por indicacao) -- obteve SMD={smd_hba1c:.3f}"


def test_propensity_matching_balances_all_features_on_real_cross_sectional_data(real_nhanes_adults):
    """
    ACHADO REAL: o pareamento por propensao (`match_on_propensity`)
    funciona corretamente em dado TRANSVERSAL real (NHANES) -- zera o
    desequilibrio nas 15 Features, incluindo o HbA1c que estava
    severamente desbalanceado antes. Primeira validacao do modulo causal
    completo (balance + propensity) fora de SAOS.
    """
    from biospace.causal import match_on_propensity
    from biospace.plugins.metabolic import load_from_dataframe

    df_diabeticos = _diabeticos_com_insulina(real_nhanes_adults)
    cohort, representation = load_from_dataframe(df_diabeticos)
    order = representation.domain_names()

    resultado = match_on_propensity(cohort, "treatment", "insulina", order=order)
    assert resultado.n_matched > 300, f"Esperava >300 pares (achado documentado: 369 de 413) -- obteve {resultado.n_matched}"
    assert resultado.balance_after.is_balanced is True, "Esperava balanceamento completo apos pareamento (achado documentado)."


def test_matched_effect_correctly_refuses_on_cross_sectional_data(real_nhanes_adults):
    """
    ACHADO REAL, negativo mas importante: `estimate_matched_effect`
    calcula diferenca-em-diferencas (ultimo exame - primeiro exame),
    exigindo >=2 exames por paciente -- NHANES e' transversal (1 exame
    por pessoa). O modulo RECUSA corretamente (ValueError claro) em vez
    de devolver um numero degenerado (zero trivial) que pareceria uma
    estimativa causal valida sem ser. Documenta uma limitacao estrutural
    real: pareamento por propensao funciona em dado transversal,
    estimativa de EFEITO exige longitudinal -- as duas etapas do modulo
    causal tem requisitos de dado diferentes.
    """
    import pytest

    from biospace.causal import estimate_matched_effect, match_on_propensity
    from biospace.plugins.metabolic import load_from_dataframe

    df_diabeticos = _diabeticos_com_insulina(real_nhanes_adults)
    cohort, representation = load_from_dataframe(df_diabeticos)
    order = representation.domain_names()

    resultado = match_on_propensity(cohort, "treatment", "insulina", order=order)
    with pytest.raises(ValueError, match="diferença-em-diferença|diferenca-em-diferenca"):
        estimate_matched_effect(cohort, resultado, order=order)
