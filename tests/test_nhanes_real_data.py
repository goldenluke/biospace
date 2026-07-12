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
ARQUIVOS_NHANES = {
    "demo": "P_DEMO.xpt", "ghb": "P_GHB.xpt", "glu": "P_GLU.xpt", "bmx": "P_BMX.xpt", "bpxo": "P_BPXO.xpt", "diq": "P_DIQ.xpt",
    "biopro": "P_BIOPRO.xpt", "tchol": "P_TCHOL.xpt", "hdl": "P_HDL.xpt", "triglycerides": "P_TRIGLY.xpt",
}

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
    ACHADO REAL, atualizado quando lipídios/renal/sexo foram
    adicionados à representação (20 dimensões, era 16): a variância de
    K=3 COM idade aumentou substancialmente (desvio 0,018->0,270 em 20
    seeds) -- a representação mais rica tornou a partição em 3 grupos
    menos rígida, não mais quase-determinística. Média ainda alta
    (~0,79) mas não mais >0,85 com confiança. Isto é, em si, uma
    instância não planejada da tese central do projeto
    ("Representation Before Inference"): a mesma pergunta (K=3 é
    estável?) muda de resposta quando a representação muda, agora
    dentro da MESMA fonte de dados, não apenas entre fontes diferentes.

    K=2 permanece robusto de qualquer forma -- ver asserção abaixo.
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

    assert np.mean(aris_com) > 0.65, f"Esperava K=3 COM idade ainda razoavelmente estavel em media -- media={np.mean(aris_com):.3f}"
    assert np.std(aris_com) > 0.15, f"Esperava variancia bem maior que antes da representacao mais rica (achado documentado: desvio subiu de 0.018) -- desvio={np.std(aris_com):.3f}"

    # K=2 sem idade permanece robusto -- checagem preservada da versao anterior deste teste
    aris_k2_sem_idade = [stability_ari(matriz_sem_idade, 2, s) for s in seeds[:10]]
    assert all(a > 0.7 for a in aris_k2_sem_idade), (
        f"Esperava K=2 continuar estavel SEM idade em TODAS as 10 seeds testadas -- obteve {aris_k2_sem_idade}"
    )


def test_structural_curvature_discriminates_phenotype_after_lipid_renal_added(real_nhanes_adults):
    """
    ACHADO REAL, REVERTIDO quando lipídios/renal/sexo foram adicionados
    à representação: com a representação mais pobre (16 dimensões), a
    curvatura NÃO discriminava fenótipo (p=0,36, achado documentado
    anteriormente, contrastando com SAOS). Com a representação mais
    rica (20 dimensões, este teste), a curvatura PASSA a discriminar
    significativamente (p<0,01, confirmado em 3 seeds independentes:
    1,18e-03 / 3,57e-03 / 1,56e-04). Instância não planejada, dentro da
    MESMA fonte de dados, da tese central do projeto: representação
    determina se estrutura fica detectável -- adicionar dimensões
    genuinamente novas (não redundantes com o que já existia) pode
    revelar tensão estrutural na fronteira que uma representação mais
    pobre não conseguia ver.
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
    assert p < 0.05, f"Esperava SIGNIFICATIVO com a representacao enriquecida (achado documentado, revertido do anterior) -- obteve p={p:.2e}"


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


def test_raw_value_correctly_returns_none_for_missing_feature_not_zero(real_nhanes_adults):
    """
    O TESTE DECISIVO para um bug real: `_raw_value()` (usada por
    classify_diabetes_status e classify_metabolic_syndrome_risk_full)
    tinha um bug de fallback -- para uma Feature AUSENTE, `f.value` vale
    0.0 (nao None), e a versao anterior de `_raw_value` caia nesse 0.0
    silenciosamente em vez de devolver None. Isso tornava o ramo
    "indeterminado" de classify_diabetes_status inalcancavel: pacientes
    sem HbA1c NEM glicemia eram classificados como "normal" (0.0 nunca
    bate >=6.5 nem >=126), nao "indeterminado". CORRIGIDO E VERIFICADO
    aqui contra dado real: deve existir pelo menos um paciente
    verdadeiramente "indeterminado" na coorte adulta completa.
    """
    from biospace.plugins.metabolic import classify_diabetes_status, load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()

    status_encontrados = {classify_diabetes_status(space.get(sid)) for sid in cohort.trajectories}
    assert "indeterminado" in status_encontrados, (
        "Esperava pelo menos um paciente 'indeterminado' (sem HbA1c nem glicemia) -- "
        "se isto falhar, o bug de _raw_value pode ter voltado (fallback silencioso para 0.0)."
    )


def test_diabetes_sensitivity_corrected_after_raw_value_bugfix(real_nhanes_adults):
    """
    ACHADO REAL: o bug de _raw_value corrigido acima mudou um numero JA
    PUBLICADO no artigo de diabetes -- sensibilidade era reportada como
    66.6% (calculada com o bug, que classificava incorretamente
    pacientes sem NENHUM dado laboratorial como "normal" em vez de
    excluir como "indeterminado", inflando artificialmente o denominador
    com casos nao informativos). Corrigido: sensibilidade real e' ~75%,
    nao 66.6% -- a conclusao qualitativa (subdiagnostico de diabetes
    existe) permanece valida, mas o numero exato mudou e o artigo foi
    corrigido para refletir isso.
    """
    from biospace.plugins.metabolic import classify_diabetes_status, load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)

    tp = fn = fp = tn = n_indeterminado = 0
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
            n_indeterminado += 1
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

    assert n_indeterminado > 500, f"Esperava >500 pacientes indeterminados agora corretamente excluidos -- obteve {n_indeterminado}"
    sensibilidade = tp / (tp + fn)
    assert 0.70 < sensibilidade < 0.80, f"Esperava sensibilidade corrigida ~75% (achado documentado, nao mais 66.6%) -- obteve {sensibilidade:.3f}"


def test_lipid_and_renal_domains_have_real_completeness_patterns(real_nhanes_adults):
    """
    ACHADO REAL: creatinina/eGFR (~61% completo) e colesterol/HDL (~70%)
    tem completude muito maior que triglicerideos (~30-42%, subamostra
    em jejum) -- os 3 padroes de completude diferentes devem aparecer
    corretamente na representacao, nao um erro de carregamento.
    """
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()
    todos_ids = list(space.ids())  # populacao adulta INTEIRA, nao uma fatia -- fatias arbitrarias ja se mostraram nao representativas neste projeto

    completude = {"creatinina_mg_dl": 0, "colesterol_total_mg_dl": 0, "trigliceridios_mg_dl": 0}
    for sid in todos_ids:
        vetor = space.get(sid)
        for f in vetor.components["renal"]:
            if f.name == "creatinina_mg_dl" and not f.is_missing:
                completude["creatinina_mg_dl"] += 1
        for f in vetor.components["lipid"]:
            if f.name in completude and not f.is_missing:
                completude[f.name] += 1

    frac_creat = completude["creatinina_mg_dl"] / len(todos_ids)
    frac_chol = completude["colesterol_total_mg_dl"] / len(todos_ids)
    frac_trig = completude["trigliceridios_mg_dl"] / len(todos_ids)

    assert frac_trig < frac_chol, "Trigliceridios deveria ser bem menos completo que colesterol total (subamostra em jejum)."
    assert 0.75 < frac_creat < 0.95, f"Completude de creatinina em ADULTOS e' mais alta que na populacao completa (achado real: ~85%) -- obteve {frac_creat:.3f}"
    assert 0.5 < frac_chol < 0.9
    assert 0.15 < frac_trig < 0.55


def test_full_metabolic_syndrome_criterion_is_sex_specific(real_nhanes_adults):
    """
    O TESTE DECISIVO para classify_metabolic_syndrome_risk_full: os
    limiares de cintura e HDL sao sexo-especificos (102cm/40mg-dL
    masculino vs 88cm/50mg-dL feminino) -- constroi dois pacientes com
    o MESMO valor de cintura (95cm, entre os dois limiares) e confirma
    que o criterio discorda entre sexos, provando que o limiar sexo-
    especifico esta realmente sendo aplicado, nao apenas um unico
    limiar fixo disfarcado.
    """
    from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, classify_metabolic_syndrome_risk_full, exam
    from datetime import datetime

    representation = MetabolicRepresentation()

    def paciente(sexo):
        system = MetabolicSystem()
        system.observe(exam({
            "idade": 50, "sexo": sexo, "circunferencia_abdominal_cm": 95.0, "imc": 27.0,
            "pressao_sistolica_mmhg": 115.0, "pressao_diastolica_mmhg": 75.0,
            "glicemia_jejum_mg_dl": 90.0, "hdl_mg_dl": 55.0, "trigliceridios_mg_dl": 100.0, "colesterol_total_mg_dl": 180.0,
        }, timestamp=datetime(2024, 1, 1)))
        return representation.transform(system)

    vetor_masculino = paciente(1.0)  # 95cm > limiar masculino (102? nao, 95<102 -- ajustar teste)
    vetor_feminino = paciente(2.0)  # 95cm > limiar feminino (88) -- deveria contar

    r_masc = classify_metabolic_syndrome_risk_full(vetor_masculino)
    r_fem = classify_metabolic_syndrome_risk_full(vetor_feminino)

    assert r_masc["criterios"]["adiposidade_central"] is False, "95cm < 102cm (limiar masculino) -- nao deveria contar."
    assert r_fem["criterios"]["adiposidade_central"] is True, "95cm >= 88cm (limiar feminino) -- deveria contar."


def test_full_metabolic_syndrome_criteria_are_none_when_unevaluable():
    """Sem sexo, os criterios sexo-especificos (cintura, HDL) devem ficar None -- nunca assumir um sexo default."""
    from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, classify_metabolic_syndrome_risk_full, exam
    from datetime import datetime

    representation = MetabolicRepresentation()
    system = MetabolicSystem()
    system.observe(exam({
        "idade": 50, "circunferencia_abdominal_cm": 95.0, "hdl_mg_dl": 45.0,
        "pressao_sistolica_mmhg": 115.0, "glicemia_jejum_mg_dl": 90.0,
    }, timestamp=datetime(2024, 1, 1)))
    vetor = representation.transform(system)
    resultado = classify_metabolic_syndrome_risk_full(vetor)
    assert resultado["criterios"]["adiposidade_central"] is None
    assert resultado["criterios"]["hdl_baixo"] is None
    assert resultado["sexo_disponivel"] is False


def test_process_coherence_runs_with_five_simultaneous_processes_on_real_data(real_nhanes_adults):
    """
    ACHADO REAL: com os domínios novos, a representação declara 5
    processos simultâneos (antes só 1 -- glucose_homeostasis). Testa que
    check_process_coherence roda corretamente com essa complexidade
    maior, produzindo pares tanto do mesmo processo quanto de processos
    diferentes para cada um dos 5. Não asserta is_coherent=True global
    aqui deliberadamente: lipid_metabolism tem alta ausência
    (trigliceridios ~30%), reduzindo pares completos disponíveis o
    bastante para não ser um teste justo de coerência isolada -- ver
    `test_process_coherence_confirmed_on_real_population_unlike_synthetic_data`
    para o teste de coerência em si (glucose_homeostasis).
    """
    from biospace.core import check_process_coherence
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()

    assert representation.processes() == {
        "glucose_homeostasis", "body_composition", "cardiovascular_regulation", "renal_filtration", "lipid_metabolism"
    }

    relatorio = check_process_coherence(representation, space)
    assert len(relatorio.same_process_pairs) > 0, "Esperava pelo menos um par do mesmo processo entre os 5 processos declarados."
    assert len(relatorio.different_process_pairs) > 0, "Esperava pelo menos um par de processos diferentes."


def test_renal_filtration_coherence_is_moderate_not_perfectly_tautological(real_nhanes_adults):
    """
    ACHADO REAL, CORRIGIDO de uma suposição inicial errada: eu assumi
    que creatinina/eGFR correlacionariam quase perfeitamente (>0.7) por
    eGFR ser CALCULADO a partir da creatinina -- errado. A correlação
    linear real e' moderada (|r|~0.58, nao >0.7), porque a fórmula
    CKD-EPI 2021 é NÃO LINEAR em creatinina (dois regimes com expoentes
    fracionários diferentes, min/max) e tem dependência MULTIPLICATIVA
    independente de idade (0.9938^idade) -- suficiente para reduzir a
    correlação linear bem abaixo do que uma relação "quase tautológica"
    sugeriria. Registrado como correção da minha própria suposição, não
    como um número apenas ajustado para o teste passar.
    """
    from biospace.core import check_process_coherence
    from biospace.plugins.metabolic import load_from_dataframe

    cohort, representation = load_from_dataframe(real_nhanes_adults)
    space = cohort.snapshot()
    relatorio = check_process_coherence(representation, space)

    pares_renal = [p for p in relatorio.same_process_pairs if "renal" in p[0] or "renal" in p[1]]
    assert len(pares_renal) > 0
    assert 0.4 < pares_renal[0][2] < 0.8, (
        f"Esperava |r| moderado (achado real corrigido, nao 'quase tautologico') -- obteve {pares_renal[0][2]:.3f}"
    )
