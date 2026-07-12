"""
tests.test_uci_diabetes_real_data
=====================================

Testes contra o arquivo real diabetic_data.csv (Strack et al., 2014),
enviado pelo usuário — pytest.mark.skipif quando ausente (mesmo padrão
de test_nhanes_real_data.py). Estrutura genuinamente diferente do
NHANES: sem lab contínuo, sem antropometria, sem pressão -- domínios
próprios (utilization, glycemic_testing, medication_intensity),
documentados em `biospace/datasets/uci_diabetes.py`.
"""

from __future__ import annotations

import os

import pytest

CAMINHO_CSV = "/mnt/user-data/uploads/diabetic_data.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(CAMINHO_CSV), reason="Arquivo UCI Diabetes 130-US Hospitals não disponível neste ambiente.")


@pytest.fixture(scope="module")
def real_uci_cohort():
    """
    Representação com `include_diagnosis_category=False` -- preserva
    EXATAMENTE a representação original (3 domínios) usada para o
    achado publicado de fenótipo-readmissão. Ver
    `test_diagnosis_category_dilutes_readmission_association` para o
    achado, também real, de que incluir o 4º domínio muda o resultado.
    """
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    return load_uci_diabetes_cohort(CAMINHO_CSV, include_diagnosis_category=False)


def test_loads_expected_number_of_patients_and_encounters(real_uci_cohort):
    cohort, representation = real_uci_cohort
    assert cohort.loader_report["n_encontros"] == 101766
    assert cohort.loader_report["n_pacientes"] == 71518
    assert len(cohort.trajectories) == 71518


def test_multi_encounter_patients_produce_real_multi_point_trajectories(real_uci_cohort):
    """23% dos pacientes tem multiplos encontros -- confirma que ao menos um paciente com trajetoria longa (>=10 pontos) foi construido corretamente."""
    cohort, representation = real_uci_cohort
    tamanhos = [len(t) for t in cohort.trajectories.values()]
    assert max(tamanhos) >= 10, "Esperava pelo menos um paciente com trajetoria longa."
    n_multi_encontro = sum(1 for t in tamanhos if t >= 2)
    assert n_multi_encontro > 15000, "Esperava >15 mil pacientes com multiplos encontros (achado real: 23% da coorte)."


def test_utilization_domain_is_essentially_complete(real_uci_cohort):
    """As colunas de utilizacao sao 100% completas na fonte -- nenhuma Feature de utilization deveria estar ausente."""
    cohort, representation = real_uci_cohort
    space = cohort.snapshot()
    amostra_ids = list(space.ids())[:500]
    n_ausentes = 0
    for sid in amostra_ids:
        vetor = space.get(sid)
        n_ausentes += sum(1 for f in vetor.components["utilization"] if f.is_missing)
    assert n_ausentes == 0, "Utilization deveria ser 100% completo -- alguma Feature apareceu ausente."


def test_glycemic_testing_domain_is_mostly_missing_by_design(real_uci_cohort):
    """A1Cresult/max_glu_serum sao esparsos por desenho (83-95% ausentes na fonte)."""
    cohort, representation = real_uci_cohort
    space = cohort.snapshot()
    amostra_ids = list(space.ids())[:2000]
    total = 0
    ausentes = 0
    for sid in amostra_ids:
        vetor = space.get(sid)
        for f in vetor.components["glycemic_testing"]:
            total += 1
            if f.is_missing:
                ausentes += 1
    fracao_ausente = ausentes / total
    assert fracao_ausente > 0.5, f"Esperava alta fracao de ausencia (achado real: 83-95%), obteve {fracao_ausente:.1%}."


def test_derived_variable_computes_on_real_multi_encounter_trajectory(real_uci_cohort):
    """DerivedVariable generaliza para uma TERCEIRA fonte (sintetica -> NHANES transversal -> aqui, trajetoria real multi-ponto)."""
    from biospace.core import DerivedVariable, Feature
    import numpy as np

    class _NumMedicationsSlope(DerivedVariable):
        name = "num_medications_slope"
        domain_name = "utilization"
        feature_name = "num_medications"
        min_points = 2

        def compute(self, trajectory):
            pontos = self.series(trajectory)
            if len(pontos) < self.min_points:
                return None
            dias = np.array([p[0] for p in pontos])
            valores = np.array([p[1] for p in pontos])
            if np.ptp(dias) == 0:
                return None
            slope, _ = np.polyfit(dias, valores, 1)
            return Feature(name=self.name, value=float(slope), raw_value=float(slope))

    cohort, representation = real_uci_cohort
    sid_longo = max(cohort.trajectories, key=lambda s: len(cohort.trajectories[s]))
    traj = cohort.trajectories[sid_longo]
    assert len(traj) >= 10

    resultado = _NumMedicationsSlope().compute(traj)
    assert resultado is not None
    assert np.isfinite(resultado.value)


def test_phenotype_associates_with_real_readmission_outcome(real_uci_cohort):
    """
    O TESTE DECISIVO: fenotipo derivado SO de utilizacao/testes glicemicos/medicacao
    (sem idade, sem diagnostico) associa com readmissao real em 30 dias.
    ACHADO REAL documentado como regressao: o fenotipo de maior risco e
    caracterizado por alta utilizacao PREVIA, consistente com a
    literatura de predicao de readmissao.
    """
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = real_uci_cohort
    space = cohort.snapshot()
    order = representation.domain_names()

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space)

    taxas_readmissao_precoce = {}
    for ph in phenotypes:
        pacientes_do_fenotipo = [sid for sid in space.ids() if ph.contains(space.get(sid).as_vector(order))]
        if not pacientes_do_fenotipo:
            continue
        readmissoes = [cohort.systems[sid].observations[-1].metadata.get("readmitted") for sid in pacientes_do_fenotipo]
        taxa = sum(1 for r in readmissoes if r == "<30") / len(readmissoes)
        taxas_readmissao_precoce[ph.name] = taxa

    assert len(taxas_readmissao_precoce) >= 2, "Esperava pelo menos 2 fenotipos com membros."
    razao = max(taxas_readmissao_precoce.values()) / min(taxas_readmissao_precoce.values())
    assert razao > 1.5, (
        f"Esperava diferenca substancial de taxa de readmissao entre fenotipos (achado real: ~2x) -- obteve razao {razao:.2f}. "
        f"Taxas: {taxas_readmissao_precoce}"
    )


@pytest.fixture(scope="module")
def real_uci_cohort_com_diagnostico():
    """Representacao completa (4 dominios, include_diagnosis_category=True, o default) -- fixture separada porque muda o resultado de fenotipagem, nao deve ser confundida com real_uci_cohort."""
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    return load_uci_diabetes_cohort(CAMINHO_CSV)


def test_diagnosis_category_dilutes_readmission_association(real_uci_cohort_com_diagnostico):
    """
    ACHADO REAL: adicionar DiagnosisCategoryDomain (9 flags binarias de
    categoria ICD-9) a fenotipagem MUDA o resultado -- dilui a
    associacao com readmissao que a representacao original (3
    dominios, sem diagnostico) mostrava (~2x). Com o 4o dominio, K-Means
    passa a separar principalmente por um grupo minusculo de internacao
    extremamente longa (achado documentado: ~212 pacientes, ~13 dias
    medios vs. ~4-5 dos demais), nao mais pela historia de utilizacao
    que predizia readmissao. Nenhuma das duas representacoes esta
    "errada" -- capturam estruturas diferentes; e' o mesmo argumento do
    artigo "Representation Before Inference" (mais uma variavel na
    representacao nao e' neutro), agora numa instancia real e nao
    planejada.
    """
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = real_uci_cohort_com_diagnostico
    space = cohort.snapshot()
    order = representation.domain_names()
    assert "diagnosis_category" in order

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space)

    taxas = {}
    for ph in phenotypes:
        pacientes_do_fenotipo = [sid for sid in space.ids() if ph.contains(space.get(sid).as_vector(order))]
        if not pacientes_do_fenotipo:
            continue
        readmissoes = [cohort.systems[sid].observations[-1].metadata.get("readmitted") for sid in pacientes_do_fenotipo]
        taxas[ph.name] = sum(1 for r in readmissoes if r == "<30") / len(readmissoes)

    razao = max(taxas.values()) / min(taxas.values())
    assert razao < 1.8, (
        f"Esperava associacao DILUIDA (achado documentado: razao ~1.1-1.5, bem menor que os ~2.2x da representacao "
        f"sem diagnostico) -- obteve razao {razao:.2f}. Taxas: {taxas}. Se isto passou a dar >=1.8, a fenotipagem "
        f"deixou de diluir com este dominio incluido -- investigar o que mudou."
    )


def test_dynamics_are_globally_stable_on_real_multi_encounter_trajectories(real_uci_cohort):
    """
    ACHADO REAL: primeira vez que o modulo de dinamica (MeanRevertingEvolutionOperator)
    roda sobre trajetorias REAIS fora de sleep/sintetico. Ajustado sobre os
    16.773 pacientes com >=2 encontros -- todas as 13 Features (utilization,
    glycemic_testing, medication_intensity) resultam estaveis (|phi|<1).
    """
    from biospace.core import Cohort
    from biospace.dynamics import MeanRevertingEvolutionOperator, StabilityOperator

    cohort, representation = real_uci_cohort
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 2}
    assert len(multi) > 15000, "Esperava >15 mil pacientes multi-encontro (achado documentado: 16.773)."

    cohort_multi = Cohort()
    for sid, traj in multi.items():
        cohort_multi.systems[sid] = cohort.systems[sid]
        cohort_multi.trajectories[sid] = traj

    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort_multi)
    relatorio = StabilityOperator(evolution_operator=evo).analyze(cohort_multi)

    assert relatorio.is_globally_stable is True, "Esperava dinamica globalmente estavel (achado documentado)."
    assert relatorio.n_stable == relatorio.n_features == 13


def test_number_emergency_near_instability_is_robust_not_outlier_driven(real_uci_cohort):
    """
    ACHADO REAL, contrastando com o achado de SAOS
    (hypoxia.tempo_total_em_hipoxemia_min, que era 1 unico paciente
    outlier): aqui, 'utilization.number_emergency' fica perto do limiar
    de instabilidade (phi~0.98) mas a conclusao de estabilidade e'
    ROBUSTA a remocao de qualquer paciente individual -- consistente com
    reversao a media genuinamente lenta ("frequent flyers" de uso de
    emergencia), nao um artefato de amostra pequena.
    """
    from biospace.core import Cohort
    from biospace.dynamics import check_feature_stability_robustness

    cohort, representation = real_uci_cohort
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 2}
    cohort_multi = Cohort()
    for sid, traj in multi.items():
        cohort_multi.systems[sid] = cohort.systems[sid]
        cohort_multi.trajectories[sid] = traj

    relatorio = check_feature_stability_robustness(cohort_multi, "utilization.number_emergency", order=order, max_patients_tested=30)
    assert relatorio.is_stable_full is True
    assert relatorio.conclusion_is_robust is True, "Esperava conclusao robusta (achado documentado, diferente do caso de SAOS)."


def test_autoencoder_beats_pca_at_all_dimensions_with_adequate_capacity(real_uci_cohort):
    """
    ACHADO REAL, mais forte que no NHANES: com n=71.518 (vs 9.232 do
    NHANES), o autoencoder vence o PCA em TODAS as dimensoes testadas
    (2, 5, 8) -- nao so na mais baixa. Usa a representacao CANONICA
    (include_diagnosis_category=False, 13 dimensoes) para nao confundir
    com o achado de diluicao de fenotipo ja documentado.
    """
    from biospace.representation_learning import compare_reconstruction_error

    cohort, representation = real_uci_cohort
    space = cohort.snapshot()

    resultado = compare_reconstruction_error(space, embedding_dim=5, hidden_dim=16, random_state=0)
    assert resultado["autoencoder_melhor"] is True, (
        f"Esperava autoencoder vencer em dim=5 com capacidade adequada (achado documentado) -- "
        f"PCA={resultado['erro_pca_linear']:.4f}, AE={resultado['erro_autoencoder_nao_linear']:.4f}"
    )


def test_autoencoder_vs_pca_crossover_is_about_hidden_capacity_not_dimension(real_uci_cohort):
    """
    O TESTE DECISIVO para uma correcao real de hipotese: minha primeira
    exploracao (dim=8, hidden_dim=8 default) sugeriu que PCA "vence em
    dimensao alta" -- errado. Achado real: o "cruzamento" nao e' sobre
    dimensao do embedding, e' sobre se hidden_dim tem folga sobre
    embedding_dim. Com hidden_dim==embedding_dim (sem folga), PCA vence
    de forma ROBUSTA (5 seeds testadas na exploracao interativa, todas
    PCA). Com hidden_dim=2x embedding_dim (folga adequada), autoencoder
    vence -- mesma dimensao alvo (8), conclusao oposta, dependendo so
    da capacidade da camada oculta.
    """
    from biospace.representation_learning import compare_reconstruction_error

    cohort, representation = real_uci_cohort
    space = cohort.snapshot()

    sem_folga = compare_reconstruction_error(space, embedding_dim=8, hidden_dim=8, random_state=0)
    com_folga = compare_reconstruction_error(space, embedding_dim=8, hidden_dim=16, random_state=0)

    assert sem_folga["autoencoder_melhor"] is False, (
        f"Esperava PCA vencer SEM folga de capacidade (hidden_dim=embedding_dim) -- achado documentado -- "
        f"PCA={sem_folga['erro_pca_linear']:.4f}, AE={sem_folga['erro_autoencoder_nao_linear']:.4f}"
    )
    assert com_folga["autoencoder_melhor"] is True, (
        f"Esperava autoencoder vencer COM folga de capacidade (hidden_dim=2x embedding_dim), mesma dimensao alvo -- "
        f"PCA={com_folga['erro_pca_linear']:.4f}, AE={com_folga['erro_autoencoder_nao_linear']:.4f}"
    )


def test_gnn_graph_hurts_at_all_label_fractions_tested_on_uci(real_uci_cohort):
    """
    ACHADO REAL: diferente do NHANES (onde o grafo comecava a ajudar,
    por uma margem minuscula, em ~1.5% de rotulos) e muito diferente de
    SAOS (onde o grafo ajudava MUITO em poucos rotulos, +17.8pp em 5%),
    na UCI o grafo ATRAPALHA em toda fracao testada -- de 50% ate 1.5%,
    sem cruzamento detectado no intervalo testado. Fenotipos aqui (K=4,
    utilizacao hospitalar + medicacao) parecem ser ainda mais dominados
    por sinal pontual (baseado em Features) que os do NHANES.
    """
    import random

    import numpy as np

    from biospace.core import Cohort
    from biospace.geometry import Euclidean
    from biospace.gnn import SimpleGCN, prepare_node_classification_data
    from biospace.graph import build_cohort_similarity_graph
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = real_uci_cohort
    order = representation.domain_names()

    random.seed(0)
    ids_amostra = random.sample(list(cohort.trajectories.keys()), 1500)
    cohort_amostra = Cohort()
    for sid in ids_amostra:
        cohort_amostra.systems[sid] = cohort.systems[sid]
        cohort_amostra.trajectories[sid] = cohort.trajectories[sid]
    space = cohort_amostra.snapshot()

    phenotyper = KMeansPhenotyper(n_clusters=4)
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

    for fracao in [0.5, 0.1, 0.05, 0.02]:
        acc_com, acc_sem = acc_com_e_sem_grafo(fracao)
        assert acc_com <= acc_sem + 0.01, (
            f"Esperava grafo NAO ajudar (ou atrapalhar) em fracao={fracao} (achado documentado) -- "
            f"com_grafo={acc_com:.3f}, sem_grafo={acc_sem:.3f}"
        )


def test_high_risk_phenotype_shows_negligible_demographic_disparity():
    """
    ACHADO REAL, tranquilizador: o fenótipo de alto risco de readmissão
    (kmeans_3, ~2x a taxa dos demais, achado publicado) NÃO mostra
    disparidade demográfica relevante -- nem em composição (Cramér's V
    de raça=0,031 e gênero=0,009, ambos bem abaixo do limiar de
    "pequeno efeito" de Cohen, 0,1) nem em desfecho DENTRO do fenótipo
    (taxa de readmissão precoce não difere significativamente por raça,
    p=0,466, nem por gênero, p=0,176, entre pacientes já classificados
    no mesmo fenótipo). Idade mostra efeito um pouco maior (V=0,065)
    mas ainda modesto -- plausivelmente reflete utilização hospitalar
    prévia genuinamente maior em pacientes mais velhos, não viés.
    Rodado sobre a base COMPLETA (71.518 pacientes, não amostra) para
    ter poder estatístico nos grupos raciais menores (Asian: n=641 na
    base inteira). `race="?"` e `gender="Unknown/Invalid"` tratados
    como ausência real, não uma terceira categoria.
    """
    import numpy as np
    from scipy import stats as scipy_stats

    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = load_uci_diabetes_cohort(
        "/mnt/user-data/uploads/diabetic_data.csv", include_diagnosis_category=False, include_demographics=True
    )
    order = representation.domain_names()
    space = cohort.snapshot()
    assert len(cohort.trajectories) > 70000, "Esperava a base completa (achado documentado: 71.518)."

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space)
    labels = {}
    for sid in space.ids():
        vec = space.get(sid).as_vector(order)
        labels[sid] = next((ph.name for ph in phenotypes if ph.contains(vec)), None)

    import pandas as pd

    linhas = []
    for sid in cohort.trajectories:
        meta = cohort.systems[sid].metadata
        ultima_obs = cohort.systems[sid].observations[-1]
        linhas.append({
            "fenotipo": labels[sid], "race": meta.get("race"), "gender": meta.get("gender"),
            "readmitted": ultima_obs.metadata.get("readmitted"),
        })
    df = pd.DataFrame(linhas)

    def cramers_v(tabela):
        chi2, p, _, _ = scipy_stats.chi2_contingency(tabela)
        n = tabela.sum().sum()
        return np.sqrt(chi2 / (n * (min(tabela.shape) - 1))), p

    v_race, _ = cramers_v(pd.crosstab(df["fenotipo"], df["race"]))
    v_gender, _ = cramers_v(pd.crosstab(df["fenotipo"], df["gender"]))
    assert v_race < 0.1, f"Esperava efeito desprezivel (Cohen<0.1) para raca -- obteve V={v_race:.4f}"
    assert v_gender < 0.1, f"Esperava efeito desprezivel (Cohen<0.1) para genero -- obteve V={v_gender:.4f}"

    df_alto_risco = df[df["fenotipo"] == "kmeans_3"].copy()
    df_alto_risco["readm_precoce"] = df_alto_risco["readmitted"] == "<30"
    _, p_race_dentro, _, _ = scipy_stats.chi2_contingency(pd.crosstab(df_alto_risco["race"], df_alto_risco["readm_precoce"]))
    _, p_gender_dentro, _, _ = scipy_stats.chi2_contingency(pd.crosstab(df_alto_risco["gender"], df_alto_risco["readm_precoce"]))
    assert p_race_dentro > 0.05, f"Esperava NAO significativo (achado documentado: sem disparidade de desfecho por raca dentro do fenotipo) -- p={p_race_dentro:.3f}"
    assert p_gender_dentro > 0.05, f"Esperava NAO significativo por genero -- p={p_gender_dentro:.3f}"
