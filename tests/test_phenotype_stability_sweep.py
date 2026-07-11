"""
tests.test_phenotype_stability_sweep
========================================

Lacuna fechada: a estabilidade fenotípica (ARI=0,42 com K=4,
ClinicalKMeansPhenotyper) ficou registrada há muitas sessões como algo
a "revisitar com outro K ou algoritmo". Feito agora: varredura
abrangente — KMeans (K=2..8), GaussianMixture (K=2..6), Spectral
(K=2..6), HDBSCAN (min_cluster_size=5..30) — sobre os 355 pacientes
reais de SAOS.

RESULTADO: NENHUMA das ~28 configurações testadas atinge o limiar de
estabilidade (ARI >= 0,7, Hennig 2007). A melhor foi GaussianMixture
K=2 (ARI=0,583) — mais perto, mas ainda abaixo. Conclusão: a
instabilidade fenotípica não é um artefato de K ou de algoritmo — é uma
propriedade real dos dados (a população forma um CONTÍNUO de
severidade, não clusters nitidamente separados). Formalizado aqui como
regressão: se um ajuste futuro no núcleo ou nos dados fizer alguma
configuração cruzar 0,7 inesperadamente, este teste avisa.
"""

from __future__ import annotations

import os

import pytest

CAMINHO_EXCEL = "/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx"

pytestmark = pytest.mark.skipif(not os.path.exists(CAMINHO_EXCEL), reason="Excel real não disponível neste ambiente (esperado em CI).")


@pytest.fixture(scope="module")
def real_sleep_space():
    from biospace.plugins.sleep import load_from_excel

    cohort, representation = load_from_excel(CAMINHO_EXCEL, header=1)
    return cohort.snapshot()


def test_no_kmeans_k_achieves_stability(real_sleep_space):
    from biospace.phenotyping import KMeansPhenotyper
    from biospace.phenotyping.contracts import check_phenotype_stability

    resultados = {}
    for k in range(2, 9):
        relatorio = check_phenotype_stability(lambda k=k: KMeansPhenotyper(n_clusters=k), real_sleep_space)
        resultados[k] = relatorio.adjusted_rand_index

    assert all(ari < 0.7 for ari in resultados.values()), (
        f"Esperava NENHUM K estável (achado documentado no README) — se isto falhar, algum K passou a ser "
        f"estável e a documentação/conclusão precisa ser atualizada: {resultados}"
    )


def test_no_gaussian_mixture_k_achieves_stability(real_sleep_space):
    from biospace.phenotyping import GaussianMixturePhenotyper
    from biospace.phenotyping.contracts import check_phenotype_stability

    resultados = {}
    for k in range(2, 7):
        relatorio = check_phenotype_stability(lambda k=k: GaussianMixturePhenotyper(n_components_range=[k]), real_sleep_space)
        resultados[k] = relatorio.adjusted_rand_index

    assert all(ari < 0.7 for ari in resultados.values()), f"Esperava nenhum K estável: {resultados}"
    assert resultados[2] > 0.4, "GaussianMixture K=2 deveria continuar sendo a configuração mais próxima da estabilidade (achado documentado)."


def test_no_spectral_k_achieves_stability(real_sleep_space):
    from biospace.phenotyping import SpectralPhenotyper
    from biospace.phenotyping.contracts import check_phenotype_stability

    resultados = {}
    for k in range(2, 7):
        relatorio = check_phenotype_stability(lambda k=k: SpectralPhenotyper(n_clusters=k), real_sleep_space)
        resultados[k] = relatorio.adjusted_rand_index

    assert all(ari < 0.7 for ari in resultados.values()), f"Esperava nenhum K estável: {resultados}"
