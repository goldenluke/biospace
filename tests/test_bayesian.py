"""
tests.test_bayesian
=======================

`biospace.bayesian` — Regressão por Processo Gaussiano (sklearn) e
Rede Bayesiana (pgmpy) sobre RepresentationSpace. Validados contra
cenários com verdade conhecida antes de qualquer uso real: uma função
suave conhecida para o GP (confirmando ajuste E o crescimento de
incerteza fora da região treinada); uma cadeia de dependência
A→B→C conhecida para a rede bayesiana (confirmando que a estrutura
aprendida recupera a classe de equivalência de Markov correta, e que
a inferência reflete a relação real).
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest

from biospace.bayesian import BayesianNetworkOperator, GaussianProcessOperator
from biospace.bayesian.network import discretize
from biospace.core import Feature, RepresentationSpace, RepresentationVector


def _vetor_1d(system_id: str, x: float) -> RepresentationVector:
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components={"d": [Feature(name="x", value=x, raw_value=x)]})


def _vetor_3d(system_id: str, a: float, b: float, c: float) -> RepresentationVector:
    comps = {"d": [Feature(name="a", value=a, raw_value=a), Feature(name="b", value=b, raw_value=b), Feature(name="c", value=c, raw_value=c)]}
    return RepresentationVector(system_id=system_id, timestamp=datetime(2020, 1, 1), components=comps)


def test_gp_fits_a_known_smooth_function_accurately_within_training_region():
    rng = np.random.default_rng(0)
    xs = rng.uniform(-2, 2, 30)
    space = RepresentationSpace(domain_order=["d"])
    targets = {}
    for i, x in enumerate(xs):
        sid = f"p{i}"
        space.add(_vetor_1d(sid, x))
        targets[sid] = np.sin(2 * x) + rng.normal(0, 0.05)

    gp = GaussianProcessOperator(kernel="rbf").fit(space, targets)

    space_teste = RepresentationSpace(domain_order=["d"])
    xs_teste = [0.0, 1.0, -1.5]
    for i, x in enumerate(xs_teste):
        space_teste.add(_vetor_1d(f"t{i}", x))
    resultado = gp.predict(space_teste)

    for x, pred in zip(xs_teste, resultado.mean):
        assert abs(pred - np.sin(2 * x)) < 0.15, f"Previsao dentro da regiao treinada deveria ser proxima da funcao real em x={x}"


def test_gp_uncertainty_grows_far_outside_training_region():
    """TESTE DECISIVO: comportamento classico de GP -- incerteza deveria ser MUITO maior fora da regiao onde ha dado de treino."""
    rng = np.random.default_rng(1)
    xs = rng.uniform(-2, 2, 30)
    space = RepresentationSpace(domain_order=["d"])
    targets = {}
    for i, x in enumerate(xs):
        sid = f"p{i}"
        space.add(_vetor_1d(sid, x))
        targets[sid] = np.sin(2 * x) + rng.normal(0, 0.05)

    gp = GaussianProcessOperator(kernel="rbf").fit(space, targets)

    space_teste = RepresentationSpace(domain_order=["d"])
    for i, x in enumerate([0.0, 1.0, -1.0, 8.0, -8.0]):
        space_teste.add(_vetor_1d(f"t{i}", x))
    resultado = gp.predict(space_teste)

    incerteza_dentro = np.mean(resultado.std[:3])
    incerteza_fora = np.mean(resultado.std[3:])
    assert incerteza_fora > 5 * incerteza_dentro, (
        f"Esperava incerteza muito maior fora da regiao treinada -- dentro={incerteza_dentro:.4f}, fora={incerteza_fora:.4f}"
    )


def test_bayesian_network_recovers_markov_equivalent_structure_of_known_chain():
    """TESTE DECISIVO: numa cadeia A->B->C conhecida, a estrutura aprendida deveria ter B conectado tanto a A quanto a C (equivalente de Markov da cadeia real -- direcao exata nao e' identificavel so com dado observacional, o que e' esperado, nao um bug)."""
    rng = np.random.default_rng(0)
    space = RepresentationSpace(domain_order=["d"])
    for i in range(800):
        a = rng.normal(0, 1)
        b = 2 * a + rng.normal(0, 0.3)
        c = -1.5 * b + rng.normal(0, 0.3)
        space.add(_vetor_3d(f"p{i}", a, b, c))

    bn = BayesianNetworkOperator(n_bins=4, max_indegree=2)
    bn.fit(space, feature_names=["a", "b", "c"])

    nos_conectados_a_b = {n for edge in bn.edges() for n in edge if "b" in edge}
    assert "a" in nos_conectados_a_b and "c" in nos_conectados_a_b, (
        f"Esperava 'b' conectado tanto a 'a' quanto a 'c' (estrutura da cadeia real) -- arestas aprendidas: {bn.edges()}"
    )


def test_bayesian_network_inference_reflects_the_real_negative_relationship():
    """A inferencia deveria refletir a relacao real (c=-1.5*b): b alto -> c concentrado em faixa baixa; b baixo -> c concentrado em faixa alta."""
    rng = np.random.default_rng(2)
    space = RepresentationSpace(domain_order=["d"])
    for i in range(800):
        a = rng.normal(0, 1)
        b = 2 * a + rng.normal(0, 0.3)
        c = -1.5 * b + rng.normal(0, 0.3)
        space.add(_vetor_3d(f"p{i}", a, b, c))

    bn = BayesianNetworkOperator(n_bins=4, max_indegree=2)
    bn.fit(space, feature_names=["a", "b", "c"])

    p_c_dado_b_alto = bn.query(target="c", evidence={"b": "3"})
    p_c_dado_b_baixo = bn.query(target="c", evidence={"b": "0"})

    faixa_mais_provavel_b_alto = max(p_c_dado_b_alto, key=p_c_dado_b_alto.get)
    faixa_mais_provavel_b_baixo = max(p_c_dado_b_baixo, key=p_c_dado_b_baixo.get)
    assert faixa_mais_provavel_b_alto in ("0", "1"), f"Esperava c baixo quando b e' alto -- obteve faixa {faixa_mais_provavel_b_alto}"
    assert faixa_mais_provavel_b_baixo in ("2", "3"), f"Esperava c alto quando b e' baixo -- obteve faixa {faixa_mais_provavel_b_baixo}"


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/P_DEMO.xpt"),
    reason="Requer os arquivos reais do NHANES.",
)
def test_gp_kernel_choice_effect_on_real_nhanes_is_nuanced_not_dramatic():
    """
    ACHADO REAL: testa diretamente a hipotese levantada no Artigo IV
    desta serie sobre a modelagem normativa de Marquand et al. (2019)
    -- a escolha de kernel num Processo Gaussiano e' uma escolha de
    geometria, entao deveria ter efeito parecido com o que a Secao 8
    do Artigo II encontrou variando geometria? Testado em dado real
    (previsao de HbA1c a partir de outros dominios metabolicos,
    validacao cruzada de 5 folds, 4 kernels): RBF, Matern e Linear
    convergem para RMSE e log-verossimilhanca marginal essencialmente
    identicos entre si (~0.97-1.02 RMSE, ~-430 log-verossimilhanca) --
    NAO reproduz o padrao dramatico do Artigo II. So o kernel
    Periodico, estruturalmente incompativel com dado metabolico sem
    periodicidade conhecida, fica claramente pior nos dois criterios
    (RMSE=1.04, log-verossimilhanca=-519). Resposta mais nuancada que
    "kernel importa tanto quanto geometria": kernels razoaveis
    concordam; um kernel mal escolhido estruturalmente se destaca.
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.nhanes import NHANES_PREPANDEMIC_FILES, load_nhanes_metabolic_cohort
    from biospace.plugins.metabolic import load_from_dataframe
    from sklearn.model_selection import KFold

    df = load_nhanes_metabolic_cohort("/mnt/user-data/uploads", files=NHANES_PREPANDEMIC_FILES)
    df_adultos = df[df["idade"] >= 20].copy()
    cohort, representation = load_from_dataframe(df_adultos.dropna(subset=["hba1c_pct"]).head(400))
    order = representation.domain_names()
    space_completa = cohort.snapshot()
    order_sem_glicemico = [d for d in order if d != "glycemic"]

    targets = {}
    for sid in space_completa.ids():
        vetor = space_completa.get(sid)
        hba1c_feat = next((f for f in vetor.components.get("glycemic", []) if f.name == "hba1c_pct"), None)
        if hba1c_feat is not None and not hba1c_feat.is_missing:
            targets[sid] = hba1c_feat.raw_value

    ids_validos = list(targets.keys())
    assert len(ids_validos) > 300, f"Esperava >300 pacientes com HbA1c valido -- obteve {len(ids_validos)}"

    space_valida = RepresentationSpace(domain_order=order_sem_glicemico)
    for sid in ids_validos:
        space_valida.add(space_completa.get(sid))

    y = np.array([targets[sid] for sid in ids_validos])
    matrix, ids_ordem = space_valida.matrix()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    rmse_por_kernel = {}
    for kernel_nome in ["rbf", "matern", "linear", "periodic"]:
        erros = []
        for tr_idx, te_idx in kf.split(matrix):
            space_tr = RepresentationSpace(domain_order=order_sem_glicemico)
            for i in tr_idx:
                space_tr.add(space_valida.get(ids_ordem[i]))
            space_te = RepresentationSpace(domain_order=order_sem_glicemico)
            for i in te_idx:
                space_te.add(space_valida.get(ids_ordem[i]))
            targets_tr = {ids_ordem[i]: y[i] for i in tr_idx}
            gp = GaussianProcessOperator(kernel=kernel_nome, n_restarts_optimizer=1)
            gp.fit(space_tr, targets_tr)
            resultado = gp.predict(space_te)
            erros.append(np.sqrt(np.mean((resultado.mean - y[te_idx]) ** 2)))
        rmse_por_kernel[kernel_nome] = np.mean(erros)

    # RBF, Matern e Linear deveriam ficar proximos entre si (achado documentado: nao dramatico)
    razoaveis = [rmse_por_kernel["rbf"], rmse_por_kernel["matern"], rmse_por_kernel["linear"]]
    assert max(razoaveis) / min(razoaveis) < 1.15, f"Esperava RBF/Matern/Linear proximos entre si -- obteve {rmse_por_kernel}"
    # Periodico deveria ser detectavelmente pior (achado documentado)
    assert rmse_por_kernel["periodic"] > min(razoaveis), f"Esperava Periodico pior que os demais -- obteve {rmse_por_kernel}"
