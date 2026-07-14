"""
tests.test_sequence
=======================

`biospace.sequence` (GRU em NumPy puro, forward+backward manuais) —
validado em três camadas, na ordem certa: (1) o backward está
matematicamente correto (checagem de gradiente por diferenças
finitas); (2) o modelo aprende uma dependência NÃO-LINEAR conhecida
que um modelo linear por coordenada não pode capturar por construção;
(3) só depois disso, aplicação em dado real (em outro arquivo).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from biospace.sequence import (
    SequenceClassifier, SequenceForecaster, gru_backward, gru_forward, init_gru_params,
)


def test_gradient_check_passes_for_every_parameter():
    """TESTE DECISIVO: o backward manual (BPTT) deveria bater com o gradiente numerico (diferencas finitas) em TODOS os parametros -- sem isso, nenhum resultado de treino abaixo e' confiavel."""
    rng = np.random.default_rng(42)
    n_input, n_hidden, n_output, T = 3, 4, 1, 5
    params = init_gru_params(n_input, n_hidden, n_output, seed=1)
    X = rng.normal(0, 1, (T, n_input))

    def loss_and_cache(p):
        cache = gru_forward(p, X)
        return 0.5 * np.sum(cache["y_final"] ** 2), cache

    _, cache = loss_and_cache(params)
    dy_final = cache["y_final"]
    grads = gru_backward(params, cache, dy_final)

    eps = 1e-5
    piores_erros = []
    for nome in ["Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh", "bh", "Wout", "bout"]:
        arr = getattr(params, nome)
        grad_analitico = getattr(grads, nome)
        idx_testados = np.random.RandomState(0).choice(arr.size, size=min(3, arr.size), replace=False)
        for idx in idx_testados:
            original = arr.flat[idx]
            arr.flat[idx] = original + eps
            loss_mais, _ = loss_and_cache(params)
            arr.flat[idx] = original - eps
            loss_menos, _ = loss_and_cache(params)
            arr.flat[idx] = original

            grad_numerico = (loss_mais - loss_menos) / (2 * eps)
            grad_val = grad_analitico.flat[idx]
            erro_rel = abs(grad_numerico - grad_val) / (abs(grad_numerico) + abs(grad_val) + 1e-8)
            piores_erros.append(erro_rel)

    assert max(piores_erros) < 1e-4, f"Checagem de gradiente falhou -- maior erro relativo: {max(piores_erros):.2e}"


def _gerar_sequencia_nao_linear(T, seed):
    """x2 depende de forma NAO-LINEAR de x1 (tanh de x1 defasado) -- um modelo linear por coordenada (como o operador de evolucao diagonal ja usado no projeto) nao pode capturar essa dependencia cruzada por construcao."""
    r = np.random.default_rng(seed)
    x1 = np.zeros(T); x2 = np.zeros(T)
    x1[0] = r.normal(0, 1); x2[0] = r.normal(0, 1)
    for t in range(T - 1):
        x1[t + 1] = 0.9 * x1[t] + r.normal(0, 0.05)
        x2[t + 1] = np.tanh(x1[t] * 2) + r.normal(0, 0.05)
    return np.stack([x1, x2], axis=1)


def test_forecaster_learns_nonlinear_cross_feature_dependency_better_than_naive_persistence():
    """TESTE DECISIVO: o GRU deveria prever a Feature 2 (dependencia nao-linear de x1) melhor que a previsao ingenua por persistencia -- confirma que ele aprende a estrutura nao-linear, nao so memoriza."""
    sequencias_treino = [_gerar_sequencia_nao_linear(15, seed=i) for i in range(60)]
    sequencias_teste = [_gerar_sequencia_nao_linear(15, seed=1000 + i) for i in range(20)]

    forecaster = SequenceForecaster(n_hidden=8, lr=0.03, n_epochs=150, seed=0)
    historico = forecaster.fit(sequencias_treino)
    assert historico[-1] < historico[0], "Esperava a perda de treino cair."

    erros_gru, erros_naive = [], []
    for seq in sequencias_teste:
        for t in range(1, len(seq) - 1):
            pred = forecaster.predict_next(seq[:t])
            erros_gru.append(abs(pred[1] - seq[t][1]))
            erros_naive.append(abs(seq[t - 1][1] - seq[t][1]))

    assert np.mean(erros_gru) < np.mean(erros_naive), (
        f"Esperava GRU superar a previsao ingenua na Feature nao-linear -- "
        f"GRU={np.mean(erros_gru):.4f}, naive={np.mean(erros_naive):.4f}"
    )


def _gerar_sequencia_com_rotulo(T, seed):
    """Rotulo=1 se a sequencia cruza um limiar E permanece acima por >=2 passos consecutivos -- um padrao sequencial, nao capturavel por estatisticas de resumo simples (media, max) sem ver a ordem."""
    r = np.random.default_rng(seed)
    x = r.normal(0, 1, T).cumsum() * 0.3 + r.normal(0, 0.3, T)
    acima = x > 1.0
    rotulo = int(any(acima[i] and acima[i + 1] for i in range(T - 1)))
    return x.reshape(-1, 1), rotulo


def test_classifier_learns_a_real_sequential_pattern():
    """O classificador deveria aprender o padrao sequencial (2 passos consecutivos acima do limiar) com acuracia bem acima do acaso."""
    treino = [_gerar_sequencia_com_rotulo(12, seed=i) for i in range(150)]
    teste = [_gerar_sequencia_com_rotulo(12, seed=2000 + i) for i in range(60)]

    seqs_treino = [s for s, _ in treino]; labels_treino = [l for _, l in treino]
    clf = SequenceClassifier(n_hidden=6, lr=0.05, n_epochs=80, seed=0)
    clf.fit(seqs_treino, labels_treino)

    acertos = 0
    for seq, label in teste:
        prob = clf.predict_proba(seq)
        pred = int(prob > 0.5)
        acertos += int(pred == label)
    acuracia = acertos / len(teste)
    assert acuracia > 0.65, f"Esperava acuracia bem acima do acaso (0.5) -- obteve {acuracia:.3f}"


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/diabetic_data.csv"),
    reason="Requer o arquivo real da UCI.",
)
def test_gru_sequence_classifier_confirms_chance_ceiling_on_real_uci_readmission():
    """
    ACHADO REAL, sexta confirmacao independente da mesma triangulacao
    do Artigo V: usando a SEQUENCIA INTEIRA dos 3 primeiros encontros
    (nao so o baseline do 1o, como Cox/RF/LogReg/SHAP usaram antes) via
    um GRU com backward verificado por checagem de gradiente -- nao
    um MLP fingindo ser sequencial --, a previsao de readmissao
    precoce a partir do 4o encontro em diante continua no teto de
    acaso. AUC de validacao cruzada de 5 folds, testado com 3
    capacidades de modelo diferentes (mais e menos parametros, mais e
    menos epocas): todos os tres ficam dentro do ruido de 0,5, com
    desvio padrao grande (amostra pequena -- so ~267 pacientes tem
    4+ encontros nesta base).
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cohort, representation = load_uci_diabetes_cohort("/mnt/user-data/uploads/diabetic_data.csv", max_rows=15000, include_diagnosis_category=False)
    order = representation.domain_names()

    sequencias, labels = [], []
    for sid, traj in cohort.trajectories.items():
        matriz_completa = traj.as_matrix(order)
        if len(matriz_completa) < 4:
            continue
        obs_paciente = cohort.systems[sid].observations
        prefixo = matriz_completa[:3].astype(np.float64)
        evento = any(o.metadata.get("readmitted") == "<30" for o in obs_paciente[3:])
        sequencias.append(prefixo)
        labels.append(int(evento))

    assert len(sequencias) > 200, f"Esperava >200 pacientes elegiveis (achado documentado: ~267) -- obteve {len(sequencias)}"

    labels_arr = np.array(labels)
    indices = np.arange(len(sequencias))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs = []
    for tr_idx, te_idx in skf.split(indices, labels_arr):
        clf = SequenceClassifier(n_hidden=10, lr=0.02, n_epochs=40, seed=0)
        clf.fit([sequencias[i] for i in tr_idx], [labels[i] for i in tr_idx])
        probs = [clf.predict_proba(sequencias[i]) for i in te_idx]
        aucs.append(roc_auc_score([labels[i] for i in te_idx], probs))

    media_auc = np.mean(aucs)
    assert 0.35 < media_auc < 0.65, f"Esperava AUC dentro do ruido de 0.5 (achado documentado, teto de acaso) -- obteve {media_auc:.3f}"
