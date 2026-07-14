"""
tests.test_transfer_learning
================================

`biospace.transfer_learning.compare_transfer_vs_scratch` — validado
em duas camadas, na ordem em que a investigação real aconteceu:
primeiro, um cenário onde a tarefa-alvo já é fácil o bastante com
poucos exemplos brutos (Features majoritariamente monotônicas em z,
20 rótulos) — transferência NÃO ajudou (0,70 vs. 0,95 do zero),
porque o cenário não tinha a condição que faz transfer learning
genuíno valer a pena. Investigado e corrigido: com Features ruidosas
de alta dimensão e poucos rótulos (o regime few-shot genuíno),
transferência ajuda de forma decisiva.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.transfer_learning import compare_transfer_vs_scratch


def _vetor_facil(sid, z, rng):
    x1 = np.sin(z) + rng.normal(0, 0.1)
    x2 = z ** 2 + rng.normal(0, 0.1)
    x3 = np.cos(z * 1.5) + rng.normal(0, 0.1)
    x4 = z * 0.5 + rng.normal(0, 0.1)
    x5 = np.tanh(z) + rng.normal(0, 0.1)
    feats = [Feature(name=f"x{i}", value=v, raw_value=v) for i, v in enumerate([x1, x2, x3, x4, x5], 1)]
    return RepresentationVector(system_id=sid, timestamp=datetime(2020, 1, 1), components={"d": feats})


def _vetor_dificil(sid, z, rng, n_ruido=15):
    x1 = np.sin(z) + rng.normal(0, 0.3)
    x2 = z ** 2 + rng.normal(0, 0.3)
    x3 = np.cos(z * 1.5) + rng.normal(0, 0.3)
    feats = [Feature(name="x1", value=x1, raw_value=x1), Feature(name="x2", value=x2, raw_value=x2), Feature(name="x3", value=x3, raw_value=x3)]
    for j in range(n_ruido):
        v = rng.normal(0, 1)
        feats.append(Feature(name=f"ruido{j}", value=v, raw_value=v))
    return RepresentationVector(system_id=sid, timestamp=datetime(2020, 1, 1), components={"d": feats})


def test_transfer_does_not_help_when_target_task_is_already_easy_from_raw_features():
    """
    ACHADO REAL: quando as Features brutas ja tem uma relacao quase
    monotonica/linear com o rotulo, e ha exemplos suficientes (20),
    treinar do zero supera a transferencia -- nao ha "estrutura
    escondida" a mais pra recuperar via pre-treino que a regressao
    logistica direta ja nao capture. Registrado como achado real, nao
    escondido atras de um cenario artificialmente favoravel.
    """
    rng_pre = np.random.default_rng(0)
    space_pretrain = RepresentationSpace(domain_order=["d"])
    for i in range(500):
        z = rng_pre.uniform(-3, 3)
        space_pretrain.add(_vetor_facil(f"pre_{i}", z, rng_pre))

    rng_alvo = np.random.default_rng(1)
    space_target = RepresentationSpace(domain_order=["d"])
    labels = {}
    for i in range(20):
        z = rng_alvo.uniform(-3, 3)
        sid = f"alvo_{i}"
        space_target.add(_vetor_facil(sid, z, rng_alvo))
        labels[sid] = int(z > 0)

    resultado = compare_transfer_vs_scratch(space_pretrain, space_target, labels, order=["d"], embedding_dim=2, hidden_dim=8, seed=0)
    assert resultado.accuracy_from_scratch > resultado.accuracy_with_transfer, (
        f"Esperava 'do zero' superar transferencia neste cenario facil -- "
        f"do zero={resultado.accuracy_from_scratch:.3f}, transferencia={resultado.accuracy_with_transfer:.3f}"
    )


def test_transfer_helps_decisively_in_genuine_few_shot_high_noise_regime():
    """
    TESTE DECISIVO: no regime genuino de transfer learning -- poucos
    rotulos-alvo (10) E Features brutas de alta dimensao com muito
    ruido nao-informativo (15 dimensoes de ruido puro, so 3
    informativas) --, a transferencia (pre-treino em 800 exemplos sem
    rotulo) deveria superar treinar do zero por uma margem grande.
    """
    rng_pre = np.random.default_rng(0)
    space_pretrain = RepresentationSpace(domain_order=["d"])
    for i in range(800):
        z = rng_pre.uniform(-3, 3)
        space_pretrain.add(_vetor_dificil(f"pre_{i}", z, rng_pre))

    rng_alvo = np.random.default_rng(1)
    space_target = RepresentationSpace(domain_order=["d"])
    labels = {}
    for i in range(10):
        z = rng_alvo.uniform(-3, 3)
        sid = f"alvo_{i}"
        space_target.add(_vetor_dificil(sid, z, rng_alvo))
        labels[sid] = int(z > 0)

    resultado = compare_transfer_vs_scratch(space_pretrain, space_target, labels, order=["d"], embedding_dim=3, hidden_dim=10, seed=0)
    assert resultado.accuracy_with_transfer - resultado.accuracy_from_scratch > 0.3, (
        f"Esperava transferencia superar 'do zero' por margem grande no regime few-shot/ruidoso -- "
        f"transferencia={resultado.accuracy_with_transfer:.3f}, do zero={resultado.accuracy_from_scratch:.3f}"
    )


def test_raises_when_target_labels_have_only_one_class():
    space_pretrain = RepresentationSpace(domain_order=["d"])
    rng = np.random.default_rng(0)
    for i in range(50):
        space_pretrain.add(_vetor_facil(f"pre_{i}", rng.uniform(-3, 3), rng))

    space_target = RepresentationSpace(domain_order=["d"])
    labels = {}
    for i in range(5):
        sid = f"alvo_{i}"
        space_target.add(_vetor_facil(sid, 1.0, rng))
        labels[sid] = 1

    with pytest.raises(ValueError, match="classe"):
        compare_transfer_vs_scratch(space_pretrain, space_target, labels, order=["d"])


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/P_DEMO.xpt"),
    reason="Requer os arquivos reais do NHANES.",
)
def test_transfer_does_not_help_on_real_nhanes_diabetes_classification():
    """
    ACHADO REAL, coerente com o cenario sintetico 'facil' acima:
    pre-treinando em toda a populacao adulta do NHANES (sem rotulo) e
    transferindo pra uma amostra balanceada pequena (16 pacientes,
    8 diabeticos + 8 normais), treinar do zero (1,000) supera
    transferencia (0,933). Faz sentido: `classify_diabetes_status` e'
    calculado DIRETAMENTE dos valores brutos de HbA1c/glicose -- a
    representacao bruta ja contem uma regra quase tautologica pro
    rotulo, entao nao ha "estrutura escondida" que o pre-treino
    precisasse recuperar. Nao e' uma falha do metodo -- e' o mesmo
    padrao ja confirmado no cenario sintetico: transfer learning ajuda
    quando a tarefa-alvo e' genuinamente dificil de aprender direto
    das Features brutas com poucos exemplos, nao em qualquer tarefa
    com poucos rotulos.
    """
    from biospace.core import RepresentationSpace
    from biospace.datasets.nhanes import NHANES_PREPANDEMIC_FILES, load_nhanes_metabolic_cohort
    from biospace.plugins.metabolic import classify_diabetes_status, load_from_dataframe

    df = load_nhanes_metabolic_cohort("/mnt/user-data/uploads", files=NHANES_PREPANDEMIC_FILES)
    df_adultos = df[df["idade"] >= 20].copy()
    cohort, representation = load_from_dataframe(df_adultos)
    order = representation.domain_names()
    space_completo = cohort.snapshot()

    labels_completos = {}
    for sid in cohort.trajectories:
        status = classify_diabetes_status(cohort.trajectories[sid].latest())
        if status in ("diabetes", "normal"):
            labels_completos[sid] = int(status == "diabetes")

    assert len(labels_completos) > 3000, f"Esperava >3000 pacientes com rotulo valido -- obteve {len(labels_completos)}"

    rng = np.random.RandomState(0)
    diabeticos = [sid for sid, v in labels_completos.items() if v == 1]
    normais = [sid for sid, v in labels_completos.items() if v == 0]
    alvo_ids = list(rng.choice(diabeticos, 8, replace=False)) + list(rng.choice(normais, 8, replace=False))
    labels_alvo = {sid: labels_completos[sid] for sid in alvo_ids}

    space_target = RepresentationSpace(domain_order=order)
    for sid in alvo_ids:
        space_target.add(space_completo.get(sid))

    resultado = compare_transfer_vs_scratch(space_completo, space_target, labels_alvo, order=order, embedding_dim=4, hidden_dim=12, seed=0)
    assert resultado.accuracy_from_scratch >= resultado.accuracy_with_transfer, (
        f"Esperava 'do zero' igualar ou superar transferencia (achado documentado: tarefa ja facil a partir do bruto) -- "
        f"do zero={resultado.accuracy_from_scratch:.3f}, transferencia={resultado.accuracy_with_transfer:.3f}"
    )
