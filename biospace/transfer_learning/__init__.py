"""
biospace.transfer_learning
==============================

Transfer learning de verdade, não a demonstração puramente teórica de
functor já feita no Artigo I desta série: pré-treina um autoencoder
(`biospace.representation_learning.AutoencoderRepresentationLearner`,
já existente) sobre uma população GERAL, sem rótulo — depois usa o
espaço latente aprendido como característica de entrada para uma
tarefa-alvo específica, com POUCOS exemplos rotulados, e compara
contra treinar um classificador diretamente sobre as Features brutas
com os mesmos poucos exemplos (a comparação decisiva: transferência
deveria ajudar precisamente quando o exemplo-alvo é escasso e o
espaço latente pré-treinado captura estrutura genuinamente
compartilhada e relevante para a tarefa-alvo).

Esta comparação (`with_transfer` vs. `from_scratch`, mesmo orçamento
de rótulos) é o teste que separa transfer learning genuíno de
apenas "mais uma redução de dimensionalidade" — sem ele, não há como
saber se a transferência ajudou ou atrapalhou.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["TransferLearningResult", "compare_transfer_vs_scratch"]


@dataclass
class TransferLearningResult:
    accuracy_with_transfer: float
    accuracy_from_scratch: float
    n_target_labels: int
    n_pretrain_samples: int

    @property
    def transfer_helps(self) -> bool:
        return self.accuracy_with_transfer > self.accuracy_from_scratch


def compare_transfer_vs_scratch(
    space_pretrain: "RepresentationSpace",
    space_target: "RepresentationSpace",
    target_labels: dict[str, int],
    order: list[str],
    embedding_dim: int = 4,
    hidden_dim: int = 12,
    seed: int = 0,
) -> TransferLearningResult:
    """
    `space_pretrain`: população geral, sem rótulo, usada só pra
    pré-treinar o autoencoder (pode incluir ou não os pacientes de
    `space_target` -- tipicamente uma população MAIOR e mais geral).
    `space_target`: os poucos pacientes com rótulo real, a tarefa-alvo.
    `target_labels`: rótulo binário por system_id, só pra quem está em `space_target`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from biospace.representation_learning import AutoencoderRepresentationLearner

    matrix_pretrain, _ = space_pretrain.matrix()
    matrix_target, ids_target = space_target.matrix()
    y_target = np.array([target_labels[sid] for sid in ids_target])

    autoencoder = AutoencoderRepresentationLearner(embedding_dim=embedding_dim, hidden_dim=hidden_dim, random_state=seed, max_iter=1000)
    autoencoder.fit(space_pretrain, order=order)
    # transform() opera sobre um vetor por vez (nao um lote) -- API confirmada antes de assumir
    features_transferidas = np.stack([autoencoder.transform(matrix_target[i]) for i in range(matrix_target.shape[0])])

    cv = StratifiedKFold(n_splits=min(5, int(np.sum(y_target))), shuffle=True, random_state=seed) if 0 < np.sum(y_target) < len(y_target) else None
    if cv is None:
        raise ValueError("target_labels precisa ter pelo menos um exemplo de cada classe, com folds suficientes.")

    clf_transfer = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores_transfer = cross_val_score(clf_transfer, features_transferidas, y_target, cv=cv, scoring="accuracy")

    clf_scratch = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores_scratch = cross_val_score(clf_scratch, matrix_target, y_target, cv=cv, scoring="accuracy")

    return TransferLearningResult(
        accuracy_with_transfer=float(np.mean(scores_transfer)),
        accuracy_from_scratch=float(np.mean(scores_scratch)),
        n_target_labels=len(ids_target),
        n_pretrain_samples=matrix_pretrain.shape[0],
    )
