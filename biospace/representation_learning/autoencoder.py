"""
biospace.representation_learning.autoencoder
===============================================

AutoencoderRepresentationLearner: aprendizado de representação NÃO
LINEAR em cima de X — via `sklearn.neural_network.MLPRegressor`
configurado como autoencoder (entrada = saída = X, com uma camada
intermediária estreita — o "gargalo" — de dimensão `embedding_dim`).

POR QUE NÃO TORCH/TENSORFLOW: testado — `pip install torch` baixa ~1GB
de dependências CUDA (mesmo para uso puramente em CPU), inadequado para
um framework que deve continuar leve e testável em CI. `MLPRegressor`
(já em scikit-learn, dependência existente do projeto) treina uma rede
com camadas ocultas de verdade (pesos, ativação não linear, backprop via
L-BFGS/Adam) — é um autoencoder genuíno, só que sem os megabytes extra.

DIFERENÇA em relação a `FactorAnalysisLatentDomain` (já existente): a
Análise Fatorial é um modelo LINEAR (observado = cargas @ fator +
ruído). Aqui, as ativações não lineares entre camadas permitem capturar
relações que uma combinação linear não capturaria — testado e
confirmado em cenário sintético (ver README): o autoencoder reconstrói
uma estrutura latente não linear conhecida com MENOS erro que PCA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .base import RepresentationLearner

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["AutoencoderRepresentationLearner"]

_ACTIVATIONS = {
    "relu": lambda z: np.maximum(0, z),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "identity": lambda z: z,
}


class AutoencoderRepresentationLearner(RepresentationLearner):
    """
    Arquitetura: entrada -> `hidden_dim` -> `embedding_dim` (gargalo) ->
    `hidden_dim` -> saída (=entrada). `transform()` extrai a ativação NO
    GARGALO (não a reconstrução final) — esse é o embedding aprendido.

    Padroniza as Features internamente (média 0, desvio 1) antes de
    treinar — necessário para o MLP convergir bem; a padronização é
    revertida apenas na reconstrução (não afeta `transform()`, que
    devolve o embedding, sempre numa escala própria, não a escala de X).
    """

    def __init__(
        self,
        embedding_dim: int = 2,
        hidden_dim: int = 8,
        activation: str = "relu",
        max_iter: int = 2000,
        random_state: int = 42,
        alpha: float = 1e-3,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation deve ser um de {list(_ACTIVATIONS)}, recebeu {activation!r}.")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self._mlp = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, space: "RepresentationSpace", order: Optional[Sequence[str]] = None) -> "AutoencoderRepresentationLearner":
        from sklearn.neural_network import MLPRegressor

        matrix, ids = space.matrix()
        if order is not None and list(order) != space.order():
            # se uma ordem diferente da nativa do space for pedida, reconstrói a matriz nessa ordem
            matrix = np.stack([space.get(sid).as_vector(order) for sid in ids])

        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0)
        self._std[self._std < 1e-9] = 1.0
        X = (matrix - self._mean) / self._std

        self._mlp = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.embedding_dim, self.hidden_dim),
            activation=self.activation,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        self._mlp.fit(X, X)
        self.is_fitted = True
        return self

    def _forward_to_bottleneck(self, x_std: np.ndarray) -> np.ndarray:
        """Propaga `x_std` (já padronizado) até a camada do gargalo (2 primeiras matrizes de peso)."""
        act = _ACTIVATIONS[self.activation]
        a = x_std
        for i in range(2):  # entrada->hidden_dim, hidden_dim->embedding_dim (o gargalo)
            a = act(a @ self._mlp.coefs_[i] + self._mlp.intercepts_[i])
        return a

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(space) deve ser chamado antes de transform().")
        x_std = (x - self._mean) / self._std
        return self._forward_to_bottleneck(x_std.reshape(1, -1))[0]

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstrução completa (entrada -> gargalo -> saída), na escala ORIGINAL de X — para medir erro de reconstrução."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(space) deve ser chamado antes de reconstruct().")
        x_std = (x - self._mean) / self._std
        pred_std = self._mlp.predict(x_std.reshape(1, -1))[0]
        return pred_std * self._std + self._mean

    def reconstruction_error(self, space: "RepresentationSpace", order: Optional[Sequence[str]] = None) -> float:
        """Erro quadrático médio de reconstrução sobre todos os pontos de `space` — diagnóstico de quanta estrutura o embedding preserva."""
        used_order = order or space.order()
        errors = []
        for sid in space.ids():
            x = space.get(sid).as_vector(used_order)
            recon = self.reconstruct(x)
            errors.append(np.mean((x - recon) ** 2))
        return float(np.mean(errors))

    def describe(self) -> str:
        status = "ajustado" if self.is_fitted else "não ajustado"
        return f"AutoencoderRepresentationLearner(embedding_dim={self.embedding_dim}, hidden_dim={self.hidden_dim}, {status})"


def compare_reconstruction_error(
    space: "RepresentationSpace",
    embedding_dim: int = 2,
    order: Optional[Sequence[str]] = None,
    **autoencoder_kwargs,
) -> dict:
    """
    Ajusta AMBOS — `AutoencoderRepresentationLearner` (não linear) e PCA
    (linear) — na MESMA dimensão de embedding, e compara o erro de
    reconstrução. Existe porque, testado nos dados reais deste projeto
    (355 pacientes, 52 dimensões), o autoencoder ficou PIOR que PCA em
    toda dimensão e configuração de hiperparâmetros testada — o
    contrário do que aconteceu num cenário sintético com estrutura não
    linear conhecida (autoencoder ~4,5x melhor). Lição: redes neurais
    precisam de mais dados que métodos lineares para encontrar uma
    solução melhor; com poucas centenas de pacientes (comum em dados
    clínicos reais), um método LINEAR pode ser mais confiável, não
    apesar de ser mais simples, mas POR CAUSA disso. Use esta função
    para verificar qual dos dois vale a pena, em vez de assumir que
    "não linear" é sempre melhor.
    """
    from sklearn.decomposition import PCA

    matrix, ids = space.matrix()
    used_order = order or space.order()
    if order is not None and list(order) != space.order():
        matrix = np.stack([space.get(sid).as_vector(used_order) for sid in ids])

    pca = PCA(n_components=embedding_dim)
    pca.fit(matrix)
    recon_pca = pca.inverse_transform(pca.transform(matrix))
    erro_pca = float(np.mean((matrix - recon_pca) ** 2))

    ae = AutoencoderRepresentationLearner(embedding_dim=embedding_dim, **autoencoder_kwargs)
    ae.fit(space, order=order)
    erro_ae = ae.reconstruction_error(space, order=order)

    return {
        "embedding_dim": embedding_dim,
        "erro_pca_linear": erro_pca,
        "erro_autoencoder_nao_linear": erro_ae,
        "autoencoder_melhor": erro_ae < erro_pca,
        "recomendacao": "autoencoder" if erro_ae < erro_pca else "pca",
    }
