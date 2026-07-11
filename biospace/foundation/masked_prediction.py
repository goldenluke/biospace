"""
biospace.foundation.masked_prediction
========================================

Fase 10 — Foundation Model (protótipo de ARQUITETURA, não um foundation
model de verdade — ver aviso central abaixo):

    Milhões de pacientes -> BioSpace -> Foundation Model

MaskedFeaturePredictor: tarefa auto-supervisionada análoga ao Masked
Language Modeling do BERT (Devlin et al., 2019) — mas o "vocabulário"
não são palavras, são FEATURES FISIOLÓGICAS já semanticamente
estruturadas por SemanticDomain (cada uma com nome, domínio, proveniência
rastreável — diferente de um token de texto, que não carrega nenhum
significado intrínseco fora do modelo). Mascara uma fração aleatória das
Features de cada paciente e treina uma rede para reconstruí-las a partir
das Features NÃO mascaradas do mesmo paciente — a mesma lógica de "o
contexto revela o que falta", agora sobre fisiologia.

AVISO CENTRAL — leia antes de tirar qualquer conclusão: isto é uma PROVA
DE CONCEITO ARQUITETURAL, treinada em dezenas a centenas de pacientes
(sintéticos ou reais), não um foundation model — que exigiria "milhões
de pacientes" (a própria escala citada no pedido original). A diferença
não é de grau, é de ORDEM DE GRANDEZA: nada aqui generaliza para a
alegação de que "funcionaria em escala". O valor deste protótipo é
demonstrar que a REPRESENTAÇÃO (Features com proveniência semântica,
construída por SemanticDomain) é um substrato válido para pré-treino
auto-supervisionado — o mesmo padrão arquitetural usado por modelos de
linguagem, aplicado a um domínio onde o "token" tem significado clínico
verificável, não é uma unidade arbitrária de um tokenizador.

Implementação: MLPRegressor (scikit-learn) sobre um dataset expandido de
pares (x_mascarado, x_original) — várias máscaras aleatórias por
paciente, não uma reimplementação de treino com perda mascarada "pura"
(simplificação documentada, não escondida — ver `fit()`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["MaskedFeaturePredictor"]


class MaskedFeaturePredictor:
    """
    `mask_fraction`: fração das Features mascaradas por cópia de
    treino. `n_masks_per_patient`: quantas cópias com máscaras
    DIFERENTES gerar por paciente (mais cópias = mais variedade de
    padrões de máscara vistos, à custa de mais tempo de treino).
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        mask_fraction: float = 0.15,
        n_masks_per_patient: int = 20,
        max_iter: int = 2000,
        random_state: int = 42,
        alpha: float = 1e-3,
    ):
        self.hidden_dim = hidden_dim
        self.mask_fraction = mask_fraction
        self.n_masks_per_patient = n_masks_per_patient
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self._mlp = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._mask_fill_value = 0.0
        self.is_fitted = False
        self.feature_names_: list[str] = []

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def _unstandardize(self, X: np.ndarray) -> np.ndarray:
        return X * self._std + self._mean

    def _make_masked_dataset(self, X_std: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gera `n_masks_per_patient` cópias mascaradas de cada linha de X_std. Retorna (X_mascarado, X_original, mascara_booleana)."""
        n, d = X_std.shape
        n_total = n * self.n_masks_per_patient
        X_masked = np.tile(X_std, (self.n_masks_per_patient, 1)).copy()
        X_original = np.tile(X_std, (self.n_masks_per_patient, 1)).copy()
        mask = rng.random((n_total, d)) < self.mask_fraction
        X_masked[mask] = self._mask_fill_value
        return X_masked, X_original, mask

    def fit(self, space: "RepresentationSpace", order: Optional[Sequence[str]] = None) -> "MaskedFeaturePredictor":
        """
        `space`: o RepresentationSpace de UMA doença (Fase 10 completa
        exigiria combinar várias — ver README para a discussão honesta
        dessa limitação). Treina reconstruindo o vetor ORIGINAL completo
        a partir da versão mascarada — SIMPLIFICAÇÃO em relação ao MLM
        "puro" (que só penaliza erro nas posições mascaradas): aqui a
        perda de mínimos quadrados do `MLPRegressor` cobre todo o vetor,
        não só as posições mascaradas, porque `sklearn.MLPRegressor` não
        aceita uma função de perda com máscara por amostra. Avaliação
        (`masked_reconstruction_error`) SEMPRE mede erro só nas posições
        efetivamente mascaradas, então a MÉTRICA reportada é honesta,
        mesmo que o TREINO seja uma aproximação.
        """
        from sklearn.neural_network import MLPRegressor

        matrix, ids = space.matrix()
        used_order = order or space.order()
        if order is not None and list(order) != space.order():
            matrix = np.stack([space.get(sid).as_vector(used_order) for sid in ids])

        self.feature_names_ = []
        for domain_name in used_order:
            vec = space.get(ids[0])
            for f in vec.components.get(domain_name, []):
                self.feature_names_.append(f"{domain_name}.{f.name}")

        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0)
        self._std[self._std < 1e-9] = 1.0
        X_std = self._standardize(matrix)

        rng = np.random.default_rng(self.random_state)
        X_masked, X_original, _ = self._make_masked_dataset(X_std, rng)

        self._mlp = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim),
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        self._mlp.fit(X_masked, X_original)
        self.is_fitted = True
        return self

    def predict_masked(self, x: np.ndarray, masked_indices: Sequence[int]) -> np.ndarray:
        """
        Recebe um vetor `x` (escala ORIGINAL de X, não padronizada) com
        as posições em `masked_indices` desconhecidas (o valor ali é
        ignorado, sempre substituído pelo preenchimento de máscara antes
        de prever) — retorna o vetor completo RECONSTRUÍDO, também na
        escala original.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(space) deve ser chamado antes de predict_masked().")
        x_std = self._standardize(x.copy())
        x_std[list(masked_indices)] = self._mask_fill_value
        pred_std = self._mlp.predict(x_std.reshape(1, -1))[0]
        return self._unstandardize(pred_std)

    def masked_reconstruction_error(
        self, space: "RepresentationSpace", order: Optional[Sequence[str]] = None, mask_fraction: Optional[float] = None, seed: int = 0
    ) -> dict:
        """
        Avaliação HONESTA (só nas posições efetivamente mascaradas, ao
        contrário da perda de treino — ver docstring de `fit()`): erro
        quadrático médio de reconstrução por Feature, sobre `space`
        (pode ser dados nunca vistos no treino), mascarando
        `mask_fraction` (padrão: o mesmo usado em `fit()`) de cada
        paciente e comparando a reconstrução ao valor verdadeiro.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit(space) deve ser chamado antes de avaliar.")
        frac = mask_fraction if mask_fraction is not None else self.mask_fraction
        used_order = order or space.order()
        matrix, ids = space.matrix()
        if order is not None and list(order) != space.order():
            matrix = np.stack([space.get(sid).as_vector(used_order) for sid in ids])

        rng = np.random.default_rng(seed)
        n, d = matrix.shape
        erros_por_feature: dict[str, list[float]] = {name: [] for name in self.feature_names_}

        for i in range(n):
            n_mask = max(1, int(round(d * frac)))
            masked_idx = rng.choice(d, size=n_mask, replace=False)
            reconstruido = self.predict_masked(matrix[i], masked_idx)
            for idx in masked_idx:
                nome = self.feature_names_[idx] if idx < len(self.feature_names_) else f"feature_{idx}"
                erro = (reconstruido[idx] - matrix[i, idx]) ** 2
                erros_por_feature[nome].append(erro)

        mse_por_feature = {nome: float(np.mean(erros)) for nome, erros in erros_por_feature.items() if erros}
        return {
            "mse_por_feature": mse_por_feature,
            "mse_global": float(np.mean([e for erros in erros_por_feature.values() for e in erros])),
        }

    def describe(self) -> str:
        status = "ajustado" if self.is_fitted else "não ajustado"
        return f"MaskedFeaturePredictor(hidden_dim={self.hidden_dim}, mask_fraction={self.mask_fraction}, {status})"
