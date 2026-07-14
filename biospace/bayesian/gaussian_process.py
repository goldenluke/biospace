"""
biospace.bayesian.gaussian_process
======================================

Regressão por Processo Gaussiano (GP) sobre um `RepresentationSpace`
— envelope fino sobre `sklearn.gaussian_process`, mas com um propósito
específico além de "mais um regressor": a escolha de kernel NUM
Processo Gaussiano é uma escolha de geometria (uma noção de quão
"parecidos" dois pontos são, antes de qualquer dado ser observado) —
exatamente o argumento do Artigo IV desta série sobre a modelagem
normativa de Marquand et al. (2019), que usa Regressão por Processo
Gaussiano quase universalmente e nunca varia a escolha de kernel.

Este módulo permite testar essa hipótese diretamente: mesma
representação, mesmos dados, kernel variado -- a predição (e,
decisivamente, a INCERTEZA associada a ela) muda do mesmo jeito que a
geometria mudou a estrutura detectável no Artigo II?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from biospace.core import RepresentationSpace

__all__ = ["GPFitResult", "GaussianProcessOperator", "KERNEL_PRESETS"]


def _build_kernel_presets():
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, ExpSineSquared, Matern, WhiteKernel
    return {
        "rbf": ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        "matern": ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
        "linear": ConstantKernel(1.0) * DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1),
        "periodic": ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0) + WhiteKernel(noise_level=0.1),
    }


KERNEL_PRESETS = None  # populado sob demanda em GaussianProcessOperator.__init__, para nao importar sklearn.gaussian_process no import do modulo


@dataclass
class GPFitResult:
    mean: np.ndarray       # predicao pontual por amostra
    std: np.ndarray        # incerteza (desvio padrao) por amostra -- o que um regressor pontual nao da
    ids: list


class GaussianProcessOperator:
    """`kernel`: uma string de KERNEL_PRESETS ('rbf','matern','linear','periodic') ou um objeto Kernel do sklearn diretamente."""

    def __init__(self, kernel="rbf", n_restarts_optimizer: int = 3, random_state: int = 0):
        from sklearn.gaussian_process import GaussianProcessRegressor

        presets = _build_kernel_presets()
        kernel_obj = presets[kernel] if isinstance(kernel, str) else kernel
        self.kernel_name = kernel if isinstance(kernel, str) else repr(kernel)
        self.estimator = GaussianProcessRegressor(kernel=kernel_obj, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state, normalize_y=True)
        self._fitted = False

    def fit(self, space: "RepresentationSpace", targets: dict[str, float]) -> "GaussianProcessOperator":
        matrix, ids = space.matrix()
        y = np.array([targets[sid] for sid in ids])
        self.estimator.fit(matrix, y)
        self._fitted = True
        return self

    def predict(self, space: "RepresentationSpace") -> GPFitResult:
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")
        matrix, ids = space.matrix()
        mean, std = self.estimator.predict(matrix, return_std=True)
        return GPFitResult(mean=mean, std=std, ids=list(ids))

    def log_marginal_likelihood(self) -> float:
        """Quao bem o kernel (com hiperparametros otimizados) explica os dados de treino -- usado pra comparar kernels de forma objetiva, nao so visualmente."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de log_marginal_likelihood().")
        return float(self.estimator.log_marginal_likelihood(self.estimator.kernel_.theta))
