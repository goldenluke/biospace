"""
examples/11_masked_feature_prediction.py
===========================================

Fase 10 -- Foundation Model (PROTOTIPO DE ARQUITETURA, nao um foundation
model de verdade):

    Milhoes de pacientes -> BioSpace -> Foundation Model
    (treinado em estados fisiologicos, nao em texto)

Aqui: dezenas a centenas de pacientes (sinteticos), nao milhoes -- a
diferenca e de ORDEM DE GRANDEZA, nao de grau. O valor deste prototipo e
mostrar que a REPRESENTACAO (Features com proveniencia semantica) e um
substrato valido para pre-treino auto-supervisionado -- masked feature
prediction, o mesmo padrao do BERT (Devlin et al., 2019), mas sobre
Features fisiologicas em vez de palavras.

Rode com: python3 examples/11_masked_feature_prediction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import numpy as np

from biospace.core import Feature, RepresentationSpace, RepresentationVector
from biospace.foundation import MaskedFeaturePredictor


def build_correlated_space(n=300, seed=0):
    """f2=2*f1+ruido pequeno, f3=-f1+ruido pequeno (correlacionadas); f4,f5 ruido puro."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = 2 * f1 + rng.normal(0, 0.2, n)
    f3 = -f1 + rng.normal(0, 0.2, n)
    f4 = rng.normal(0, 1, n)
    f5 = rng.normal(0, 1, n)
    X = np.stack([f1, f2, f3, f4, f5], axis=1)

    space = RepresentationSpace()
    for i in range(n):
        vec = RepresentationVector(
            system_id=f"p{i}", timestamp=datetime(2024, 1, 1),
            components={"d": [Feature(name=f"f{j+1}", value=float(X[i, j])) for j in range(5)]},
        )
        space.add(vec)
    return space, X


def main():
    print("--- Masked Feature Prediction: estrutura de correlacao CONHECIDA ---")
    print("f2=2*f1 (correlacionada), f3=-f1 (correlacionada), f4/f5=ruido puro (sem relacao)\n")

    space, X = build_correlated_space()
    modelo = MaskedFeaturePredictor(hidden_dim=16, mask_fraction=0.2, n_masks_per_patient=30, max_iter=3000, random_state=0)
    modelo.fit(space)

    resultado = modelo.masked_reconstruction_error(space, mask_fraction=0.2, seed=1)

    print(f"{'Feature':10s} {'MSE reconstrucao':>18s} {'variancia (baseline)':>22s} {'razao':>8s}")
    for j, nome in enumerate(["d.f1", "d.f2", "d.f3", "d.f4", "d.f5"]):
        mse = resultado["mse_por_feature"][nome]
        var = np.var(X[:, j])
        print(f"{nome:10s} {mse:18.4f} {var:22.4f} {mse/var:8.3f}")

    print()
    print("f1/f2/f3 (correlacionadas): razao << 1 -- o modelo aprendeu a estrutura real")
    print("f4/f5 (ruido puro): razao ~ 1 -- o modelo NAO inventa estrutura onde nao ha\n")

    print("--- O mesmo modelo, sem NENHUMA alteracao, rodando sobre um plugin de doenca diferente ---")
    from biospace.plugins.diabetes import generate_synthetic_dataframe, load_from_dataframe

    df = generate_synthetic_dataframe(n_per_group=30, seed=42)
    cohort, representation = load_from_dataframe(df)
    space_diabetes = cohort.snapshot()

    modelo2 = MaskedFeaturePredictor(hidden_dim=16, mask_fraction=0.15, n_masks_per_patient=15, max_iter=1500, random_state=0)
    modelo2.fit(space_diabetes)
    resultado2 = modelo2.masked_reconstruction_error(space_diabetes, seed=1)
    print(f"Plugin de diabetes -- MSE global: {resultado2['mse_global']:.3f} (funciona sem alterar UMA linha do MaskedFeaturePredictor)")

    print()
    print("=== Aviso central (repetido de propósito) ===")
    print("Isto e uma prova de conceito ARQUITETURAL, nao um foundation model.")
    print("'Milhoes de pacientes' exigiria dados que este projeto nao tem e nao deveria fingir ter.")


if __name__ == "__main__":
    main()
