"""
examples/05_diabetes_toy_disease.py
======================================

Segundo plugin de doença completo — diabetes tipo 2, inteiramente
sintético (nenhum dado de paciente real), com a mesma disciplina
arquitetural do plugin sleep: 6 domínios semânticos, gerador
longitudinal realista, domínio latente com hipótese declarada, e
validação contra os contratos formais + fenotipagem + inferência causal
sem NENHUMA alteração no núcleo.

Rode com: python3 examples/05_diabetes_toy_disease.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from biospace.causal import check_baseline_balance
from biospace.core.contracts import check_reproducibility, check_temporality
from biospace.phenotyping import KMeansPhenotyper
from biospace.plugins.diabetes import (
    AnthropometricDomain,
    DiabetesSystem,
    GlycemicDomain,
    InsulinResistanceProxyDomain,
    generate_synthetic_dataframe,
    load_from_dataframe,
)


def main():
    print("--- Gerando coorte sintética longitudinal (90 pacientes) ---")
    df = generate_synthetic_dataframe(n_per_group=30, seed=42)
    cohort, representation = load_from_dataframe(df)
    order = representation.domain_names()
    print(f"{len(cohort)} pacientes, domínios: {order}")
    n_multi = sum(1 for t in cohort.trajectories.values() if len(t) >= 2)
    print(f"{n_multi} pacientes com >=2 exames\n")

    print("--- Contratos formais ---")
    glycemic_domain = next(d for d in representation.domains if d.name == "glycemic")
    system = next(iter(cohort.systems.values()))
    print("Reprodutibilidade:", check_reproducibility(glycemic_domain, system))
    report_temp = check_temporality(representation, lambda: DiabetesSystem(), system.observations)
    print("Temporalidade:", report_temp.is_compliant, "\n")

    print("--- Fenotipagem (genérica, sem código específico de diabetes) ---")
    space = cohort.snapshot()
    phenotypes = KMeansPhenotyper(n_clusters=3).fit(space)
    print(f"{len(phenotypes)} fenótipos encontrados\n")

    print("--- Domínio latente: proxy de resistência à insulina ---")
    anthro_domain = next(d for d in representation.domains if d.name == "anthropometric")
    proxy = InsulinResistanceProxyDomain(glycemic_domain, anthro_domain)
    proxy.fit(cohort)
    print(f"is_validated = {proxy.is_validated} (hipótese, não fato — ver hypothesis declarada)")
    for nome, carga in proxy.top_loadings(n=5):
        print(f"  {nome}: {carga:+.3f}")
    print()

    print("--- Achado de rigor: declínio renal correlacionado com exposição glicêmica crônica ---")
    pares = []
    for traj in cohort.trajectories.values():
        if len(traj) < 3:
            continue
        hba1c_medios = [
            f.raw_value
            for pt in [traj.at(i) for i in range(len(traj))]
            for f in pt.components["glycemic"]
            if f.name == "hba1c_pct" and f.raw_value is not None
        ]
        egfr_i = next((f.raw_value for f in traj.at(0).components["renal"] if f.name == "taxa_filtracao_glomerular"), None)
        egfr_f = next((f.raw_value for f in traj.at(-1).components["renal"] if f.name == "taxa_filtracao_glomerular"), None)
        if egfr_i is not None and egfr_f is not None and hba1c_medios:
            pares.append((float(np.mean(hba1c_medios)), egfr_i - egfr_f))
    hba1c_medios, quedas = zip(*pares)
    corr = np.corrcoef(hba1c_medios, quedas)[0, 1]
    print(f"Correlação HbA1c médio × queda de eGFR: {corr:+.3f} (esperado positivo)\n")

    print("--- Inferência causal: confundimento por indicação (adoção de insulina) ---")
    balance = check_baseline_balance(cohort, "treatment", "insulina", order=order)
    print(balance.summary())


if __name__ == "__main__":
    main()
