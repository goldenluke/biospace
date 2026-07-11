"""
examples/12_physiological_processes.py
==========================================

PhysiologicalProcess: camada OPCIONAL e ADITIVA entre Observable e
SemanticDomain -- nomeia o mecanismo biologico que uma grandeza mede,
independente de qual dominio a consome. Motivacao levantada em revisao
externa: "diabetes nao e' um conjunto de exames, e' um conjunto de
processos biologicos -- as observacoes medem esses processos".

Retrocompatibilidade por construcao: plugin sleep nunca foi tocado por
esta camada (Observable.process default = None) e continua funcionando
identico. Plugin metabolic anota 4 processos reais como exemplo
concreto.

Rode com: python3 examples/12_physiological_processes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

from biospace.plugins.metabolic import ALL_PROCESSES, MetabolicRepresentation, MetabolicSystem, exam
from biospace.plugins.sleep import SleepRepresentation


def main():
    print("--- Retrocompatibilidade: plugin sleep nunca usou esta camada ---")
    sleep_representation = SleepRepresentation()
    print(f"SleepRepresentation.processes() = {sleep_representation.processes()}  (vazio -- nunca foi tocado)")
    print()

    print("--- Plugin metabolic: 4 processos fisiologicos reais declarados ---")
    for p in ALL_PROCESSES:
        print(f"  {p.name}: {p.description}")
    print()

    representation = MetabolicRepresentation()
    print(f"MetabolicRepresentation.processes() = {representation.processes()}")
    print()

    print("--- Consulta cruzando fronteira de dominio: features_by_process() ---")
    system = MetabolicSystem()
    system.observe(exam({
        "hba1c_pct": 7.5, "glicemia_jejum_mg_dl": 145.0, "idade": 55,
        "imc": 31.0, "circunferencia_abdominal_cm": 105.0,
        "pressao_sistolica_mmhg": 135.0, "pressao_diastolica_mmhg": 88.0, "fc_repouso_bpm": 78.0,
        "creatinina_mg_dl": 1.1, "taxa_filtracao_glomerular": 75.0,
    }, timestamp=datetime(2024, 1, 1)))
    vector = representation.transform(system)

    agrupado = representation.features_by_process(vector)
    for processo, features in agrupado.items():
        nomes = [f"{f.name}={f.raw_value}" for f in features]
        print(f"  {processo}: {nomes}")

    print()
    print("'idade' nao aparece em nenhum grupo (deliberado -- e' covariavel, nao mede um processo especifico).")
    todos_nomes = {f.name for features in agrupado.values() for f in features}
    assert "idade" not in todos_nomes
    print("Confirmado.")


if __name__ == "__main__":
    main()
