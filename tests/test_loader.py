"""
tests.test_loader
====================

Regressão do bug real mais impactante encontrado no projeto: a primeira
versão do loader criava um SleepSystem por LINHA da planilha. Nos dados
reais, isso transformou 1557 linhas (exames) em 1557 "pacientes" falsos,
quando na verdade eram só 355 pacientes únicos, 296 com múltiplos exames.
"""

from __future__ import annotations

from biospace.plugins.sleep import load_from_dataframe


def test_loader_groups_rows_by_patient_not_by_row(synthetic_dataframe):
    """6 pacientes esperados (3 com 1 exame, 3 com múltiplos) — nunca 1 sistema por linha."""
    cohort, representation = load_from_dataframe(synthetic_dataframe)

    n_rows = len(synthetic_dataframe)
    n_expected_patients = 6

    assert len(cohort) == n_expected_patients, (
        f"Esperava {n_expected_patients} pacientes (agrupados), mas veio {len(cohort)} — "
        f"se veio {n_rows} (o nº de linhas), o bug de 'um sistema por linha' voltou."
    )


def test_multi_exam_patients_have_correct_trajectory_length(synthetic_dataframe):
    """Pacientes com N exames devem ter uma trajetória de EXATAMENTE N pontos, não N trajetórias de 1 ponto."""
    cohort, representation = load_from_dataframe(synthetic_dataframe)

    lengths = sorted(len(traj) for traj in cohort.trajectories.values())
    assert lengths == [1, 1, 1, 3, 4, 5], f"Comprimentos de trajetória inesperados: {lengths}"


def test_single_system_id_per_patient_across_exams(synthetic_dataframe):
    """O mesmo paciente (múltiplos exames) deve corresponder a UM ÚNICO system_id na Cohort."""
    cohort, representation = load_from_dataframe(synthetic_dataframe)

    paciente_originais = [
        system.metadata.get("paciente_original") for system in cohort.systems.values()
    ]
    assert len(paciente_originais) == len(set(paciente_originais)), (
        "Encontrado mais de um system_id para o mesmo 'paciente_original' — "
        "o bug de duplicar sistemas por exame voltou."
    )


def test_trajectory_is_chronologically_ordered(synthetic_dataframe):
    """A trajetória de um paciente multi-exame deve estar ordenada por data, mesmo se a planilha não estiver."""
    shuffled = synthetic_dataframe.sample(frac=1, random_state=0).reset_index(drop=True)
    cohort, representation = load_from_dataframe(shuffled)

    for traj in cohort.trajectories.values():
        timestamps = [traj.at(i).timestamp for i in range(len(traj))]
        assert timestamps == sorted(timestamps), "Trajetória fora de ordem cronológica."


def test_blank_row_is_discarded(synthetic_dataframe):
    """Uma linha sem identificação de paciente (comum no fim de planilhas Excel reais) deve ser descartada, não virar um paciente fantasma."""
    import pandas as pd

    blank_row = {col: None for col in synthetic_dataframe.columns}
    df_with_blank = pd.concat([synthetic_dataframe, pd.DataFrame([blank_row])], ignore_index=True)

    cohort, representation = load_from_dataframe(df_with_blank)

    assert len(cohort) == 6, "A linha em branco não deveria ter virado um paciente."
    assert cohort.loader_report["n_rows_discarded"] == 1
