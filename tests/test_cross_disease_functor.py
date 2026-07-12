"""
tests.test_cross_disease_functor
====================================

`project_to_process_space` — o mapeamento de OBJETOS de um functor
categórico entre categorias de representação de doenças diferentes
(§4.17 do manuscrito, nomeado lá como "trabalho futuro, não realizado")
para uma categoria alvo compartilhada (vetores indexados por processo
fisiológico). Testa:

1. Bem-definição em duas fontes estruturalmente diferentes (sleep,
   metabolic), incluindo ausência parcial.
2. Uma propriedade do tipo naturalidade -- atualizar um domínio NÃO
   relacionado a um processo não deveria mudar a projeção desse
   processo, ecoando `check_domain_update_independence`.
3. Demonstração real: pacientes sleep e NHANES de verdade, projetados
   no processo compartilhado `cardiovascular_regulation`, comparáveis
   na mesma escala apesar de zero variáveis brutas em comum.

ESCOPO HONESTO, repetido aqui: isto testa o mapeamento de OBJETOS e
uma propriedade de naturalidade específica -- não verifica
exaustivamente que TODO morfismo admissível nas duas categorias fonte
preserva identidade/composição sob este functor, o que a definição
categórica completa exigiria.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from biospace.core import Observation, project_to_process_space


def _paciente_sleep_valores(**overrides):
    base = {
        "idade": 50.0, "peso_kg": 85.0, "altura_cm": 170.0, "imc": 29.0, "ido": 20.0, "ido_sono": 18.0,
        "no_de_dessaturacoes": 120.0, "tempo_total_de_ronco_min": 45.0, "tempo_em_ronco_baixo": 15.0,
        "tempo_em_ronco_medio": 20.0, "tempo_em_ronco_alto": 10.0, "spo2_minima": 85.0, "spo2_media": 93.0,
        "spo2_maxima": 97.0, "tempo_spo2_90": 10.0, "carga_hipoxica_min_h": 25.0, "no_de_eventos_de_hipoxemia": 90.0,
        "tempo_total_em_hipoxemia_min": 30.0, "tempo_para_dormir_min": 20.0, "tempo_total_de_sono_min": 380.0,
        "tempo_acordado_pos_sono_min": 25.0, "eficiencia_do_sono": 85.0, "fc_minima_bpm": 58.0, "fc_media_bpm": 72.0,
        "fc_maxima_bpm": 105.0, "doencas": "", "sintomas": "", "tratamentos": "",
    }
    base.update(overrides)
    return base


def test_projection_is_well_defined_for_sleep_representation():
    from biospace.plugins.sleep import SleepRepresentation, SleepSystem
    from biospace.plugins.sleep.builders import exam

    representation = SleepRepresentation()
    system = SleepSystem()
    system.observe(exam(_paciente_sleep_valores(), timestamp=datetime(2024, 1, 1)))
    vetor = representation.transform(system)

    projecao = project_to_process_space(representation, vetor)
    assert set(projecao.keys()) == {"cardiovascular_regulation"}, "Sleep so tem 1 processo declarado (deliberado -- ver test_physiological_process.py)."


def test_projection_is_well_defined_for_metabolic_representation():
    from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, exam

    representation = MetabolicRepresentation()
    system = MetabolicSystem()
    system.observe(exam({
        "hba1c_pct": 7.0, "glicemia_jejum_mg_dl": 130.0, "idade": 50, "sexo": 1.0,
        "imc": 28.0, "circunferencia_abdominal_cm": 95.0,
        "pressao_sistolica_mmhg": 125.0, "pressao_diastolica_mmhg": 80.0,
        "creatinina_mg_dl": 1.0, "taxa_filtracao_glomerular": 85.0,
        "colesterol_total_mg_dl": 190.0, "hdl_mg_dl": 50.0,
    }, timestamp=datetime(2024, 1, 1)))
    vetor = representation.transform(system)

    projecao = project_to_process_space(representation, vetor)
    assert set(projecao.keys()) == {"glucose_homeostasis", "body_composition", "cardiovascular_regulation", "renal_filtration", "lipid_metabolism"}


def test_projection_excludes_processes_with_no_non_missing_features():
    """Se TODAS as Features de um processo estao ausentes, o processo nao deveria aparecer na projecao -- nunca um 0.0 arbitrario."""
    from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, exam

    representation = MetabolicRepresentation()
    system = MetabolicSystem()
    system.observe(exam({"hba1c_pct": 7.0, "glicemia_jejum_mg_dl": 130.0}, timestamp=datetime(2024, 1, 1)))
    vetor = representation.transform(system)

    projecao = project_to_process_space(representation, vetor)
    assert "lipid_metabolism" not in projecao, "Sem NENHUM dado lipidico, o processo nao deveria aparecer."
    assert "glucose_homeostasis" in projecao


def test_naturality_unrelated_domain_update_does_not_change_process_projection():
    """
    O TESTE DECISIVO de naturalidade: atualizar um domínio NÃO
    relacionado a cardiovascular_regulation (ex.: glycemic) não deveria
    mudar a projeção desse processo -- ecoa diretamente
    check_domain_update_independence (core/contracts.py), agora
    verificado especificamente para o mapeamento de objetos do functor.
    """
    from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, exam

    representation = MetabolicRepresentation()
    system = MetabolicSystem()
    system.observe(exam({
        "hba1c_pct": 7.0, "glicemia_jejum_mg_dl": 130.0,
        "pressao_sistolica_mmhg": 125.0, "pressao_diastolica_mmhg": 80.0,
    }, timestamp=datetime(2024, 1, 1)))
    vetor_antes = representation.transform(system)
    projecao_antes = project_to_process_space(representation, vetor_antes)

    system.observe(Observation(timestamp=datetime(2024, 3, 1), source="exame_parcial", values={"hba1c_pct": 9.5}))
    vetor_depois = representation.transform(system)
    projecao_depois = project_to_process_space(representation, vetor_depois)

    assert projecao_depois["cardiovascular_regulation"] == pytest.approx(projecao_antes["cardiovascular_regulation"]), (
        "Atualizar glycemic (nao relacionado) nao deveria mudar a projecao de cardiovascular_regulation."
    )
    assert projecao_depois["glucose_homeostasis"] != pytest.approx(projecao_antes["glucose_homeostasis"]), (
        "Em contraste, glucose_homeostasis DEVERIA mudar -- prova que o teste discrimina, nao e' trivialmente sempre igual."
    )


@pytest.mark.skipif(
    not os.path.exists("/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx"),
    reason="Requer a coorte real de SAOS.",
)
def test_real_cross_disease_comparison_sleep_vs_nhanes():
    """
    A DEMONSTRAÇÃO real: um paciente SAOS real e um paciente NHANES
    real, projetados no MESMO processo compartilhado
    (cardiovascular_regulation), numa escala genuinamente comparável --
    apesar de zero variáveis brutas em comum entre as duas
    representações. O ponto não é que os valores devam ser parecidos
    (não há razão para esperar isso) -- é que a comparação é
    *sintaticamente possível de fazer* através do functor, o que não
    seria verdade comparando as representações brutas diretamente.
    """
    from biospace.plugins.sleep import load_from_excel

    cohort_sleep, representation_sleep = load_from_excel("/mnt/user-data/uploads/Exames_realizados_dados_anonimizados.xlsx", header=1)
    space_sleep = cohort_sleep.snapshot()
    sid_sleep = next(iter(space_sleep.ids()))
    projecao_sleep = project_to_process_space(representation_sleep, space_sleep.get(sid_sleep))

    from biospace.datasets.nhanes import NHANES_PREPANDEMIC_FILES, load_nhanes_metabolic_cohort
    from biospace.plugins.metabolic import load_from_dataframe

    df = load_nhanes_metabolic_cohort("/mnt/user-data/uploads", files=NHANES_PREPANDEMIC_FILES)
    df_adultos = df[df["idade"] >= 20].copy()
    cohort_nhanes, representation_nhanes = load_from_dataframe(df_adultos)
    space_nhanes = cohort_nhanes.snapshot()
    sid_nhanes = next(iter(cohort_nhanes.trajectories))
    projecao_nhanes = project_to_process_space(representation_nhanes, space_nhanes.get(sid_nhanes))

    assert "cardiovascular_regulation" in projecao_sleep
    assert "cardiovascular_regulation" in projecao_nhanes
    diferenca = abs(projecao_sleep["cardiovascular_regulation"] - projecao_nhanes["cardiovascular_regulation"])
    assert diferenca == diferenca  # confirma que nao e' NaN -- a comparacao produziu um numero real
