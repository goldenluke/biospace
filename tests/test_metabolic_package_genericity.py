"""
tests.test_metabolic_package_genericity
==========================================

A prova de que `biospace.plugins.metabolic` não é "diabetes disfarçado":
a MESMA MetabolicRepresentation, sem nenhuma alteração, sustenta duas
interpretações clínicas independentes -- diabetes (critério glicêmico,
ADA) e síndrome metabólica (critério composto, adaptado do NCEP ATP
III) -- aplicadas como funções PURAS sobre um vetor já representado
(N(R(B))), nunca como propriedades da representação em si (R).

TESTE DECISIVO: constrói pacientes com perfis clínicos KNOWN (diabetes
sem síndrome metabólica; síndrome metabólica sem diabetes; as duas;
nenhuma das duas) e confirma que as duas interpretações discordam nos
casos em que deveriam discordar -- se sempre concordassem, seria
evidência de que "síndrome metabólica" é só "diabetes" com outro nome,
não uma segunda lente genuína.
"""

from __future__ import annotations

from datetime import datetime

from biospace.plugins.diabetes import DiabetesRepresentation, DiabetesSystem
from biospace.plugins.metabolic import (
    MetabolicRepresentation,
    MetabolicSystem,
    classify_diabetes_status,
    classify_metabolic_syndrome_risk,
    exam,
)


def _make_patient(system_cls, values: dict) -> "any":
    system = system_cls()
    system.observe(exam(values, timestamp=datetime(2024, 1, 1)))
    return system


def test_diabetes_and_metabolic_syndrome_are_independent_lenses_over_same_representation():
    """O TESTE DECISIVO: 4 perfis clínicos conhecidos, as duas interpretações devem discordar exatamente onde deveriam."""
    representation = MetabolicRepresentation()

    perfis = {
        "diabetes_sem_sindrome": {
            # HbA1c alto (diabetes), mas sem adiposidade/pressao/glicemia elevadas o suficiente para sindrome
            "hba1c_pct": 7.5, "glicemia_jejum_mg_dl": 145.0,
            "circunferencia_abdominal_cm": 85.0, "imc": 23.0,
            "pressao_sistolica_mmhg": 115.0, "pressao_diastolica_mmhg": 75.0,
        },
        "sindrome_sem_diabetes": {
            # HbA1c normal (sem diabetes), mas adiposidade + pressao elevadas (sindrome)
            "hba1c_pct": 5.2, "glicemia_jejum_mg_dl": 90.0,
            "circunferencia_abdominal_cm": 110.0, "imc": 33.0,
            "pressao_sistolica_mmhg": 138.0, "pressao_diastolica_mmhg": 90.0,
        },
        "as_duas": {
            "hba1c_pct": 8.5, "glicemia_jejum_mg_dl": 180.0,
            "circunferencia_abdominal_cm": 115.0, "imc": 34.0,
            "pressao_sistolica_mmhg": 145.0, "pressao_diastolica_mmhg": 92.0,
        },
        "nenhuma_das_duas": {
            "hba1c_pct": 5.0, "glicemia_jejum_mg_dl": 85.0,
            "circunferencia_abdominal_cm": 80.0, "imc": 22.0,
            "pressao_sistolica_mmhg": 110.0, "pressao_diastolica_mmhg": 70.0,
        },
    }

    resultados = {}
    for nome, valores in perfis.items():
        system = _make_patient(MetabolicSystem, valores)
        vector = representation.transform(system)
        resultados[nome] = {
            "diabetes": classify_diabetes_status(vector),
            "sindrome": classify_metabolic_syndrome_risk(vector)["risco_elevado"],
        }

    assert resultados["diabetes_sem_sindrome"]["diabetes"] == "diabetes"
    assert resultados["diabetes_sem_sindrome"]["sindrome"] is False, "Este perfil foi desenhado para ter diabetes SEM sindrome metabolica -- se desse True, as duas interpretacoes nao seriam independentes."

    assert resultados["sindrome_sem_diabetes"]["diabetes"] != "diabetes"
    assert resultados["sindrome_sem_diabetes"]["sindrome"] is True, "Este perfil foi desenhado para ter sindrome metabolica SEM diabetes -- se desse False, a interpretacao de sindrome nao estaria funcionando."

    assert resultados["as_duas"]["diabetes"] == "diabetes"
    assert resultados["as_duas"]["sindrome"] is True

    assert resultados["nenhuma_das_duas"]["diabetes"] == "normal"
    assert resultados["nenhuma_das_duas"]["sindrome"] is False


def test_interpretation_never_modifies_the_representation():
    """As interpretacoes sao funcoes PURAS -- chamar as duas nao pode alterar o vetor nem a representacao subjacente."""
    representation = MetabolicRepresentation()
    system = _make_patient(MetabolicSystem, {"hba1c_pct": 7.0, "glicemia_jejum_mg_dl": 130.0})
    vector = representation.transform(system)

    vetor_antes = vector.as_vector(representation.domain_names()).copy()
    classify_diabetes_status(vector)
    classify_metabolic_syndrome_risk(vector)
    vetor_depois = vector.as_vector(representation.domain_names())

    assert (vetor_antes == vetor_depois).all(), "Interpretacoes clinicas nao podem alterar a representacao -- N(R(B)) le R(B), nunca escreve nela."


def test_diabetes_package_is_a_thin_compatibility_layer_not_a_copy():
    """DiabetesSystem/DiabetesRepresentation devem ser literalmente MetabolicSystem/MetabolicRepresentation, nao copias -- prova de que nao existe mais um R proprio para diabetes."""
    assert DiabetesSystem is MetabolicSystem
    assert DiabetesRepresentation is MetabolicRepresentation


def test_metabolic_syndrome_criteria_are_individually_auditable():
    """O criterio de sindrome metabolica deve expor CADA criterio avaliado, nao so um booleano -- para permanecer auditavel."""
    representation = MetabolicRepresentation()
    system = _make_patient(MetabolicSystem, {
        "circunferencia_abdominal_cm": 110.0, "pressao_sistolica_mmhg": 140.0,
        "pressao_diastolica_mmhg": 90.0, "glicemia_jejum_mg_dl": 95.0, "imc": 28.0,
    })
    vector = representation.transform(system)
    resultado = classify_metabolic_syndrome_risk(vector)

    assert "criterios" in resultado
    assert set(resultado["criterios"].keys()) == {"adiposidade_central", "imc_elevado", "pressao_elevada", "glicemia_elevada"}
    assert resultado["criterios"]["adiposidade_central"] is True
    assert resultado["criterios"]["pressao_elevada"] is True
    assert resultado["criterios"]["glicemia_elevada"] is False
    assert "adaptacao_reconhecida" in resultado, "A adaptacao do criterio clinico original deve ser reconhecida explicitamente no retorno, nao escondida."
