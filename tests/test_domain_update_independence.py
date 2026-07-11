"""
tests.test_domain_update_independence
=========================================

check_domain_update_independence: R(t) -> Observation -> R(t+1) deve
atualizar só os domínios que a nova observação realmente toca —
domínios independentes não sofrem alteração. Conecta diretamente com o
bug real do parâmetro `as_of` já documentado (uma teoria de
representação que produz testes formais capazes de detectar erros
reais de implementação).

TESTE DECISIVO: confirma tanto que domínios NÃO relacionados ficam
intocados quanto que o domínio de fato afetado MUDA — provando que o
contrato discrimina de verdade, não que está sempre trivialmente OK.
"""

from __future__ import annotations

from datetime import datetime

from biospace.core import Observation
from biospace.core.contracts import check_domain_update_independence
from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, exam


def _make_system():
    system = MetabolicSystem()
    system.observe(
        exam(
            {
                "hba1c_pct": 7.0,
                "glicemia_jejum_mg_dl": 130.0,
                "idade": 50,
                "imc": 28.0,
                "circunferencia_abdominal_cm": 95.0,
                "pressao_sistolica_mmhg": 125.0,
                "pressao_diastolica_mmhg": 80.0,
                "fc_repouso_bpm": 72.0,
                "creatinina_mg_dl": 1.0,
                "taxa_filtracao_glomerular": 85.0,
            },
            timestamp=datetime(2024, 1, 1),
        )
    )
    return system


def test_unrelated_domains_stay_unchanged():
    representation = MetabolicRepresentation()
    system = _make_system()
    obs_renal_apenas = Observation(timestamp=datetime(2024, 3, 1), source="exame_parcial", values={"creatinina_mg_dl": 1.3})

    resultado = check_domain_update_independence(
        system, representation, obs_renal_apenas,
        protected_domain_names=["glycemic", "anthropometric", "cardiovascular", "comorbidity", "treatment"],
    )
    assert all(r.is_independent for r in resultado.values()), [r.summary() for r in resultado.values() if not r.is_independent]


def test_the_actually_affected_domain_does_change():
    """
    O TESTE DECISIVO: o dominio 'renal' (tocado pela nova observacao)
    DEVE mudar -- se o contrato reportasse 'independente' mesmo para o
    dominio afetado, ele nao estaria testando nada de verdade, so
    retornando True sempre.
    """
    representation = MetabolicRepresentation()
    system = _make_system()
    obs_renal_apenas = Observation(timestamp=datetime(2024, 3, 1), source="exame_parcial", values={"creatinina_mg_dl": 2.5})

    resultado = check_domain_update_independence(
        system, representation, obs_renal_apenas, protected_domain_names=["renal"],
    )
    assert resultado["renal"].is_independent is False, "O dominio renal deveria ter mudado -- a observacao alterou creatinina_mg_dl diretamente."
    assert "creatinina_mg_dl" in resultado["renal"].changed_feature_names


def test_observation_with_same_value_as_before_does_not_count_as_changed():
    """Reobservar o MESMO valor nao deveria contar como mudanca -- o contrato compara VALORES, nao apenas 'houve uma nova observacao'."""
    representation = MetabolicRepresentation()
    system = _make_system()
    obs_mesmo_valor = Observation(timestamp=datetime(2024, 3, 1), source="exame_parcial", values={"creatinina_mg_dl": 1.0})

    resultado = check_domain_update_independence(system, representation, obs_mesmo_valor, protected_domain_names=["renal"])
    assert resultado["renal"].is_independent is True


def test_independence_holds_across_all_six_metabolic_domains():
    """Varredura: tocar CADA dominio, um de cada vez, confirmando que so ele muda -- nao um teste isolado, uma varredura sistematica."""
    todos_dominios = ["glycemic", "anthropometric", "cardiovascular", "renal", "comorbidity", "treatment"]
    observacoes_por_dominio = {
        "glycemic": {"hba1c_pct": 9.0},
        "anthropometric": {"imc": 35.0},
        "cardiovascular": {"pressao_sistolica_mmhg": 160.0},
        "renal": {"creatinina_mg_dl": 2.0},
        "comorbidity": {"hipertensao": 1},
        "treatment": {"metformina": 1},
    }

    for dominio_tocado, valores in observacoes_por_dominio.items():
        representation = MetabolicRepresentation()
        system = _make_system()
        obs = Observation(timestamp=datetime(2024, 3, 1), source="exame_parcial", values=valores)

        outros_dominios = [d for d in todos_dominios if d != dominio_tocado]
        resultado = check_domain_update_independence(system, representation, obs, protected_domain_names=outros_dominios)

        nao_independentes = [nome for nome, r in resultado.items() if not r.is_independent]
        assert not nao_independentes, f"Tocando '{dominio_tocado}', estes dominios NAO deveriam ter mudado mas mudaram: {nao_independentes}"
