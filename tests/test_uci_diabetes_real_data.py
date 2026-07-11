"""
tests.test_uci_diabetes_real_data
=====================================

Testes contra o arquivo real diabetic_data.csv (Strack et al., 2014),
enviado pelo usuário — pytest.mark.skipif quando ausente (mesmo padrão
de test_nhanes_real_data.py). Estrutura genuinamente diferente do
NHANES: sem lab contínuo, sem antropometria, sem pressão -- domínios
próprios (utilization, glycemic_testing, medication_intensity),
documentados em `biospace/datasets/uci_diabetes.py`.
"""

from __future__ import annotations

import os

import pytest

CAMINHO_CSV = "/mnt/user-data/uploads/diabetic_data.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(CAMINHO_CSV), reason="Arquivo UCI Diabetes 130-US Hospitals não disponível neste ambiente.")


@pytest.fixture(scope="module")
def real_uci_cohort():
    """
    Representação com `include_diagnosis_category=False` -- preserva
    EXATAMENTE a representação original (3 domínios) usada para o
    achado publicado de fenótipo-readmissão. Ver
    `test_diagnosis_category_dilutes_readmission_association` para o
    achado, também real, de que incluir o 4º domínio muda o resultado.
    """
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    return load_uci_diabetes_cohort(CAMINHO_CSV, include_diagnosis_category=False)


def test_loads_expected_number_of_patients_and_encounters(real_uci_cohort):
    cohort, representation = real_uci_cohort
    assert cohort.loader_report["n_encontros"] == 101766
    assert cohort.loader_report["n_pacientes"] == 71518
    assert len(cohort.trajectories) == 71518


def test_multi_encounter_patients_produce_real_multi_point_trajectories(real_uci_cohort):
    """23% dos pacientes tem multiplos encontros -- confirma que ao menos um paciente com trajetoria longa (>=10 pontos) foi construido corretamente."""
    cohort, representation = real_uci_cohort
    tamanhos = [len(t) for t in cohort.trajectories.values()]
    assert max(tamanhos) >= 10, "Esperava pelo menos um paciente com trajetoria longa."
    n_multi_encontro = sum(1 for t in tamanhos if t >= 2)
    assert n_multi_encontro > 15000, "Esperava >15 mil pacientes com multiplos encontros (achado real: 23% da coorte)."


def test_utilization_domain_is_essentially_complete(real_uci_cohort):
    """As colunas de utilizacao sao 100% completas na fonte -- nenhuma Feature de utilization deveria estar ausente."""
    cohort, representation = real_uci_cohort
    space = cohort.snapshot()
    amostra_ids = list(space.ids())[:500]
    n_ausentes = 0
    for sid in amostra_ids:
        vetor = space.get(sid)
        n_ausentes += sum(1 for f in vetor.components["utilization"] if f.is_missing)
    assert n_ausentes == 0, "Utilization deveria ser 100% completo -- alguma Feature apareceu ausente."


def test_glycemic_testing_domain_is_mostly_missing_by_design(real_uci_cohort):
    """A1Cresult/max_glu_serum sao esparsos por desenho (83-95% ausentes na fonte)."""
    cohort, representation = real_uci_cohort
    space = cohort.snapshot()
    amostra_ids = list(space.ids())[:2000]
    total = 0
    ausentes = 0
    for sid in amostra_ids:
        vetor = space.get(sid)
        for f in vetor.components["glycemic_testing"]:
            total += 1
            if f.is_missing:
                ausentes += 1
    fracao_ausente = ausentes / total
    assert fracao_ausente > 0.5, f"Esperava alta fracao de ausencia (achado real: 83-95%), obteve {fracao_ausente:.1%}."


def test_derived_variable_computes_on_real_multi_encounter_trajectory(real_uci_cohort):
    """DerivedVariable generaliza para uma TERCEIRA fonte (sintetica -> NHANES transversal -> aqui, trajetoria real multi-ponto)."""
    from biospace.core import DerivedVariable, Feature
    import numpy as np

    class _NumMedicationsSlope(DerivedVariable):
        name = "num_medications_slope"
        domain_name = "utilization"
        feature_name = "num_medications"
        min_points = 2

        def compute(self, trajectory):
            pontos = self.series(trajectory)
            if len(pontos) < self.min_points:
                return None
            dias = np.array([p[0] for p in pontos])
            valores = np.array([p[1] for p in pontos])
            if np.ptp(dias) == 0:
                return None
            slope, _ = np.polyfit(dias, valores, 1)
            return Feature(name=self.name, value=float(slope), raw_value=float(slope))

    cohort, representation = real_uci_cohort
    sid_longo = max(cohort.trajectories, key=lambda s: len(cohort.trajectories[s]))
    traj = cohort.trajectories[sid_longo]
    assert len(traj) >= 10

    resultado = _NumMedicationsSlope().compute(traj)
    assert resultado is not None
    assert np.isfinite(resultado.value)


def test_phenotype_associates_with_real_readmission_outcome(real_uci_cohort):
    """
    O TESTE DECISIVO: fenotipo derivado SO de utilizacao/testes glicemicos/medicacao
    (sem idade, sem diagnostico) associa com readmissao real em 30 dias.
    ACHADO REAL documentado como regressao: o fenotipo de maior risco e
    caracterizado por alta utilizacao PREVIA, consistente com a
    literatura de predicao de readmissao.
    """
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = real_uci_cohort
    space = cohort.snapshot()
    order = representation.domain_names()

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space)

    taxas_readmissao_precoce = {}
    for ph in phenotypes:
        pacientes_do_fenotipo = [sid for sid in space.ids() if ph.contains(space.get(sid).as_vector(order))]
        if not pacientes_do_fenotipo:
            continue
        readmissoes = [cohort.systems[sid].observations[-1].metadata.get("readmitted") for sid in pacientes_do_fenotipo]
        taxa = sum(1 for r in readmissoes if r == "<30") / len(readmissoes)
        taxas_readmissao_precoce[ph.name] = taxa

    assert len(taxas_readmissao_precoce) >= 2, "Esperava pelo menos 2 fenotipos com membros."
    razao = max(taxas_readmissao_precoce.values()) / min(taxas_readmissao_precoce.values())
    assert razao > 1.5, (
        f"Esperava diferenca substancial de taxa de readmissao entre fenotipos (achado real: ~2x) -- obteve razao {razao:.2f}. "
        f"Taxas: {taxas_readmissao_precoce}"
    )


@pytest.fixture(scope="module")
def real_uci_cohort_com_diagnostico():
    """Representacao completa (4 dominios, include_diagnosis_category=True, o default) -- fixture separada porque muda o resultado de fenotipagem, nao deve ser confundida com real_uci_cohort."""
    from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

    return load_uci_diabetes_cohort(CAMINHO_CSV)


def test_diagnosis_category_dilutes_readmission_association(real_uci_cohort_com_diagnostico):
    """
    ACHADO REAL: adicionar DiagnosisCategoryDomain (9 flags binarias de
    categoria ICD-9) a fenotipagem MUDA o resultado -- dilui a
    associacao com readmissao que a representacao original (3
    dominios, sem diagnostico) mostrava (~2x). Com o 4o dominio, K-Means
    passa a separar principalmente por um grupo minusculo de internacao
    extremamente longa (achado documentado: ~212 pacientes, ~13 dias
    medios vs. ~4-5 dos demais), nao mais pela historia de utilizacao
    que predizia readmissao. Nenhuma das duas representacoes esta
    "errada" -- capturam estruturas diferentes; e' o mesmo argumento do
    artigo "Representation Before Inference" (mais uma variavel na
    representacao nao e' neutro), agora numa instancia real e nao
    planejada.
    """
    from biospace.phenotyping import KMeansPhenotyper

    cohort, representation = real_uci_cohort_com_diagnostico
    space = cohort.snapshot()
    order = representation.domain_names()
    assert "diagnosis_category" in order

    phenotyper = KMeansPhenotyper(n_clusters=4)
    phenotypes = phenotyper.fit(space)

    taxas = {}
    for ph in phenotypes:
        pacientes_do_fenotipo = [sid for sid in space.ids() if ph.contains(space.get(sid).as_vector(order))]
        if not pacientes_do_fenotipo:
            continue
        readmissoes = [cohort.systems[sid].observations[-1].metadata.get("readmitted") for sid in pacientes_do_fenotipo]
        taxas[ph.name] = sum(1 for r in readmissoes if r == "<30") / len(readmissoes)

    razao = max(taxas.values()) / min(taxas.values())
    assert razao < 1.8, (
        f"Esperava associacao DILUIDA (achado documentado: razao ~1.1-1.5, bem menor que os ~2.2x da representacao "
        f"sem diagnostico) -- obteve razao {razao:.2f}. Taxas: {taxas}. Se isto passou a dar >=1.8, a fenotipagem "
        f"deixou de diluir com este dominio incluido -- investigar o que mudou."
    )


def test_dynamics_are_globally_stable_on_real_multi_encounter_trajectories(real_uci_cohort):
    """
    ACHADO REAL: primeira vez que o modulo de dinamica (MeanRevertingEvolutionOperator)
    roda sobre trajetorias REAIS fora de sleep/sintetico. Ajustado sobre os
    16.773 pacientes com >=2 encontros -- todas as 13 Features (utilization,
    glycemic_testing, medication_intensity) resultam estaveis (|phi|<1).
    """
    from biospace.core import Cohort
    from biospace.dynamics import MeanRevertingEvolutionOperator, StabilityOperator

    cohort, representation = real_uci_cohort
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 2}
    assert len(multi) > 15000, "Esperava >15 mil pacientes multi-encontro (achado documentado: 16.773)."

    cohort_multi = Cohort()
    for sid, traj in multi.items():
        cohort_multi.systems[sid] = cohort.systems[sid]
        cohort_multi.trajectories[sid] = traj

    evo = MeanRevertingEvolutionOperator(order=order)
    evo.fit(cohort_multi)
    relatorio = StabilityOperator(evolution_operator=evo).analyze(cohort_multi)

    assert relatorio.is_globally_stable is True, "Esperava dinamica globalmente estavel (achado documentado)."
    assert relatorio.n_stable == relatorio.n_features == 13


def test_number_emergency_near_instability_is_robust_not_outlier_driven(real_uci_cohort):
    """
    ACHADO REAL, contrastando com o achado de SAOS
    (hypoxia.tempo_total_em_hipoxemia_min, que era 1 unico paciente
    outlier): aqui, 'utilization.number_emergency' fica perto do limiar
    de instabilidade (phi~0.98) mas a conclusao de estabilidade e'
    ROBUSTA a remocao de qualquer paciente individual -- consistente com
    reversao a media genuinamente lenta ("frequent flyers" de uso de
    emergencia), nao um artefato de amostra pequena.
    """
    from biospace.core import Cohort
    from biospace.dynamics import check_feature_stability_robustness

    cohort, representation = real_uci_cohort
    order = representation.domain_names()

    multi = {sid: t for sid, t in cohort.trajectories.items() if len(t) >= 2}
    cohort_multi = Cohort()
    for sid, traj in multi.items():
        cohort_multi.systems[sid] = cohort.systems[sid]
        cohort_multi.trajectories[sid] = traj

    relatorio = check_feature_stability_robustness(cohort_multi, "utilization.number_emergency", order=order, max_patients_tested=30)
    assert relatorio.is_stable_full is True
    assert relatorio.conclusion_is_robust is True, "Esperava conclusao robusta (achado documentado, diferente do caso de SAOS)."
