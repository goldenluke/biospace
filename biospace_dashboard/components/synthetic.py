"""
components.synthetic
=======================

Gera uma coorte sintética LONGITUDINAL — múltiplos exames por paciente,
ao longo de meses/anos, com:

  - Nível PRÓPRIO de cada paciente (traço individual, sorteado uma vez)
    + ruído exame-a-exame MENOR que a variação entre pacientes — para
    que os exames de um mesmo paciente pareçam de fato do mesmo paciente,
    não pontos independentes disfarçados de trajetória.
  - Nº de exames por paciente com distribuição parecida com a real
    (mediana baixa, cauda longa até ~17 — ver README do biospace).
  - Adoção de tratamento (AAM/CPAP) CORRELACIONADA com severidade —
    confundimento por indicação deliberado, o mesmo padrão encontrado na
    planilha real — com efeito de melhora gradual pós-adoção nas
    Features de apneia/hipoxemia.
  - Idade avança de verdade entre exames do mesmo paciente; comorbidades
    só se acumulam, nunca desaparecem.

Isso importa porque boa parte do dashboard (Sobrevivência, Early
Warning, Sistemas Dinâmicos, Inferência Causal, DTW/Gromov-Wasserstein
em Geometrias) depende de trajetórias com múltiplos pontos — sem isso, o
modo de demonstração ficava com essas páginas vazias ou sem sentido.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from biospace.plugins.sleep import MAPA_DOENCAS, MAPA_SINTOMAS

_PROFILES = {
    "leve": dict(
        idade=(38, 8), peso_kg=(75, 10), altura_cm=(168, 8), imc=(24, 2),
        ido=(6, 2), ido_sono=(6, 2), no_de_dessaturacoes=(40, 15),
        tempo_total_de_ronco_min=(20, 10), tempo_em_ronco_baixo=(12, 5),
        tempo_em_ronco_medio=(6, 3), tempo_em_ronco_alto=(2, 2),
        spo2_minima=(90, 2), spo2_media=(96, 1), spo2_maxima=(98, 1),
        tempo_spo2_90=(2, 1), carga_hipoxica_min_h=(8, 3),
        no_de_eventos_de_hipoxemia=(30, 15), tempo_total_em_hipoxemia_min=(10, 5),
        tempo_para_dormir_min=(15, 6), tempo_total_de_sono_min=(400, 40),
        tempo_acordado_pos_sono_min=(15, 8), eficiencia_do_sono=(90, 4),
        fc_minima_bpm=(52, 5), fc_media_bpm=(65, 6), fc_maxima_bpm=(95, 10),
    ),
    "moderada": dict(
        idade=(48, 9), peso_kg=(90, 12), altura_cm=(168, 8), imc=(29, 3),
        ido=(20, 4), ido_sono=(20, 4), no_de_dessaturacoes=(140, 30),
        tempo_total_de_ronco_min=(70, 20), tempo_em_ronco_baixo=(25, 8),
        tempo_em_ronco_medio=(30, 10), tempo_em_ronco_alto=(15, 6),
        spo2_minima=(84, 3), spo2_media=(93, 2), spo2_maxima=(97, 1),
        tempo_spo2_90=(12, 5), carga_hipoxica_min_h=(28, 8),
        no_de_eventos_de_hipoxemia=(110, 30), tempo_total_em_hipoxemia_min=(35, 12),
        tempo_para_dormir_min=(22, 8), tempo_total_de_sono_min=(370, 45),
        tempo_acordado_pos_sono_min=(30, 12), eficiencia_do_sono=(83, 5),
        fc_minima_bpm=(58, 6), fc_media_bpm=(74, 7), fc_maxima_bpm=(112, 12),
    ),
    "grave": dict(
        idade=(56, 10), peso_kg=(105, 15), altura_cm=(168, 8), imc=(36, 4),
        ido=(45, 8), ido_sono=(45, 8), no_de_dessaturacoes=(280, 50),
        tempo_total_de_ronco_min=(140, 30), tempo_em_ronco_baixo=(30, 10),
        tempo_em_ronco_medio=(50, 12), tempo_em_ronco_alto=(50, 15),
        spo2_minima=(74, 5), spo2_media=(88, 3), spo2_maxima=(95, 2),
        tempo_spo2_90=(28, 8), carga_hipoxica_min_h=(55, 12),
        no_de_eventos_de_hipoxemia=(220, 40), tempo_total_em_hipoxemia_min=(70, 15),
        tempo_para_dormir_min=(35, 12), tempo_total_de_sono_min=(340, 50),
        tempo_acordado_pos_sono_min=(55, 18), eficiencia_do_sono=(72, 6),
        fc_minima_bpm=(65, 7), fc_media_bpm=(82, 8), fc_maxima_bpm=(130, 14),
    ),
}

# Desvio padrão de exame-para-exame, como FRAÇÃO do desvio padrão do
# grupo — bem menor que a variação ENTRE pacientes, de propósito.
_WITHIN_PATIENT_NOISE_FRACTION = 0.15

# Features que melhoram gradualmente após adoção de tratamento (multiplicador < 1)
_FEATURES_QUE_MELHORAM_COM_TRATAMENTO = (
    "ido", "ido_sono", "no_de_dessaturacoes", "tempo_total_de_ronco_min",
    "carga_hipoxica_min_h", "tempo_spo2_90", "tempo_total_em_hipoxemia_min", "no_de_eventos_de_hipoxemia",
)
# Features que pioram numericamente quando o quadro melhora (multiplicador > 1: SpO2 sobe, eficiência sobe)
_FEATURES_QUE_SOBEM_COM_TRATAMENTO = ("spo2_minima", "spo2_media", "eficiencia_do_sono")

_PROB_TRATAMENTO_POR_SEVERIDADE = {"leve": 0.05, "moderada": 0.25, "grave": 0.55}
_N_DOENCAS_BASE_POR_SEVERIDADE = {"leve": 0, "moderada": 1, "grave": 2}
_N_SINTOMAS_POR_SEVERIDADE = {"leve": 1, "moderada": 2, "grave": 4}


def _sample_n_exams(rng: np.random.Generator) -> int:
    """Distribuição de nº de exames por paciente — parecida com a real (mediana baixa, cauda longa até ~17)."""
    r = rng.random()
    if r < 0.30:
        return 1
    if r < 0.70:
        return int(rng.integers(2, 5))
    if r < 0.90:
        return int(rng.integers(5, 9))
    return int(rng.integers(9, 18))


def _clip(key: str, value: float) -> float:
    if key in ("spo2_minima", "spo2_media", "spo2_maxima", "eficiencia_do_sono"):
        return min(100.0, max(0.0, value))
    return max(0.0, value)


def generate_synthetic_dataframe(n_per_group: int = 30, seed: int = 42, missingness: bool = True) -> pd.DataFrame:
    """
    Gera um DataFrame sintético LONGITUDINAL com as mesmas colunas/regras
    clínicas reais — ver docstring do módulo para o que isso cobre.
    """
    rng = np.random.default_rng(seed)
    severities = ["leve"] * n_per_group + ["moderada"] * n_per_group + ["grave"] * n_per_group
    rng.shuffle(severities)

    rows: list[dict] = []
    t_start_base = datetime(2019, 1, 1)

    for i, severity in enumerate(severities):
        profile = _PROFILES[severity]
        patient_id = f"DEMO_{i:04d}"
        genero = rng.choice(["Masculino", "Feminino"])

        # Traço individual do paciente — sorteado UMA VEZ, em torno da média do grupo.
        patient_level = {key: rng.normal(mean, std) for key, (mean, std) in profile.items()}

        n_exams = _sample_n_exams(rng)
        first_date = t_start_base + timedelta(days=int(rng.integers(0, 365 * 5)))
        exam_dates = [first_date]
        for _ in range(n_exams - 1):
            exam_dates.append(exam_dates[-1] + timedelta(days=int(rng.integers(20, 400))))

        # Confundimento por indicação deliberado: adoção de tratamento
        # correlacionada com severidade, num ponto aleatório da trajetória.
        adota_tratamento = rng.random() < _PROB_TRATAMENTO_POR_SEVERIDADE[severity]
        exame_de_adocao = int(rng.integers(1, n_exams)) if (adota_tratamento and n_exams > 1) else None
        tipo_tratamento = rng.choice(["Aparelho de avanço mandibular", "CPAP"], p=[0.6, 0.4]) if adota_tratamento else None

        doencas_paciente = list(
            rng.choice(list(MAPA_DOENCAS.keys()), size=_N_DOENCAS_BASE_POR_SEVERIDADE[severity], replace=False)
        ) if _N_DOENCAS_BASE_POR_SEVERIDADE[severity] else []

        for exam_idx, exam_date in enumerate(exam_dates):
            tratado = exame_de_adocao is not None and exam_idx >= exame_de_adocao
            exams_desde_tratamento = (exam_idx - exame_de_adocao) if tratado else 0

            row: dict = {}
            for key, (_, std) in profile.items():
                v = patient_level[key] + rng.normal(0, std * _WITHIN_PATIENT_NOISE_FRACTION)

                if tratado and key in _FEATURES_QUE_MELHORAM_COM_TRATAMENTO:
                    fracao = min(0.5, 0.15 * (exams_desde_tratamento + 1))
                    v *= 1 - fracao
                elif tratado and key in _FEATURES_QUE_SOBEM_COM_TRATAMENTO:
                    fracao = min(0.15, 0.03 * (exams_desde_tratamento + 1))
                    v *= 1 + fracao

                row[key] = _clip(key, v)

            # Idade avança de verdade entre exames do mesmo paciente.
            anos_passados = (exam_date - exam_dates[0]).days / 365.25
            row["idade"] = patient_level["idade"] + anos_passados

            row["status"] = "Concluído"
            row["paciente"] = patient_id
            row["inicio"] = exam_date
            row["genero"] = genero

            # Comorbidades só se acumulam (nunca desaparecem) — pequena chance por exame de acompanhamento.
            if exam_idx > 0 and rng.random() < 0.05:
                candidatos = [d for d in MAPA_DOENCAS if d not in doencas_paciente]
                if candidatos:
                    doencas_paciente.append(rng.choice(candidatos))
            row["doencas"] = ", ".join(doencas_paciente)
            row["condicoes"] = row["doencas"]

            sintomas = rng.choice(list(MAPA_SINTOMAS.keys()), size=_N_SINTOMAS_POR_SEVERIDADE[severity], replace=False)
            row["sintomas"] = ", ".join(sintomas)

            row["tratamentos"] = tipo_tratamento if tratado else ""
            row["medicamentos"] = ""
            row["fa_resultado"] = "Negativo"
            rows.append(row)

    df = pd.DataFrame(rows)

    if missingness:
        # Mesma ausência estrutural e heterogênea observada nos dados reais
        # (ver README do biospace) — para que a demonstração seja honesta
        # sobre essa característica do problema, não só sobre a longitudinalidade.
        _apply_missingness(df, "ido_sono", rng, frac=0.45)
        _apply_missingness(df, "carga_hipoxica_min_h", rng, frac=0.40)
        _apply_missingness(df, "no_de_eventos_de_hipoxemia", rng, frac=0.70)
        _apply_missingness(df, "tempo_total_em_hipoxemia_min", rng, frac=0.70)
        for col in ["tempo_para_dormir_min", "tempo_total_de_sono_min", "tempo_acordado_pos_sono_min", "eficiencia_do_sono"]:
            _apply_missingness(df, col, rng, frac=0.26, shared_mask_seed=99)

    return df


def _apply_missingness(df: pd.DataFrame, col: str, rng: np.random.Generator, frac: float, shared_mask_seed: int | None = None) -> None:
    if shared_mask_seed is not None:
        local_rng = np.random.default_rng(shared_mask_seed)
        mask = local_rng.random(len(df)) < frac
    else:
        mask = rng.random(len(df)) < frac
    df.loc[mask, col] = np.nan
