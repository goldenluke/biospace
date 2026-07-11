"""
biospace.plugins.diabetes.synthetic
======================================

Gerador de coorte sintética LONGITUDINAL para diabetes tipo 2 — mesma
disciplina do gerador usado no dashboard de sleep
(`biospace_dashboard/components/synthetic.py`): nível individual do
paciente + ruído exame-a-exame menor que a variação entre pacientes,
nº de exames com distribuição realista, adoção de tratamento
correlacionada com severidade (confundimento por indicação deliberado),
e uma mecânica adicional específica de diabetes — **declínio renal lento
e correlacionado com controle glicêmico crônico ruim**, não apenas com a
severidade basal (hiperglicemia crônica danifica os rins — mecanismo
real, não inventado).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_PROFILES = {
    "controlado": dict(
        idade=(50, 10), imc=(27, 3), circunferencia_abdominal_cm=(92, 8),
        glicemia_jejum_mg_dl=(105, 10), hba1c_pct=(6.3, 0.4),
        pressao_sistolica_mmhg=(125, 10), pressao_diastolica_mmhg=(80, 6), fc_repouso_bpm=(72, 6),
        creatinina_mg_dl=(0.9, 0.15), taxa_filtracao_glomerular=(90, 10),
    ),
    "moderado": dict(
        idade=(58, 10), imc=(31, 4), circunferencia_abdominal_cm=(102, 9),
        glicemia_jejum_mg_dl=(155, 20), hba1c_pct=(7.8, 0.6),
        pressao_sistolica_mmhg=(138, 12), pressao_diastolica_mmhg=(86, 8), fc_repouso_bpm=(76, 7),
        creatinina_mg_dl=(1.1, 0.2), taxa_filtracao_glomerular=(75, 12),
    ),
    "descompensado": dict(
        idade=(63, 11), imc=(35, 5), circunferencia_abdominal_cm=(112, 10),
        glicemia_jejum_mg_dl=(250, 40), hba1c_pct=(10.2, 1.2),
        pressao_sistolica_mmhg=(150, 14), pressao_diastolica_mmhg=(92, 9), fc_repouso_bpm=(82, 8),
        creatinina_mg_dl=(1.4, 0.3), taxa_filtracao_glomerular=(55, 15),
    ),
}

_WITHIN_PATIENT_NOISE_FRACTION = 0.15
_PROB_METFORMINA_POR_SEVERIDADE = {"controlado": 0.3, "moderado": 0.7, "descompensado": 0.85}
_PROB_INSULINA_POR_SEVERIDADE = {"controlado": 0.02, "moderado": 0.15, "descompensado": 0.55}
_PROB_COMORBIDADE_BASE_POR_SEVERIDADE = {"controlado": 0.05, "moderado": 0.15, "descompensado": 0.30}


def _sample_n_exams(rng: np.random.Generator) -> int:
    r = rng.random()
    if r < 0.25:
        return 1
    if r < 0.65:
        return int(rng.integers(2, 5))
    if r < 0.90:
        return int(rng.integers(5, 9))
    return int(rng.integers(9, 15))


def _clip(key: str, value: float) -> float:
    if key == "hba1c_pct":
        return max(4.0, min(16.0, value))
    if key == "taxa_filtracao_glomerular":
        return max(5.0, min(130.0, value))
    return max(0.0, value)


def generate_synthetic_dataframe(n_per_group: int = 30, seed: int = 42, missingness: bool = True) -> pd.DataFrame:
    """Gera um DataFrame sintético longitudinal com as colunas do plugin de diabetes."""
    rng = np.random.default_rng(seed)
    severities = ["controlado"] * n_per_group + ["moderado"] * n_per_group + ["descompensado"] * n_per_group
    rng.shuffle(severities)

    rows: list[dict] = []
    t_start_base = datetime(2019, 1, 1)

    for i, severity in enumerate(severities):
        profile = _PROFILES[severity]
        patient_id = f"DIAB_{i:04d}"
        patient_level = {key: rng.normal(mean, std) for key, (mean, std) in profile.items()}

        n_exams = _sample_n_exams(rng)
        first_date = t_start_base + timedelta(days=int(rng.integers(0, 365 * 5)))
        exam_dates = [first_date]
        for _ in range(n_exams - 1):
            exam_dates.append(exam_dates[-1] + timedelta(days=int(rng.integers(30, 300))))

        # Adoção de tratamento: metformina cedo (1ª linha), insulina mais tarde e mais rara —
        # ambas correlacionadas com severidade (confundimento por indicação deliberado).
        usa_metformina_desde = 0 if rng.random() < _PROB_METFORMINA_POR_SEVERIDADE[severity] else None
        exame_insulina = (
            int(rng.integers(1, n_exams)) if (rng.random() < _PROB_INSULINA_POR_SEVERIDADE[severity] and n_exams > 1) else None
        )

        comorbidades_paciente: list[str] = []
        carga_hiperglicemica_acumulada = 0.0

        for exam_idx, exam_date in enumerate(exam_dates):
            usa_metformina = usa_metformina_desde is not None and exam_idx >= usa_metformina_desde
            usa_insulina = exame_insulina is not None and exam_idx >= exame_insulina

            row: dict = {}
            for key, (_, std) in profile.items():
                v = patient_level[key] + rng.normal(0, std * _WITHIN_PATIENT_NOISE_FRACTION)

                if key in ("glicemia_jejum_mg_dl", "hba1c_pct"):
                    if usa_insulina:
                        v *= 1 - min(0.35, 0.08 * (exam_idx - (exame_insulina or 0) + 1))
                    elif usa_metformina:
                        v *= 1 - min(0.20, 0.04 * (exam_idx - (usa_metformina_desde or 0) + 1))

                row[key] = _clip(key, v)

            anos_passados = (exam_date - exam_dates[0]).days / 365.25
            row["idade"] = patient_level["idade"] + anos_passados

            carga_hiperglicemica_acumulada += max(0.0, row["hba1c_pct"] - 7.0)
            declinio_egfr = 0.15 * carga_hiperglicemica_acumulada
            aumento_creatinina = 0.003 * carga_hiperglicemica_acumulada
            row["taxa_filtracao_glomerular"] = _clip("taxa_filtracao_glomerular", row["taxa_filtracao_glomerular"] - declinio_egfr)
            row["creatinina_mg_dl"] = max(0.4, row["creatinina_mg_dl"] + aumento_creatinina)

            row["paciente"] = patient_id
            row["data_exame"] = exam_date

            if rng.random() < _PROB_COMORBIDADE_BASE_POR_SEVERIDADE[severity] * (1 + 0.1 * exam_idx):
                candidatos = [c for c in ["hipertensao", "retinopatia", "neuropatia", "doenca_cardiovascular"] if c not in comorbidades_paciente]
                if candidatos:
                    comorbidades_paciente.append(rng.choice(candidatos))
            for c in ["hipertensao", "retinopatia", "neuropatia", "doenca_cardiovascular"]:
                row[c] = 1 if c in comorbidades_paciente else 0

            row["metformina"] = 1 if usa_metformina else 0
            row["insulina"] = 1 if usa_insulina else 0

            rows.append(row)

    df = pd.DataFrame(rows)

    if missingness:
        _apply_missingness(df, "taxa_filtracao_glomerular", rng, frac=0.35)
        _apply_missingness(df, "circunferencia_abdominal_cm", rng, frac=0.40)

    return df


def _apply_missingness(df: pd.DataFrame, col: str, rng: np.random.Generator, frac: float) -> None:
    mask = rng.random(len(df)) < frac
    df.loc[mask, col] = np.nan
