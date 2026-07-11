"""
biospace.plugins.sleep.clinical_maps
=======================================

Mapas de texto livre -> código clínico, migrados literalmente de
06_features_texto.py (MAPA_DOENCAS, MAPA_SINTOMAS, MAPA_TRATAMENTOS).

Mantidos como dados puros (não lógica) para que ComorbidityDomain,
SymptomsDomain e TreatmentDomain os consumam de forma determinística.
"""

from __future__ import annotations

__all__ = ["MAPA_DOENCAS", "MAPA_SINTOMAS", "MAPA_TRATAMENTOS"]

MAPA_DOENCAS: dict[str, str] = {
    "Hipertensão arterial": "hipertensao",
    "Diabetes": "diabetes",
    "Dislipidemia": "dislipidemia",
    "Depressão": "depressao",
    "Arritmia cardíaca": "arritmia",
    "Refluxo gastroesofágico": "refluxo",
    "Insuficiência cardíaca": "insuficiencia_cardiaca",
    "Doença respiratória": "doenca_respiratoria",
    "Doença arterial coronária": "doenca_coronaria",
    "Asma": "asma",
    "Bronquite / Enfisema": "bronquite_enfisema",
    "Câncer": "cancer",
}

MAPA_SINTOMAS: dict[str, str] = {
    "Ronco alto e frequente": "ronco",
    "Sono não reparador": "sono_nao_reparador",
    "Sonolência excessiva diurna": "sonolencia_diurna",
    "Despertares frequentes": "despertares_frequentes",
    "Dificuldade de concentração": "dificuldade_concentracao",
    "Perda de memória": "perda_memoria",
    "Dificuldade para manter o sono": "insonia_manutencao",
    "Diminuição da libido ou disfunção erétil": "disfuncao_sexual",
    "Refluxo gastroesofágico": "refluxo_sintoma",
    "Cefaleia matinal": "cefaleia_matinal",
    "Engasgos noturnos": "engasgos_noturnos",
    "Irritabilidade": "irritabilidade",
    "falta de ar": "dispneia",
}

MAPA_TRATAMENTOS: dict[str, str] = {
    "Aparelho de avanço mandibular": "aam",
    "Aparelho de avanço": "aam",
    "Placa de avanço": "aam",
    "Placa": "aam",
    "aparelho de avanço": "aam",
    "CPAP": "cpap",
    "Terapia posicional": "terapia_posicional",
    "Oxigênio": "oxigenio",
    "Dilatador nasal": "dilatador_nasal",
}
