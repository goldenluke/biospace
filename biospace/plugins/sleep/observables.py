"""
biospace.plugins.sleep.observables
====================================

Observables migrados diretamente das colunas normalizadas da planilha real
("Exames realizados (SD).xlsx"), conforme 03_ingestao.py / 04_preprocessamento.py.
Cada `key` é o nome de coluna já normalizado (snake_case, sem acentos).
"""

from __future__ import annotations

from biospace.core import Observable

__all__ = [
    # Antropometria / demografia
    "Idade", "PesoKg", "AlturaCm", "Imc",
    # Apneia (proxy por dessaturação) e ronco
    "Ido", "IdoSono", "NoDeDessaturacoes",
    "TempoTotalDeRoncoMin", "TempoEmSilencio", "TempoEmRoncoBaixo", "TempoEmRoncoMedio", "TempoEmRoncoAlto",
    # Hipoxemia
    "Spo2Minima", "Spo2Media", "Spo2Maxima", "TempoSpo290", "CargaHipoxicaMinH",
    "NoDeEventosDeHipoxemia", "TempoTotalEmHipoxemiaMin",
    # Arquitetura do sono
    "TempoParaDormirMin", "TempoTotalDeSonoMin", "TempoAcordadoPosSonoMin", "EficienciaDoSono",
    # Cardiovascular
    "FcMinimaBpm", "FcMediaBpm", "FcMaximaBpm",
    # Texto livre estruturado
    "DoencasTexto", "SintomasTexto", "TratamentosTexto", "CondicoesTexto", "MedicamentosTexto",
    # Identificação / metadados
    "PacienteTexto", "GeneroTexto", "StatusTexto", "FaResultadoTexto",
]


# --- Antropometria / demografia -------------------------------------------------------------
class Idade(Observable):
    key = "idade"
    unit = "anos"


class PesoKg(Observable):
    key = "peso_kg"
    unit = "kg"


class AlturaCm(Observable):
    key = "altura_cm"
    unit = "cm"


class Imc(Observable):
    key = "imc"
    unit = "kg/m2"


# --- Apneia (proxy por dessaturação) e ronco -------------------------------------------------------------
class Ido(Observable):
    key = "ido"
    unit = "eventos/hora"
    description = "Índice de Dessaturação de Oxigênio (usado como proxy de apneia neste dataset)"


class IdoSono(Observable):
    key = "ido_sono"
    unit = "eventos/hora"


class NoDeDessaturacoes(Observable):
    key = "no_de_dessaturacoes"
    unit = "count"


class TempoTotalDeRoncoMin(Observable):
    key = "tempo_total_de_ronco_min"
    unit = "min"


class TempoEmSilencio(Observable):
    key = "tempo_em_silencio"
    unit = "min"


class TempoEmRoncoBaixo(Observable):
    key = "tempo_em_ronco_baixo"
    unit = "min"


class TempoEmRoncoMedio(Observable):
    key = "tempo_em_ronco_medio"
    unit = "min"


class TempoEmRoncoAlto(Observable):
    key = "tempo_em_ronco_alto"
    unit = "min"


# --- Hipoxemia -------------------------------------------------------------
class Spo2Minima(Observable):
    key = "spo2_minima"
    unit = "%"


class Spo2Media(Observable):
    key = "spo2_media"
    unit = "%"


class Spo2Maxima(Observable):
    key = "spo2_maxima"
    unit = "%"


class TempoSpo290(Observable):
    key = "tempo_spo2_90"
    unit = "%"
    description = "Percentual do tempo de sono com SpO2 < 90% (T90)"


class CargaHipoxicaMinH(Observable):
    key = "carga_hipoxica_min_h"
    unit = "%min/hora"


class NoDeEventosDeHipoxemia(Observable):
    key = "no_de_eventos_de_hipoxemia"
    unit = "count"


class TempoTotalEmHipoxemiaMin(Observable):
    key = "tempo_total_em_hipoxemia_min"
    unit = "min"


# --- Arquitetura do sono -------------------------------------------------------------
class TempoParaDormirMin(Observable):
    key = "tempo_para_dormir_min"
    unit = "min"


class TempoTotalDeSonoMin(Observable):
    key = "tempo_total_de_sono_min"
    unit = "min"


class TempoAcordadoPosSonoMin(Observable):
    key = "tempo_acordado_pos_sono_min"
    unit = "min"


class EficienciaDoSono(Observable):
    key = "eficiencia_do_sono"
    unit = "%"


# --- Cardiovascular -------------------------------------------------------------
class FcMinimaBpm(Observable):
    key = "fc_minima_bpm"
    unit = "bpm"


class FcMediaBpm(Observable):
    key = "fc_media_bpm"
    unit = "bpm"


class FcMaximaBpm(Observable):
    key = "fc_maxima_bpm"
    unit = "bpm"


# --- Texto livre estruturado -------------------------------------------------------------
class DoencasTexto(Observable):
    key = "doencas"
    unit = "texto"


class SintomasTexto(Observable):
    key = "sintomas"
    unit = "texto"


class TratamentosTexto(Observable):
    key = "tratamentos"
    unit = "texto"


class CondicoesTexto(Observable):
    key = "condicoes"
    unit = "texto"


class MedicamentosTexto(Observable):
    key = "medicamentos"
    unit = "texto"


# --- Identificação / metadados -------------------------------------------------------------
class PacienteTexto(Observable):
    key = "paciente"
    unit = "texto"


class GeneroTexto(Observable):
    key = "genero"
    unit = "texto"


class StatusTexto(Observable):
    key = "status"
    unit = "texto"


class FaResultadoTexto(Observable):
    key = "fa_resultado"
    unit = "texto"
    description = "Resultado de Fibrilação Atrial"
