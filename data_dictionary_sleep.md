# Ontologia: sleep

8 domínios, 28 observables.

## anthropometric
_Características antropométricas e demográficas_

| Observable | Unidade | Descrição |
|---|---|---|
| `idade` | anos | — |
| `imc` | kg/m2 | — |
| `peso_kg` | kg | — |
| `altura_cm` | cm | — |

## apnea
_Índice de dessaturação (proxy de apneia) e ronco associado_

| Observable | Unidade | Descrição |
|---|---|---|
| `ido` | eventos/hora | Índice de Dessaturação de Oxigênio (usado como proxy de apneia neste dataset) |
| `ido_sono` | eventos/hora | — |
| `no_de_dessaturacoes` | count | — |
| `tempo_total_de_ronco_min` | min | — |
| `tempo_em_ronco_baixo` | min | — |
| `tempo_em_ronco_medio` | min | — |
| `tempo_em_ronco_alto` | min | — |

## hypoxia
_Carga de dessaturação de oxigênio_

| Observable | Unidade | Descrição |
|---|---|---|
| `spo2_minima` | % | — |
| `spo2_media` | % | — |
| `spo2_maxima` | % | — |
| `tempo_spo2_90` | % | Percentual do tempo de sono com SpO2 < 90% (T90) |
| `carga_hipoxica_min_h` | %min/hora | — |
| `no_de_eventos_de_hipoxemia` | count | — |
| `tempo_total_em_hipoxemia_min` | min | — |

## sleep_architecture
_Latência, duração e eficiência do sono_

| Observable | Unidade | Descrição |
|---|---|---|
| `tempo_para_dormir_min` | min | — |
| `tempo_total_de_sono_min` | min | — |
| `tempo_acordado_pos_sono_min` | min | — |
| `eficiencia_do_sono` | % | — |

## cardiovascular
_Frequência cardíaca durante o sono_

| Observable | Unidade | Descrição |
|---|---|---|
| `fc_minima_bpm` | bpm | — |
| `fc_media_bpm` | bpm | — |
| `fc_maxima_bpm` | bpm | — |

## comorbidity
_Comorbidades reportadas (texto livre estruturado)_

| Observable | Unidade | Descrição |
|---|---|---|
| `doencas` | texto | — |

## symptoms
_Sintomas clínicos reportados (texto livre estruturado)_

| Observable | Unidade | Descrição |
|---|---|---|
| `sintomas` | texto | — |

## treatment
_Tratamentos em uso (texto livre estruturado)_

| Observable | Unidade | Descrição |
|---|---|---|
| `tratamentos` | texto | — |
