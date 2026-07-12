# BioSpace Dashboard — NHANES

Dashboard Streamlit para explorar a coorte NHANES real (ciclo
Pre-pandemic, agosto/2017–março/2020) através do framework BioSpace —
`biospace.plugins.metabolic` (`MetabolicRepresentation`) e as
interpretações clínicas `classify_diabetes_status` /
`classify_metabolic_syndrome_risk`.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run App.py
```

Requer os 6 arquivos `.XPT` do ciclo Pre-pandemic em
`/mnt/user-data/uploads` (ou outro caminho informado na tela inicial):
`P_DEMO.xpt`, `P_GHB.xpt`, `P_GLU.xpt`, `P_BMX.xpt`, `P_BPXO.xpt`,
`P_DIQ.xpt`. São dados públicos do CDC — baixe em
https://wwwn.cdc.gov/nchs/nhanes/tutorials/PreliminaryData.aspx se não
os tiver localmente.

## Páginas

| Página | O que mostra |
|---|---|
| Visão Geral | Distribuições demográficas e laboratoriais |
| Diagnóstico | Classificação laboratorial (critério ADA) x autorrelato — sensibilidade 75,0%, especificidade 95,0% |
| Síndrome Metabólica | Critério adaptado NCEP ATP III (2 de 4 critérios disponíveis) |
| Coerência de Processo | `check_process_coherence` — confirma em população real (p=0,0022), diferente do gerador sintético usado em outras partes do projeto |
| Paciente | Busca individual, vetor de representação completo |
| Estabilidade e Curvatura | Varredura de estabilidade fenotípica com/sem idade, curvatura estrutural — mesma metodologia de SAOS |

## Achado novo: o oposto exato de SAOS na estabilidade fenotípica

Em SAOS, nenhuma das 28 configurações testadas cruzava o limiar de
estabilidade (ARI≥0,7). No NHANES, K-Means chega a 0,957 em K=2.
Testado se é só idade disfarçada de estrutura metabólica: em K=2 a
estabilidade sobrevive quase intacta sem idade — estrutura genuína
além dela; em K=3 desaba sem idade — ali, idade sustentava a
partição sozinha. Curvatura estrutural, por outro lado, **não**
discrimina fronteira de fenótipo aqui (diferente de SAOS) — achado
negativo, consistente com a interpretação de que curvatura é uma
assinatura de fronteiras frágeis em contínuos mal separados, ausente
quando o fenótipo já é bem separado.

## Achados documentados nesta coorte

- Subdiagnóstico de diabetes: sensibilidade 75,0%, especificidade
  95,0% do critério laboratorial contra autorrelato (corrigido de um bug
  real em `_raw_value()` — ver `METABOLISM_FINDINGS.md`).
- Subdiagnóstico de pré-diabetes ainda mais acentuado: 91,5% dos casos
  classificados como pré-diabetes não têm autorrelato de diabetes.
- Coerência de processo fisiológico (HbA1c e glicemia correlacionam
  mais entre si que com outras variáveis) confirma-se estatisticamente
  em população real — o oposto do achado no gerador sintético de
  diabetes usado em outras partes do projeto.

Ver `biospace/METABOLISM_FINDINGS.md` para o documento consolidado com
todos os achados do processo, e o artigo "Achados Empíricos sobre
Diabetes Mellitus Tipo 2 em Duas Fontes de Dados Reais e
Independentes" para a análise completa com tabelas.

## Testado com

`streamlit.testing.v1.AppTest` — `App.py` + as 5 páginas, incluindo o
clique real do botão de carga (não apenas pipeline pré-carregado em
`session_state`).
