# BioSpace Dashboard — UCI Diabetes 130-US Hospitals

Dashboard Streamlit para explorar a coorte UCI Diabetes 130-US
Hospitals real (Strack et al., 2014) através do framework BioSpace —
`biospace.datasets.uci_diabetes` (`UCIHospitalRepresentation`), uma
representação genuinamente diferente da usada para o NHANES (sem
laboratório contínuo, mas com utilização hospitalar completa e
trajetórias reais multi-encontro).

## Como rodar

```bash
pip install -r requirements.txt
streamlit run App.py
```

Requer `diabetic_data.csv` em `/mnt/user-data/uploads` (ou outro
caminho informado na tela inicial). Dado público — ver Strack et al.
(2014), *BioMed Research International*.

A base completa tem ~100 mil linhas — o carregamento inicial pode
levar 1-2 minutos. A tela inicial oferece uma amostra reduzida (5.000
encontros) para exploração rápida.

## Páginas

| Página | O que mostra |
|---|---|
| Visão Geral | Distribuições de utilização hospitalar, readmissão |
| Fenótipos e Readmissão | A associação mais forte do projeto — fenótipo sem idade/diagnóstico prevê readmissão |
| Trajetórias | Pacientes com múltiplos encontros (23,4% da coorte), variável derivada (slope) |
| Paciente | Busca individual, vetor de representação completo |

## Achado mais forte do projeto

Fenotipagem K-Means sobre utilização hospitalar + testagem glicêmica
esparsa + intensidade de medicação — **sem idade, sem diagnóstico** —
produz um fenótipo com quase o dobro da taxa de readmissão em 30 dias
(8,75% contra 3,97%–4,64% dos demais). O fenótipo de maior risco não é
o de maior intensidade de tratamento farmacológico — é o de maior
utilização hospitalar **prévia** (visitas ambulatoriais, de emergência
e internações anteriores ao encontro atual), consistente com a
literatura estabelecida de predição de readmissão hospitalar.

Ver `biospace/METABOLISM_FINDINGS.md` para o documento consolidado com
todos os achados do processo, e o artigo "Achados Empíricos sobre
Diabetes Mellitus Tipo 2 em Duas Fontes de Dados Reais e
Independentes" para a análise completa com tabelas.

## Testado com

`streamlit.testing.v1.AppTest` — `App.py` + as 4 páginas, incluindo o
clique real do botão de carga.
