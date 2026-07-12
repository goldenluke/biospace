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
| Dinâmica | `MeanRevertingEvolutionOperator` na trajetória real multi-encontro, diagnóstico de robustez |
| Sobrevivência | Kaplan-Meier/Cox — predição prospectiva por fenótipo de baseline, nuancia o achado transversal publicado |
| Alerta Precoce | Critical slowing down — achado negativo real, com interpretação honesta do limite de dado |
| Predição e Explicabilidade | RandomForest/LogisticRegression + SHAP — triangula com a página de Sobrevivência |
| Coortes Geométricas | Coorte por proximidade a um centroide de fenótipo vs. cluster K-Means original |

## Achado novo: dinâmica estável, sem artefato de outlier

Primeira aplicação do módulo de dinâmica a trajetória real fora de
sleep/sintético — 16.773 pacientes com ≥2 encontros, dinâmica
globalmente estável (13/13 Features). A Feature mais perto da
instabilidade (`number_emergency`, φ=0,98) foi testada no mesmo
diagnóstico de robustez que expôs um artefato de outlier em SAOS —
aqui a conclusão é robusta, consistente com o fenômeno real de
"frequent flyers" em uso de emergência, não um artefato de amostra.

## Achado mais forte do projeto — e uma qualificação real, testada

Fenotipagem K-Means sobre utilização hospitalar + testagem glicêmica
esparsa + intensidade de medicação — **sem idade, sem diagnóstico** —
produz um fenótipo com quase o dobro da taxa de readmissão em 30 dias
(8,75% contra 3,97%–4,64% dos demais). O fenótipo de maior risco não é
o de maior intensidade de tratamento farmacológico — é o de maior
utilização hospitalar **prévia** (visitas ambulatoriais, de emergência
e internações anteriores ao encontro atual), consistente com a
literatura estabelecida de predição de readmissão hospitalar.

**Isso não é a história inteira, e testamos isso, não só afirmamos.**
Adicionando um quarto domínio — categorias diagnósticas ICD-9,
extraídas de `diag_1`/`diag_2`/`diag_3` — o fenótipo dominante muda:
passa a isolar um grupo minúsculo (212 pacientes, 0,3%) de internação
extremamente longa, e a associação com readmissão fica bem mais fraca
(~1,5x, não mais ~2,2x). **A página "Fenótipos e Readmissão" agora
mostra qual das duas representações está ativa** e explica a
diferença — a tela inicial tem um seletor para alternar entre elas e
comparar você mesmo. Nenhuma das duas está errada; capturam estruturas
diferentes do mesmo dado real — a mesma tese do artigo "Representation
Before Inference", agora numa instância real dentro de uma única fonte
de dados.

Ver `biospace/METABOLISM_FINDINGS.md` para o documento consolidado com
todos os achados do processo, e o artigo "Achados Empíricos sobre
Diabetes Mellitus Tipo 2 em Duas Fontes de Dados Reais e
Independentes" para a análise completa com tabelas.

## Testado com

`streamlit.testing.v1.AppTest` — `App.py` + as 4 páginas, incluindo o
clique real do botão de carga.
