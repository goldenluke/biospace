# BioSpace Dashboard — Diabetes Tipo 2 (sintético)

Segundo dashboard Streamlit, irmão de `biospace_dashboard/` (SAOS),
construído sobre o segundo plugin de doença (`biospace.plugins.diabetes`).
Prova de que a arquitetura generaliza — **não** uma ferramenta clínica:
não existe planilha real de diabetes neste projeto, só uma coorte
sintética longitudinal gerada localmente.

## Diferença de propósito em relação ao dashboard de SAOS

| | `biospace_dashboard/` (SAOS) | `biospace_dashboard_diabetes/` |
|---|---|---|
| Dados reais | Sim (upload de .xlsx) | Não — só sintéticos |
| Objetivo | Ferramenta de análise validada | Prova de arquitetura (segundo plugin) |
| Nº de páginas | 15 | 8 |

## Estrutura

```
biospace_dashboard_diabetes/
├── App.py                    gera a coorte sintética, mostra resumo
├── requirements.txt           mesmas dependências do dashboard de SAOS
├── components/
│   ├── _bootstrap.py           garante `import biospace`
│   ├── pipeline.py              DataFrame -> Cohort/Representation/Phenotypes
│   ├── state.py                 acesso ao Pipeline via st.session_state
│   └── charts.py                 helpers de layout plotly
└── pages/
    ├── 1_Visao_Geral.py
    ├── 2_Controle_Glicemico.py    glicemia/HbA1c por classe de controle
    ├── 3_Funcao_Renal.py           eGFR/creatinina + mecanismo de declínio por exposição
    ├── 4_Perfis.py                  fenotipagem (KMeans genérico)
    ├── 5_Dominio_Latente.py          proxy de resistência à insulina (InsulinResistanceProxyDomain)
    ├── 6_Sistemas_Dinamicos.py        EvolutionOperator/StabilityOperator
    ├── 7_Inferencia_Causal.py          balanceamento + efeito observacional
    └── 8_Paciente.py                    busca individual + trajetória
```

## Rodando localmente

```bash
pip install -r requirements.txt
streamlit run App.py
```

Clique em "Gerar dados sintéticos" na tela inicial — não há upload,
porque não existe dado real de diabetes neste projeto.

## Achado de rigor destacado na página "Função Renal"

O gerador sintético (`biospace.plugins.diabetes.synthetic`) liga o
declínio de eGFR/aumento de creatinina à **exposição glicêmica crônica
acumulada** ao longo da trajetória do paciente, não apenas à severidade
do instante — hiperglicemia crônica danifica os rins, um mecanismo
clínico real. A página mostra a correlação (positiva, como esperado)
entre HbA1c médio na trajetória e queda de eGFR, calculada em tempo real
sobre a coorte carregada.

## Fases 8, 9 e 10 no dashboard

Mesmas 3 páginas novas + extensão do dashboard de sleep, adaptadas para
o plugin de diabetes (Curvatura, GNN, Foundation Model, e simulação em
conjunto na página de Sistemas Dinâmicos) — mesmo código-fonte por trás
(`biospace.geometry`, `biospace.gnn`, `biospace.foundation`,
`biospace.causal.DigitalTwin`), sem alteração nenhuma, só a camada de
interface adaptada. Prova de genericidade mais uma vez: o mesmo
ferramental funciona sobre um plugin de doença completamente diferente.

Testado nas 12 páginas (9 antigas + 3 novas) + `App.py`, com dados
sintéticos (inclusive no tamanho mínimo do slider) — zero erros.
