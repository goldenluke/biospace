# BioSpace Dashboard — SAOS

Dashboard Streamlit multi-página construído **sobre o biospace**, não sobre
pandas com feature engineering solto. Nenhuma página recalcula
representação ou fenótipo — tudo isso já foi feito por `biospace` antes de
qualquer linha ser desenhada; o dashboard só projeta `Cohort` /
`RepresentationSpace` / `Phenotype` em gráficos e tabelas.

## Estrutura

```
biospace_dashboard/
├── App.py                          página inicial: upload de planilha OU dados sintéticos
├── requirements.txt
├── components/
│   ├── _bootstrap.py                garante que `import biospace` funcione (sibling package)
│   ├── pipeline.py                  Pipeline: DataFrame -> Cohort/Representation/Phenotypes/display_df
│   ├── state.py                     acesso ao Pipeline via st.session_state
│   ├── filters.py                   filtros de sidebar (fenótipo, classe, gênero, idade)
│   ├── charts.py                    histograma_faixas() e helpers de layout plotly
│   └── synthetic.py                 gerador de dados sintéticos (mesmas colunas/regras reais)
└── pages/
    ├── 1_Visao_Geral.py
    ├── 2_Apneia.py
    ├── 3_Hipoxemia.py
    ├── 4_Frequencia_Cardiaca.py
    ├── 5_Sintomas_e_Comorbidades.py
    ├── 6_Perfis.py                   fenotipagem: K-Means/HDBSCAN/GMM/Spectral + transição longitudinal
    ├── 7_Qualidade_de_Dados.py       ausência estrutural + pesos de completude por campo
    ├── 8_Paciente.py                 consulta individual + trajetória
    ├── 9_Sobrevivencia.py            Kaplan-Meier: tempo até fenótipo grave / início de tratamento
    ├── 10_Ontologia.py               dicionário de dados gerado automaticamente (Ontology)
    ├── 11_Early_Warning.py           Critical Slowing Down com significância por substitutos AR(1)
    ├── 12_Geometrias.py              geometrias pontuais, DTW/Gromov-Wasserstein, geometria aprendida (NCA), Riemanniana/dinâmica
    ├── 13_Sistemas_Dinamicos.py      NOVA — EvolutionOperator, StabilityOperator, previsão/simulação por paciente
    ├── 14_Inferencia_Causal.py       NOVA — balanceamento de linha de base, efeito observacional, gêmeo digital + cenários
    └── 15_Dominios_Latentes.py       NOVA — Inflamação/Fragilidade/Balanço Autonômico (proxies, não validados)
```

## Como rodar

O pacote `biospace` precisa estar em um diretório **irmão** deste
(`_bootstrap.py` adiciona o diretório pai ao `sys.path` automaticamente).
Isto é, a estrutura deve ser:

```
algum_diretorio/
├── biospace/                 <- o pacote (entities, geometry, phenotyping, plugins/sleep)
└── biospace_dashboard/        <- este dashboard
```

Depois:

```bash
cd biospace_dashboard
pip install -r requirements.txt
streamlit run App.py
```

Na página inicial, envie a planilha real (`.xlsx`, mesmo formato de
"Exames realizados") ou clique em "Gerar dados sintéticos" para testar sem
um arquivo real. Isso roda o pipeline completo:

```
DataFrame -> load_from_dataframe() -> Cohort/SleepRepresentation
    -> cohort.snapshot() -> RepresentationSpace
    -> ClinicalKMeansPhenotyper().fit() -> Phenotypes
```

e guarda tudo em `st.session_state` para as páginas em `pages/`.

## O que cada página mostra

- **Visão Geral**: KPIs populacionais, distribuição por classe de apneia (IDO) e por fenótipo.
- **Apneia**: distribuição do IDO, comparação com `classificar_apneia()`.
- **Hipoxemia**: SpO2, carga hipóxica, com aviso inline se o domínio tiver campos muito ausentes.
- **Frequência Cardíaca**: FC e amplitude cardíaca.
- **Sintomas e Comorbidades**: prevalência calculada com os mesmos `MAPA_DOENCAS`/`MAPA_SINTOMAS`/`MAPA_TRATAMENTOS` do biospace.
- **Perfis**: varredura de K por silhouette; seletor para comparar K-Means/HDBSCAN/GMM/Spectral sobre o MESMO RepresentationSpace; perfil por domínio (z-score médio); e transição de fenótipos longitudinal (`TransitionAnalyzer`, com filtro de intervalo mínimo entre exames).
- **Qualidade de Dados**: ausência estrutural por campo e o peso de completude efetivamente aplicado na geometria (ver `domain.feature_weights()`).
- **Paciente**: busca individual + trajetória (se houver mais de um exame).
- **Sobrevivência**: curvas de Kaplan-Meier — tempo até entrar no fenótipo mais severo, tempo até início de tratamento (AAM/CPAP), com censura tratada corretamente.
- **Ontologia**: dicionário de dados gerado automaticamente a partir da `Representation` atual (`Ontology.from_representation()`), com botão de download em Markdown.
- **Early Warning**: `CriticalSlowingDownDetector` interativo — escolha o indicador (distância multivariada ou uma Feature específica), rode a detecção com teste de significância por substitutos AR(1), veja a série de variância/autocorrelação por paciente, e a correlação (honesta, com aviso de que o resultado é misto nesta coorte) entre τ e progressão real de severidade.
- **Geometrias**: compare 2 pacientes sob 5 geometrias pontuais diferentes; compare 2 trajetórias inteiras via DTW (com visualização do alinhamento) e Gromov-Wasserstein sob demanda (1 par por vez — GW é caro); ajuste uma geometria aprendida (NCA) e veja o ganho de silhouette com um rótulo independente; explore a geometria Riemanniana (geodésica não-plana) e a geometria da doença (métrica local por fenótipo).
- **Sistemas Dinâmicos**: ajusta um `EvolutionOperator` (Ornstein-Uhlenbeck discreto por Feature) sobre toda a coorte, mostra quais Features são estáveis/instáveis, e permite prever/simular a evolução espontânea de um paciente específico.
- **Inferência Causal**: balanceamento de linha de base (detecta confundimento por indicação) + efeito observacional estimado; gêmeo digital (`clone()` → `do()` → `simulate()`) comparando múltiplos braços de intervenção contra um controle — com o aviso de que nada aqui é uma inferência causal identificada no sentido formal de Pearl.
- **Domínios Latentes**: ajusta os três domínios latentes (Inflamação, Fragilidade, Balanço Autonômico) e mostra as cargas fatoriais — todos claramente marcados como proxies não validados.

## Testado com

`streamlit.testing.v1.AppTest` — todas as 8 páginas + `App.py` rodam sem
exceção sobre dados sintéticos (ver processo de desenvolvimento). Não
inclui teste automatizado formal (pytest) nesta entrega.

## Limitações conhecidas

- Os filtros de sidebar (`components/filters.py`) atuam apenas sobre o
  `display_df` (visualização) — eles NÃO re-ajustam a `Representation`
  nem recalculam a fenotipagem sobre o subconjunto filtrado. Isso é
  intencional (evita vazamento/recomputação de referência a cada filtro),
  mas significa que os fenótipos exibidos após um filtro continuam sendo
  os mesmos estimados sobre a população inteira.
- Não há autenticação/controle de acesso — não use com dados sensíveis
  não anonimizados em um ambiente compartilhado sem adicionar isso.

## Melhorias gerais (esta rodada)

### Gerador sintético agora é LONGITUDINAL

Antes: `generate_synthetic_dataframe()` produzia 1 linha = 1 paciente,
sem nenhuma estrutura temporal — quebrando silenciosamente (dados
vazios/degenerados, não erros) as páginas que dependem de trajetórias:
Sobrevivência, Early Warning, Sistemas Dinâmicos, Inferência Causal,
DTW/Gromov-Wasserstein em Geometrias.

Agora, cada paciente sintético tem:
- **Nº de exames** com distribuição parecida com a real (mediana ~3-4,
  cauda longa até 17 — testado: 90 pacientes/404 exames com
  `n_per_group=30`, quase idêntico ao perfil documentado da planilha real).
- **Nível individual** (traço do paciente, sorteado uma vez) + ruído
  exame-a-exame MENOR que a variação entre pacientes — para que os
  exames de um mesmo paciente pareçam de fato do mesmo paciente.
- **Idade avançando de verdade** entre exames; **comorbidades que só se
  acumulam**, nunca desaparecem.
- **Adoção de tratamento (AAM/CPAP) correlacionada com severidade** —
  confundimento por indicação deliberado, o mesmo padrão da planilha
  real — com melhora gradual pós-adoção nas Features de apneia/hipoxemia,
  para que a página de Inferência Causal tenha sinal real para mostrar
  mesmo em modo de demonstração.

Testado: com `n_per_group=30`, 65/90 pacientes (72%) têm ≥2 exames, 17
têm ≥8 (o mínimo padrão do Early Warning) — todas as 15 páginas passam
sem erro, tanto no tamanho mínimo (`n_per_group=15`) quanto no padrão.

### App.py (página inicial)

- **Indicador de origem dos dados** (🧪 sintético / 📄 real) sempre visível.
- **Estatísticas longitudinais** (pacientes com ≥2 exames, mediana/máximo
  de exames) com aviso automático se a coorte carregada tiver poucos
  dados longitudinais para as páginas que precisam disso.
- **Botão "Limpar dados"** — remove o pipeline E qualquer análise já
  rodada nas outras páginas (`EvolutionOperator` ajustado, resultados de
  pareamento causal, etc.), para trocar de dataset sem misturar
  resultados antigos.
- **Diretório de páginas categorizado** (Transversal / Longitudinal /
  Geometria / Causal) na tela inicial, dado que agora são 15 páginas.

### Página 11 (Early Warning)

Corrigido um aviso do scipy (`ConstantInputWarning`) quando todos os
pacientes elegíveis tinham o mesmo τ ou a mesma progressão de
severidade — agora checado explicitamente antes de chamar `spearmanr`,
com uma mensagem clara em vez de um warning no console.

### Página 14 (Inferência Causal)

Mensagens de erro de amostra pequena (`check_baseline_balance`/
`match_on_propensity` com menos de 2 pacientes num grupo — pode
acontecer por acaso com poucos dados sintéticos, já que o tratamento se
divide entre AAM/CPAP) agora sugerem uma ação concreta (aumentar a
amostra ou trocar de tratamento) em vez de só mostrar a mensagem crua da
exceção.

## Fases 8, 9 e 10 no dashboard

3 páginas novas + 1 extensão, trazendo o que antes só existia em
código/exemplos/testes:

- **Sistemas Dinâmicos** (extensão): `twin.simulate_ensemble()` — múltiplos
  futuros simulados com faixa de incerteza (95%), não só um ponto
  determinístico.
- **Curvatura** (nova): as 3 formas independentes do projeto — temporal
  (via φ do EvolutionOperator), densidade populacional (poços de
  potencial / metaestabilidade), e estrutural (Ollivier-Ricci sobre o
  grafo de similaridade — inclui a comparação dentro vs. entre
  fenótipos que deu p=5,7e-19 nos dados reais).
- **GNN** (nova): treina uma `SimpleGCN` (NumPy puro) contra a mesma
  arquitetura sem propagação de mensagens, deixando o usuário variar a
  fração de pacientes rotulados — reproduz o padrão de cruzamento real
  (grafo atrapalha com muito rótulo, ajuda muito com pouco).
- **Foundation Model** (nova): protótipo de masked feature prediction,
  com aviso explícito de que não é um foundation model de verdade — só
  a arquitetura. Deixa escolher um paciente e mascarar Features
  específicas para ver a reconstrução.

Testado nas 19 páginas (16 antigas + 3 novas) + `App.py`, com dados
reais e sintéticos (inclusive no tamanho mínimo do slider) — zero erros.
