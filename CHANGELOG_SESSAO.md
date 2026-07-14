# BioSpace — Changelog Consolidado desta Sessão

Este documento resume, num só lugar, o que foi construído, testado e
descoberto ao longo desta sessão — para orientação, não como
substituto de `biospace/METABOLISM_FINDINGS.md` (o registro técnico
detalhado, seção por seção, de cada achado empírico) nem de
`biospace/README.pt-BR.md` (a referência arquitetural completa).

**Nota sobre precisão**: esta sessão foi longa o bastante para incluir
partes que não ficaram visíveis nesta janela de contexto — em pelo
menos seis ocasiões, uma investigação encontrou trabalho substancial
já feito (dashboards, testes, correções) que precisou ser descoberto e
verificado, não assumido. Os números abaixo foram conferidos agora,
rodando os testes e contando arquivos de verdade — não reconstruídos
de memória.

## Estado atual verificado

- **324 testes** coletados (`pytest --collect-only`), organizados em
  46 arquivos.
- **4 dashboards** Streamlit: sleep (18 páginas, dado real de 355
  pacientes), diabetes sintético (11 páginas), NHANES (6 páginas,
  dado real do CDC), UCI Diabetes 130-US Hospitals (7 páginas, dado
  real de 101.766 encontros).
- **19 módulos** no núcleo (`biospace/`): `core`, `causal`,
  `datasets`, `dynamics`, `early_warning`, `foundation`, `geometry`,
  `gnn`, `graph`, `intervention`, `latent`, `longitudinal`,
  `ontology`, `phenotyping`, `plugins`, `prediction`,
  `representation_learning`, `risk`, `survival` — todos com pelo
  menos um arquivo de teste real referenciando.
- **3 plugins de doença**: sleep (SAOS, dado clínico real),
  metabolic/diabetes (NHANES + UCI, dado real), com o pacote
  `diabetes/` reduzido a uma camada fina de reexportação sobre
  `metabolic/`.
- **2 fontes de dado real do CDC/UCI** totalmente integradas: NHANES
  (10 arquivos `.XPT`, ciclo Pre-pandemic) e UCI Diabetes 130-US
  Hospitals (101.766 encontros, 71.518 pacientes).
- **Documentos**: `manuscript.html` (926 linhas, teoria axiomática
  completa — axiomas, métrica, álgebra de operadores, teoremas,
  categoria), `index.html` (736 linhas, landing page bilíngue),
  `README.pt-BR.md` (1001 linhas), `METABOLISM_FINDINGS.md` (640
  linhas, 14 seções), mais uma dezena de artigos científicos em
  `artigo/` e `artigo_diabetes/`.

## Linha do tempo desta sessão (a parte que acompanhei diretamente)

### 1. Fundação teórica — do meta-modelo aos axiomas
A pedido de uma revisão externa detalhada, o manuscrito ganhou uma
seção inteira de teoria axiomática (§4.12–4.17): 5 axiomas conectados
a contratos já testados, o espaço de representação formalizado como
métrica (com a ressalva honesta de que DTW não satisfaz a desigualdade
triangular — confirmado contra um contraexemplo publicado, corrigindo
uma citação errada que eu mesmo tinha feito de memória), álgebra de
operadores (monoide, não grupo — a maioria dos operadores não é
inversível), teoremas (ponto fixo de Banach conectado ao φ que o
projeto já mede; continuidade com exceções documentadas nas
transições de ausência de dado), e organização categórica.

### 2. Dado real: NHANES e UCI
Integração completa das duas fontes, com bugs reais encontrados só
com arquivo real (não fabricado): `format="xport"` não `"xpt"`;
ciclo NHANES errado assumido inicialmente; um bug real em
`_raw_value()` que afetou um número **já publicado** no artigo de
diabetes (sensibilidade corrigida de 66,6% para 75,0% — nota de
correção registrada com transparência, não escondida).

### 3. Camada de lipídios/renal, síndrome metabólica completa
eGFR real via fórmula CKD-EPI 2021 (coeficientes confirmados contra a
National Kidney Foundation, não de memória), `LipidDomain` novo,
síndrome metabólica de 5 critérios sexo-específicos substituindo a
versão adaptada de 4.

### 4. Um achado de representação, não planejado
Ao enriquecer a representação do NHANES, dois achados já "fechados"
**mudaram de verdade** — estabilidade fenotípica ficou mais variável,
e curvatura estrutural passou a discriminar fenótipo significativamente
(era não-significativo antes). Instância real, dentro da mesma fonte
de dado, da tese central do projeto (Representação Antes da
Inferência).

### 5. Auditoria geral e a pasta `_legacy/`
Um apanhado do projeto achou 5 arquivos órfãos na raiz (protótipo
original do meta-modelo de 6 de julho, rascunhos de artigo
desatualizados, um dataset sintético com nome enganoso) — investigados
individualmente (diff de conteúdo, não só nome) antes de mover, todos
confirmados sem conteúdo único perdido. Movidos para `_legacy/` com um
README explicando cada um, não apagados.

### 6. Análise de sobrevivência, alerta precoce, e a descoberta da duplicação
Implementei Kaplan-Meier/Cox (`biospace.survival`, usando `lifelines`)
e sinais de alerta precoce (critical slowing down) — só para descobrir,
investigando módulos supostamente não testados, que versões **mais
rigorosas** de ambos já existiam (`biospace.longitudinal.survival`,
`biospace.early_warning.CriticalSlowingDownDetector`) de uma parte
anterior da sessão que não vi. A versão de alerta precoce mais simples
foi removida em favor da mais completa (validada contra uma bifurcação
sela-nó genuína, não um AR(1) arbitrário); os dois módulos de
sobrevivência foram mantidos, com a diferença de design documentada
explicitamente nos dois lados (tempo ordinal vs. calendário real).

### 7. Triangulação: sobrevivência, alerta precoce e aprendizado supervisionado concordam
Três métodos completamente diferentes (Cox por fenótipo, RandomForest,
LogisticRegression) convergem para o mesmo teto (AUC/C-index≈0,50-0,53)
ao tentar prever readmissão futura só com dado do primeiro encontro —
evidência forte de que a limitação é do dado (pouco acompanhamento por
paciente na UCI), não da escolha de método.

### 8. A auditoria de módulos não testados (6 rodadas seguidas)
`prediction/`, `risk/`, `latent/factor_analysis.py`, e os 3 arquivos
de `longitudinal/` — todos achados com zero cobertura de teste,
arquiteturalmente completos, nunca aplicados a dado nenhum. 45 testes
novos no total, incluindo uma validação cruzada decisiva: a
implementação própria de Kaplan-Meier do projeto (sem `lifelines`)
concorda com `lifelines` até 1e-9 de precisão no mesmo dado.

## Achados negativos, registrados com o mesmo peso que os positivos

Esta sessão documentou tantos achados **nulos** quanto positivos, sem
escondê-los:
- Curvatura estrutural não discrimina fenótipo no NHANES original
  (achado revertido depois, ver item 4 acima).
- O grafo (GNN) atrapalha, não ajuda, em quase toda fração de rótulo
  testada no NHANES e na UCI — oposto do padrão em SAOS.
- Sinais de alerta precoce (critical slowing down) não predizem
  readmissão futura na UCI — poder estatístico insuficiente, não falha
  do método (confirmado com a versão mais rigorosa do detector).
- Predição prospectiva de readmissão (fenótipo, RandomForest,
  LogisticRegression) não supera o acaso de forma relevante.
- Composição demográfica do fenótipo de alto risco na UCI não mostra
  disparidade relevante por raça/gênero (Cramér's V<0,1).

## O que ficou explicitamente pendente

- E-mails para os dois contatos externos identificados (Andrey
  Zinchuk para SAOS, o autor do trabalho de Marquand para a crítica) —
  conteúdo pronto, não enviado.
- MIMIC-IV — terceira base do roteiro original, precisa de
  credenciamento formal (CITI + acordo de uso), não só um upload.
- Funtor formal entre categorias de doenças além de sleep↔metabolic.
- Dado adicional do NHANES eventualmente relevante (nenhum pendente
  específico identificado no momento).

## Onde cavar mais fundo

- `biospace/METABOLISM_FINDINGS.md` — 14 seções, cada achado empírico
  com números exatos, testes correspondentes, e o raciocínio por trás
  de cada correção.
- `biospace/README.pt-BR.md` — referência arquitetural completa.
- `manuscript.html` — a teoria formal completa.
- `_legacy/README.md` — o que foi arquivado e por quê.
