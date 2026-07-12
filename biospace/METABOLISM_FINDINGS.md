# Metabolismo e Diabetes no BioSpace: Documento Consolidado

Este documento reúne, num só lugar, tudo que foi construído e descoberto
ao longo do processo de generalizar o BioSpace para o sistema
metabólico/endócrino — da proposta inicial ("não implemente diabetes,
implemente um Metabolic Domain Package") até a validação em duas fontes
de dados públicas reais. As seções correspondentes no `README.pt-BR.md`
continuam sendo a referência técnica de cada peça; este documento é a
narrativa e o resumo de achados, para quem quer entender o que foi feito
e por quê sem ler o código.

## 1. A guinada arquitetural: de "plugin de diabetes" para "pacote metabólico + interpretação"

O projeto tinha um `plugins/diabetes/` — um pacote de representação
inteiro, com domínios, sistema e representação nomeados em torno de
"diabetes". Uma revisão externa apontou o problema: o núcleo do
BioSpace já organiza domínios por significado fisiológico (`GlycemicDomain`,
`RenalDomain`, ...), não por doença — mas o *pacote* continuava
estruturado em torno de um nome de doença, contradizendo o próprio
princípio.

**O que mudou**: `plugins/metabolic/` passou a ser o pacote de
representação genérico — `MetabolicSystem`, `MetabolicRepresentation`,
os mesmos 6 domínios físiológicos da época da refatoração (hoje são 7
— `LipidDomain` foi adicionado depois, Seção 6), nenhum deles sabendo
o que é diabetes. `plugins/diabetes/` virou uma camada fina: reexporta
tudo de `metabolic` sob os nomes antigos (`DiabetesSystem is
MetabolicSystem`, literalmente a mesma classe, testado) e contribui só
o que é genuinamente específico — o critério clínico de diagnóstico e
o
gerador sintético de cenário.

**A prova de que isso não é só reorganização de código**: implementamos
uma segunda interpretação clínica independente sobre a mesma
representação — `classify_metabolic_syndrome_risk` (critérios NCEP ATP
III adaptados) ao lado de `classify_diabetes_status` — e testamos que
as duas discordam exatamente onde deveriam discordar (diabetes sem
síndrome, síndrome sem diabetes, as duas, nenhuma). Se sempre
concordassem, síndrome metabólica seria só diabetes com outro nome.

## 2. Camada de Processos Fisiológicos

Proposta na mesma revisão: "diabetes não é um conjunto de exames, é um
conjunto de processos biológicos — as observações medem esses
processos". Implementamos `PhysiologicalProcess` como camada opcional e
aditiva: `Observable.process` (nome, default `None`), consultas
computadas `SemanticDomain.processes()` e
`Representation.features_by_process()` que cruzam fronteira de
domínio de verdade (testado com um caso decisivo: duas Features de
domínios diferentes declarando o mesmo processo aparecem juntas).

**Retrocompatibilidade testada, não assumida**: o plugin sleep (real,
validado) nunca foi tocado por esta mudança — `SleepRepresentation().processes()`
continua vazio, verificado em teste.

**`check_process_coherence`**: valida a alegação de que Features do
mesmo processo correlacionam mais entre si que Features de processos
diferentes — não apenas declara. Validado em cenários sintéticos com
verdade conhecida (confirma quando há correlação real; não confirma em
ruído puro) antes de aplicar a qualquer dado real.

## 3. Variáveis Derivadas

Também proposto na revisão: distinção entre Observação (uma glicemia)
e Variável Derivada (variabilidade glicêmica, carga hiperglicêmica
acumulada, slope de HbA1c — precisam da trajetória inteira, não de um
instante). `DerivedVariable` é uma entidade paralela a
`SemanticDomain` (não subclasse — `encode()` só enxerga um instante,
nunca a trajetória). `GlycemicBurdenVariable` reaproveita
exatamente o mecanismo já validado no gerador sintético de diabetes
(carga = soma do excesso de HbA1c acima de 7,0%), não inventa um novo.

Testado contra trajetória com resultado calculado à mão (precisão
1e-3) e, mais tarde, contra uma trajetória hospitalar REAL de 40
pontos (Seção 5) — a primeira vez que rodou em dado real multi-ponto,
não sintético nem transversal.

## 4. NHANES: primeira fonte real

Tentamos baixar diretamente — bloqueado por restrição de rede do
ambiente (`403 host_not_allowed` contra `cdc.gov`, verificado
empiricamente). O usuário enviou os 6 arquivos `.XPT` manualmente.

**Dois bugs reais, só descobertos com arquivo real** (nenhum teste com
dado fabricado os pegaria, porque mockavam a leitura de arquivo
inteira):

1. `pandas.read_sas` exige `format="xport"`, não `format="xpt"` como o
   módulo assumia.
2. Os arquivos enviados eram do ciclo "Pre-pandemic" (ago/2017–mar/2020
   combinado), não 2017-2018 isolado — pressão arterial mudou de
   metodologia (`P_BPXO`, variáveis `BPXOSY1`/`BPXODI1`, não
   `P_BPX`/`BPXSY1`/`BPXDI1`).

**Achados clínicos reais** (n=9.232 adultos ≥20 anos):

- `classify_diabetes_status` (critério ADA) vs. diagnóstico
  autorreferido: sensibilidade 75,0%, especificidade 95,0%, acurácia
  91,8% (n=7.848 comparáveis) — consistente com subdiagnóstico de
  diabetes, fenômeno documentado na literatura. **Correção
  registrada**: valor original (66,6%/95,6%/91,0%, n=8.965) continha
  um bug real — `_raw_value()` caía silenciosamente em 0.0 em vez de
  `None` para Feature ausente, tornando o ramo "indeterminado"
  inalcançável; 1.140 adultos sem HbA1c nem glicemia eram
  classificados como "normal" em vez de excluídos corretamente.
  Corrigido, artigo e testes atualizados (`test_raw_value_correctly_returns_none_for_missing_feature_not_zero`,
  `test_diabetes_sensitivity_corrected_after_raw_value_bugfix`).
- 91,5% dos casos classificados como pré-diabetes por laboratório NÃO
  têm autorrelato de diabetes — subdiagnóstico de pré-diabetes ainda
  mais acentuado.
- `check_process_coherence` **confirma** em população real (|r|
  mesmo-processo=0,782 vs. diferente-processo=0,151, p=0,0022) — o
  oposto exato do achado no gerador sintético de diabetes
  (`is_coherent=False`). Mesma ferramenta, duas fontes com propriedades
  genuinamente diferentes, duas respostas corretas.
- Síndrome metabólica (critério adaptado, 2 de 4 critérios
  disponíveis): 43,9% de risco elevado numa amostra de 2.000 adultos.
- NHANES amostra toda a população, inclusive crianças (idade mínima
  ≈0) — achado esperado do desenho do inquérito, não um bug; filtramos
  para adultos antes de qualquer interpretação clínica de diabetes.

## 5. UCI Diabetes 130-US Hospitals: segunda fonte real, estrutura genuinamente diferente

Inspecionado antes de qualquer mapeamento: **não tem** HbA1c/glicemia
contínuas (só categorias, 83-95% ausentes), **não tem** IMC,
circunferência abdominal, pressão arterial, creatinina. **Tem** 8
variáveis de utilização hospitalar 100% completas e 23 classes de
medicação detalhadas.

**Consequência honesta, não escondida**: isto não é "a mesma
`MetabolicRepresentation` numa segunda fonte" — construímos uma
representação genuinamente diferente (`HospitalUtilizationDomain`,
`GlycemicTestingDomain`, `MedicationIntensityDomain`), conectada ao
NHANES só onde há conexão mecanística real: `GlycemicTestingDomain`
declara o mesmo `process="glucose_homeostasis"`.

**Achado estrutural não previsto**: 23,4% dos pacientes (16.773 de
71.518) têm múltiplos encontros — até 39 no máximo observado. A
primeira trajetória real de verdade do projeto (não sintética, não
transversal). `encounter_id` não é data real, mas cresce
monotonicamente — usado como proxy de ordem, documentado
explicitamente como tal.

**O achado mais forte de todo o processo**: fenotipagem K-Means (K=4,
só 3 clusters não-vazios — achado honesto, registrado) sobre
utilização + testes glicêmicos + medicação — **sem idade, sem
diagnóstico** — produz um fenótipo (6.091 pacientes) com quase o dobro
da readmissão em 30 dias (8,75% vs. 3,97%/4,64%). Caracterizado: não é
o grupo com mais insulina/mudança de medicação (esse é outro fenótipo,
risco intermediário) — é o grupo com utilização PRÉVIA alta
(outpatient 2,24 vs. 0,13-0,23; emergência 0,74 vs. 0,07-0,15;
internação prévia 1,85 vs. 0,30-0,48) — consistente com a literatura
de predição de readmissão, onde utilização prévia é um dos preditores
mais fortes conhecidos.

**Esse achado não é absoluto — depende da representação, e isso foi
testado, não só afirmado.** Adicionamos um quarto domínio,
`DiagnosisCategoryDomain` — 9 categorias ICD-9 (diabetes, circulatório,
respiratório, digestivo, lesão, musculoesquelético, geniturinário,
neoplasias, outro), agrupamento padrão usado pelo próprio artigo
original da base (Strack et al., 2014), extraído de `diag_1`/`diag_2`/
`diag_3`. Com esse domínio incluído (`include_diagnosis_category=True`,
agora o default de `load_uci_diabetes_cohort`), a fenotipagem K-Means
muda de estrutura: em vez de organizar principalmente por utilização
prévia, passa a isolar um grupo minúsculo (212 pacientes, 0,30% da
coorte) de internação extremamente longa (13,41 dias médios vs.
4,22-4,57 dos demais) — e a associação com readmissão em 30 dias fica
bem mais fraca (razão 1,49x, não mais ~2,2x).

Nenhuma das duas representações está "errada" — capturam estruturas
diferentes do mesmo dado real. É a mesma tese central do artigo
"Representation Before Inference" (adicionar uma variável à
representação não é neutro), agora numa instância real e não
planejada, dentro de uma única fonte de dados, não entre duas fontes
diferentes. As duas representações continuam disponíveis lado a lado
(`include_diagnosis_category=False` reproduz exatamente o achado
original) — nenhuma foi descartada em favor da outra.

## 6. Estabilidade, curvatura, dinâmica, representação e GNN: a mesma metodologia de SAOS, aplicada às duas fontes novas

Depois da publicação do artigo de diabetes, apliquei ao NHANES e à UCI
exatamente a bateria de análise já usada em SAOS — não porque fosse
óbvio o que ia dar, mas porque não tínhamos testado ainda.

### NHANES: o oposto exato de SAOS na estabilidade fenotípica

Em SAOS, nenhuma das 28 configurações testadas cruzava o limiar de
estabilidade (ARI≥0,7). No NHANES, com idade incluída, K-Means chega a
ARI=0,957 em K=2 e fica acima de 0,7 até K=7 — quase todas as
configurações estáveis.

**Investigado antes de aceitar isso**: será que é só idade disfarçada
de estrutura metabólica (idade correlaciona com quase tudo numa
população real — mais velho, pior em cada eixo)? Removi a Feature
`idade` e repeti. **Correção registrada honestamente**: uma primeira
verificação com apenas 3 seeds sugeriu que K=3 sempre desestabilizava
sem idade — uma reexecução independente, mais tarde, contradisse isso
(uma das mesmas 3 seeds deu ARI=0,953, estável). Investigado com 20
seeds antes de corrigir a alegação. O achado real, mais preciso: em
**K=2**, a estabilidade é robusta sem idade em 20/20 seeds (mínimo
0,871) — há estrutura metabólica genuína além da idade. Em **K=3**, o
padrão não é "sempre instável" — é MUITO mais **variável**: com idade,
ARI fica em 0,938±0,018 (quase sem variância, nunca abaixo de 0,7 em
20 seeds); sem idade, cai para 0,533±0,195 (desvio 10x maior), com 18
de 20 seeds abaixo do limiar de estabilidade. A idade não "sustenta"
a partição em K=3 de forma absoluta — torna-a muito mais reproduzível.

### NHANES: curvatura estrutural — achado negativo, e faz sentido

Em SAOS, arestas que cruzam fenótipos diferentes têm curvatura mais
negativa que arestas dentro do mesmo fenótipo (p=5,7e-19). No NHANES,
a mesma comparação (K=2, amostra de 1.500) **não é significativa**
(p=0,36). Interpretação: a assinatura de curvatura parece ser
específica de fronteiras estruturalmente frágeis num contínuo mal
separado — quando o fenótipo já é bem separado e estável (como no
NHANES, ao contrário de SAOS), sobra pouca "tensão estrutural" na
fronteira para a curvatura detectar.

### UCI: primeira dinâmica ajustada numa trajetória real fora de sleep/sintético

16.773 pacientes com ≥2 encontros — `MeanRevertingEvolutionOperator`
ajustado neles pela primeira vez em dado genuinamente longitudinal,
não sintético nem transversal. Resultado: 13/13 Features estáveis
(globalmente estável). A mais perto do limite de instabilidade
(`utilization.number_emergency`, φ=0,98) foi testada no mesmo
diagnóstico de robustez (`check_feature_stability_robustness`) que
expôs o artefato de outlier em SAOS — aqui, ao contrário, a conclusão
é **robusta** (não muda removendo nenhum dos 40 pacientes mais
extremos). Consistente com um fenômeno real e bem documentado em
pesquisa de serviços de saúde ("frequent flyers" em uso de
emergência), não um artefato de amostra.

### NHANES: autoencoder vence PCA em dim=2 — a primeira vez neste projeto

Em SAOS (n=355), PCA venceu em toda dimensão testada. No NHANES
(n=9.232), o autoencoder vence em dim=2, robusto em 9 configurações (3
tamanhos de camada oculta × 3 seeds) — a margem aumenta com mais
capacidade (até 0,06 vs. 0,102 de erro com hidden_dim=32). Confirma
diretamente a hipótese de tamanho de amostra: com dado suficiente, o
método não linear encontra solução melhor que a ótima linear do PCA.
Em dim=5 e dim=8, porém, PCA volta a vencer — o efeito é específico de
dimensão baixa, não universal.

### NHANES: o grafo atrapalha até com poucos rótulos — o oposto de SAOS

Em SAOS, com 5% de rótulos, o grafo ajudava muito (+17,8pp). No
NHANES, o grafo atrapalha (ou não ajuda) de 5% a 50% de rótulos.
Interpretação, coerente com os dois achados anteriores desta seção: os
fenótipos aqui já são tão bem separados (alta estabilidade, curvatura
que não discrimina fronteira) que a classificação por Features sozinha
já satura, sobrando pouco espaço para a estrutura relacional ajudar.
Em frações extremas (~1,5%, n=22), há sinal de que o grafo começa a
ajudar, mas a margem é pequena demais (+0,009) para afirmar com
confiança — registrado como achado possível, não conclusivo.

### UCI: autoencoder vence PCA em TODA dimensão testada — mais forte que no NHANES

Réplica da metodologia de autoencoder vs. PCA, agora na UCI (n=71.518,
representação canônica de 3 domínios). Diferente do NHANES (onde o
autoencoder só vencia em dim=2), na UCI ele vence em 2, 5 e 8 —
plausível dado que a UCI tem mais estrutura categórica/ordinal
(contagens de utilização, flags de medicação) onde relações não
lineares têm mais o que capturar que os dados majoritariamente
contínuos do NHANES.

**Correção de uma hipótese inicial errada, registrada honestamente**:
uma primeira exploração (dim=8, hidden_dim=8 por padrão) sugeriu que
"PCA vence em dimensão alta" — falso. O achado real: o cruzamento não
é sobre a dimensão do embedding, é sobre se `hidden_dim` (a camada
oculta do autoencoder) tem **folga** sobre `embedding_dim`. Com
hidden_dim=embedding_dim (sem folga), PCA vence de forma robusta; com
hidden_dim=2× embedding_dim (folga adequada), autoencoder vence — na
MESMA dimensão-alvo (8), conclusão oposta, dependendo só da
capacidade da camada oculta.

### UCI: o grafo atrapalha em toda fração testada, sem cruzamento detectado

Diferente do NHANES (onde havia um sinal ambíguo de cruzamento em
~1,5% de rótulos) e muito diferente de SAOS (onde o grafo ajudava
muito, +17,8pp em 5%), na UCI o grafo atrapalha de 50% até 1,5% de
rótulos, sem cruzamento detectado no intervalo testado (replicado
numa exploração interativa adicional até 1%, mesmo padrão, margem
dentro do ruído). Os fenótipos da UCI (utilização hospitalar +
medicação) parecem ainda mais dominados por sinal pontual (baseado em
Features) que os do NHANES.

### UCI: o fenótipo de alto risco não mostra disparidade demográfica relevante

Primeira análise de equidade do projeto. Estendido o loader com
`include_demographics=True` (raça, gênero, faixa etária como
**metadado**, não Feature — deliberado: a pergunta é se o fenótipo já
calculado por utilização/medicação distribui diferente por grupo, não
misturar demografia na própria fenotipagem, o que só re-rotularia a
disparidade em vez de revelá-la). `race="?"` (marcador de ausência
real do dataset) e `gender="Unknown/Invalid"` tratados como ausência,
não uma terceira categoria; usa o primeiro valor não-ausente por
paciente (1,5% dos pacientes multi-encontro têm `race` inconsistente
entre encontros, desprezível em `gender`).

Rodado sobre a base completa (71.518 pacientes, necessário para poder
estatístico nos grupos raciais menores — Asian, n=641 no total).
**Composição** do fenótipo de alto risco (kmeans_3) por raça e gênero:
efeito desprezível em ambos (Cramér's V=0,031 e 0,009 respectivamente,
bem abaixo do limiar de "pequeno efeito" de Cohen, 0,1) — apesar do
qui-quadrado dar "significativo" (p=3,78e-25 pra raça), um caso
clássico de significância estatística sem significância prática
quando n é grande. **Desfecho DENTRO do fenótipo**: taxa de
readmissão precoce não difere significativamente por raça (p=0,466)
nem por gênero (p=0,176) entre pacientes já classificados no mesmo
fenótipo. Faixa etária mostra efeito um pouco maior (V=0,065) mas
ainda modesto — plausivelmente reflete utilização hospitalar prévia
genuinamente maior em pacientes mais velhos, não viés demográfico.

### NHANES: confundimento por indicação real na adoção de insulina, e um limite estrutural honesto do módulo causal

Primeira aplicação do módulo causal (`check_baseline_balance`,
`match_on_propensity`, `estimate_matched_effect`) fora de SAOS.
Restrito a adultos com diagnóstico autorreferido de diabetes (n=1.420):
quem usa insulina tem HbA1c de linha de base muito mais alto que quem
não usa (SMD=+0,619, o maior desequilíbrio entre 15 Features) —
confundimento por indicação clássico, insulina prescrita pra diabetes
mais difícil de controlar, não atribuída ao acaso. Isso só funcionou
porque uma correção anterior (documentada na própria docstring de
`causal.balance._collect_baseline`) excluiu a própria Feature de
tratamento do baseline — sem essa correção, dado transversal como o
NHANES mascararia o desequilíbrio como SMD=0 trivial.

Pareamento por propensão funcionou perfeitamente: 369 de 413 pacientes
pareados, desequilíbrio zerado nas 15 Features (incluindo o HbA1c que
estava severamente desbalanceado). Mas `estimate_matched_effect`
**recusou corretamente** ao tentar estimar o efeito — o método calcula
diferença-em-diferença (último exame menos primeiro exame), exigindo
≥2 exames por paciente; NHANES é transversal, todo paciente tem
exatamente 1. Documentado como limite estrutural real, não bug: as
duas etapas do módulo causal têm exigências de dado diferentes —
balanceamento/pareamento funcionam em dado transversal, estimativa de
efeito exige longitudinal. O módulo recusa em vez de devolver um
número degenerado que pareceria válido sem ser.

Ver `tests/test_nhanes_real_data.py` (estabilidade, curvatura,
autoencoder, GNN, causal) e `tests/test_uci_diabetes_real_data.py`
(dinâmica, autoencoder, GNN, equidade) — os achados desta seção viraram testes
de regressão nos arquivos por-fonte já existentes, não um arquivo
consolidado à parte.

## 7. Onde os achados foram publicados

- **Artigo**: "Achados Empíricos sobre Diabetes Mellitus Tipo 2 em Duas
  Fontes de Dados Reais e Independentes" (PDF) — os 4 achados clínicos
  centrais das Seções 4 e 5, com tabelas de confusão e caracterização
  completa dos fenótipos.
- **Dashboards**: `biospace_dashboard_nhanes/` e
  `biospace_dashboard_uci/` — exploração interativa das duas fontes
  (ver READMEs de cada um).
- **Testes**: `tests/test_nhanes_real_data.py` (6 testes),
  `tests/test_uci_diabetes_real_data.py` (6 testes),
  `tests/test_uci_diabetes_domains.py` (6 testes, dado fabricado, roda
  em CI), `tests/test_metabolic_package_genericity.py`,
  `tests/test_physiological_process.py`,
  `tests/test_derived_variable.py`, `tests/test_process_coherence.py`,
  `tests/test_metabolic_real_data_dynamics_and_structure.py` (6
  testes — estabilidade/dependência de idade, curvatura, dinâmica de
  trajetória real).

## 8. O que ficou de fora, deliberadamente

- **`DomainPackage` como interface abstrata** para replicar este
  padrão em outros sistemas do corpo (respiratório, neurológico,
  hematológico, ...) — mudaria a interface de todo domínio já
  existente, grande demais para implementar especulativamente.
- **Processo fisiológico anotado em `utilization`/`medication_intensity`**
  (UCI) — só `glycemic_testing` foi anotado; anotar os outros sem
  confiança mecanística real por trás seria rótulo vazio.
- **Roteiro MIMIC-IV** (a terceira base do roteiro original, para
  validar dinâmica de atualização de representação ao longo de uma
  internação) — não tentado nesta rodada.
- **Fusão ou comparação direta entre as representações NHANES e UCI**
  — deliberadamente evitada; a Seção 5 explica por que isso seria
  metodologicamente impróprio dado o quanto as duas fontes diferem
  estruturalmente.

Itens que estavam nesta lista em versões anteriores e já foram
concluídos (não removidos da história, só movidos): creatinina/eGFR e
perfil lipídico completo do NHANES (Seção 6, síndrome metabólica de 5
critérios), autoencoder/GNN em NHANES (Seção 6), diagnósticos ICD-9 da
UCI e medicação de diabetes do NHANES (`DIQ050`/insulina) — ambos
também na Seção 6.

## 9. Um functor entre duas categorias de representação — o item que o manuscrito chamava de "trabalho futuro"

A organização categórica do meta-modelo (cada representação como uma
categoria de um único objeto, operadores admissíveis como
endomorfismos) deixava em aberto a existência de um functor entre duas
categorias diferentes (sleep, metabolic) — nomeado explicitamente como
"não demonstrado" em versões anteriores deste projeto.

Formalização precisa, não decorativa: um functor entre duas categorias
de um único objeto é exatamente um **homomorfismo de monoide**. O
candidato testado: `MeanRevertingEvolutionOperator.predict(x, Δt)`
(já usado, com dado real, em SAOS e na UCI) satisfaz a lei de
semigrupo de um fluxo a um parâmetro, T_{s+t} = T_s∘T_t — verificada
algebricamente por substituição direta na fórmula (μ+φ^dt·(x-μ)), e
depois testada numericamente contra dado real das DUAS categorias, não
apenas uma: 296 pacientes multi-encontro reais de SAOS, e uma amostra
multi-encontro real da UCI. A mesma lei se sustenta identicamente nos
dois objetos — o conteúdo funtorial concreto é que a MESMA receita
(fit + predict) realiza a mesma estrutura algébrica em ambas as
categorias, não apenas em uma escolhida a dedo.

Escopo declarado com precisão, não inflado: isto testa a lei de
semigrupo do componente determinístico de `predict()` (a versão
estocástica `sample()` não satisfaz a lei exatamente, por conter
ruído aleatório), e testa dois objetos concretos (sleep, metabolic) —
não é uma prova geral de que todo par de categorias futuras deste
projeto terá um functor natural entre si.

Ver `tests/test_functor_between_categories.py` (4 testes: identidade
algébrica geral, identidade em ambas as categorias, e a lei de
semigrupo testada contra dado real de SAOS e da UCI separadamente).

**Uma segunda realização, mais direta, achada só numa auditoria
posterior do projeto** (`tests/test_cross_disease_functor.py`, 5
testes, já existente antes desta seção ser escrita — não vista até a
auditoria da Seção 10): `project_to_process_space(representation,
vector)`, em `biospace.core`, é o mapeamento de OBJETOS de um functor
de projeção — leva a representação de QUALQUER doença/plugin para um
dicionário `{nome_do_processo: valor agregado}`, indexado por
`PhysiologicalProcess`, não por domínio específico de uma doença.
Demonstração real e decisiva: um paciente SAOS real e um paciente
NHANES real, projetados no mesmo processo compartilhado
(`cardiovascular_regulation`), numa escala comparável — apesar de
**zero variáveis brutas em comum** entre as duas representações. Testa
também uma propriedade de naturalidade (atualizar um domínio não
relacionado a um processo não muda a projeção desse processo, ecoando
`check_domain_update_independence`). As duas realizações são
complementares, não conflitantes: a primeira prova uma lei algébrica
sobre operadores (functor entre monoides de endomorfismos); a segunda
constrói um mapeamento concreto de objetos entre representações de
doenças diferentes rumo a um alvo compartilhado — mais perto do que o
manuscrito pedia originalmente ("uma mapa estrutural da categoria
sleep para a categoria metabolic").

## 10. Análise de sobrevivência (Kaplan-Meier/Cox) — nuancia o achado de readmissão publicado, não o contradiz

Novo módulo genérico, não específico de doença:
`biospace.survival` (`build_discrete_time_to_event`, `kaplan_meier_by_group`,
`fit_cox_model`, sobre `lifelines`). "Tempo" é sempre um índice ordinal
(posição na sequência de encontros), nunca calendário real — a UCI não
tem datas reais, só ordem. A primeira observação de cada paciente é
usada só como fonte de covariável de baseline, nunca conta como tempo
em risco — consistente com o Contrato de Temporalidade (5.7).

**A pergunta diferente que isso permite fazer**: o achado publicado no
artigo de diabetes (fenótipo → ~2x readmissão) é uma caracterização
**transversal** — o fenótipo é ajustado sobre o estado agregado/mais
recente de cada paciente, depois comparado com a readmissão nesse
mesmo estado. Isso é uma tarefa estatística genuinamente mais fácil que
predição **prospectiva**: consigo prever risco futuro usando só o que
sabia no **primeiro** encontro do paciente, antes de qualquer encontro
subsequente?

**Achado real**: testado em dado real (16.773 pacientes multi-encontro,
fenotipados só pelo primeiro encontro — sem olhar pro futuro). O
log-rank ainda é estatisticamente significativo (p<0,01), mas o efeito
prático é modesto — mediana de sobrevivência varia só entre 4 e 5
encontros entre os grupos, e o índice de concordância do modelo de Cox
fica próximo de 0,5 (quase aleatório, C-index≈0,52). Investigado antes
de aceitar como limite do método: 75% dos pacientes multi-encontro têm
só 1-2 encontros de acompanhamento após o baseline — pouca variação
temporal disponível para qualquer modelo discriminar, um limite real do
dado (poucos encontros de acompanhamento), não do método de
sobrevivência em si.

**Isto não invalida o achado publicado** — os dois são caracterizações
válidas de perguntas diferentes: "o estado atual de um paciente associa
com desfecho atual?" (achado publicado, mais forte) vs. "o estado no
primeiro contato prevê desfecho futuro?" (achado desta seção, mais
fraco, e mais difícil de responder bem com o volume de acompanhamento
disponível nesta base). Reportado como um limite honesto de predição
prospectiva com este dado específico, exatamente o tipo de nuance que
a tese central do projeto (Representação Antes da Inferência) prevê:
a MESMA representação, sob duas formulações estatísticas diferentes da
mesma pergunta clínica, produz forças de evidência diferentes.

Ver `tests/test_survival.py` (8 testes: extração de tempo-até-evento
com dado fabricado incluindo um controle negativo, e o achado real
contra a UCI completa).

## 11. Sinais de alerta precoce (critical slowing down) — uma duplicação real encontrada e corrigida numa auditoria

**Correção registrada com transparência**: esta seção descrevia
originalmente um módulo (`biospace.dynamics.early_warning`,
`compute_early_warning_indicators`) construído sem saber que uma
implementação muito mais completa e rigorosa já existia desde 6 de
julho: `biospace.early_warning.CriticalSlowingDownDetector`. Achado
numa auditoria do projeto e removido — não mantido como alternativa
"mais simples". A versão correta tem 3 indicadores (variância,
autocorrelação, **assimetria** — Guttal & Jayaprakash, 2008), teste de
**tendência** de Kendall sobre janela móvel de verdade (não um valor
único estático, que é o que a versão removida calculava), significância
por dados substitutos AR(1) (Dakos et al., 2012), detrend linear ou
gaussiano, janela por contagem de pontos ou por tempo decorrido, e
participa da hierarquia formal de `Operator` do projeto
(`EarlyWarningOperator`).

**Validado contra uma bifurcação sela-nó genuína**, não um AR(1) com φ
manipulado artificialmente — dx/dt = r(t) + x², ramo estável em
x=-√(-r) (Strogatz), o teste padrão da própria literatura de CSD.
**Erro real cometido e corrigido durante a validação**: a primeira
tentativa usou o sinal errado da fórmula e o ramo instável como ponto
de partida, causando explosão numérica imediata — não um bug no
detector, um bug na minha simulação. Corrigido, revalidado: trajetórias
genuinamente se aproximando da bifurcação mostram mais casos de
`warning=True` (4/15) e τ de autocorrelação médio maior (0,216 vs.
0,026) que trajetórias de controle com parâmetro constante, longe da
bifurcação.

**Aplicado à UCI real**: o detector completo exige `min_points=8` por
padrão — só 320 de 71.518 pacientes (0,45%) qualificam. Restringindo
a um holdout genuíno (detecção nos 8 primeiros encontros, checagem de
evento nos encontros 9+, sem vazar futuro — mesma disciplina de
`biospace.survival`), sobram 139 pacientes (0,19%), dos quais **apenas
1** mostra `warning=True` — poder estatístico insuficiente para
qualquer teste de associação com desfecho futuro. Isto **reforça**,
com um método ainda mais exigente e rigoroso que o removido, a mesma
conclusão documentada antes: a limitação é do dado (poucos encontros
de acompanhamento por paciente nesta base), não da escolha de método.

Ver `tests/test_critical_slowing_down.py` (5 testes: discriminação
contra bifurcação real, casos de borda, a propriedade de maioria de
evidência isolada do resto do pipeline, e o achado real — severamente
subamostrado — contra a UCI completa).

## 12. Aprendizado supervisionado clássico — triangulação, não coincidência

Achado numa auditoria do projeto: `biospace.prediction`
(`Predictor`/`SklearnPredictor`, um envelope genérico para qualquer
estimador compatível com a API sklearn) já existia, arquiteturalmente
completo, mas com **zero testes e zero aplicação a dado nenhum** — 7
testes escritos agora (`tests/test_prediction.py`), incluindo
recuperação de rótulo em dado linearmente separável com verdade
conhecida, tratamento de rótulo faltando, e generalização a um
`RepresentationSpace` diferente do usado no ajuste.

**A pergunta**: RandomForest e LogisticRegression, treinados sobre
*todas* as Features do 1º encontro (mesmo desenho de baseline da Seção
10 — sem olhar pro futuro), predizem readmissão precoce subsequente
melhor que o modelo de Cox baseado em fenótipo (C-index≈0,52)?

**Achado real, triangulado com as Seções 10 e 11, não um achado
isolado**: não — validado com validação cruzada de 5 folds,
RandomForest fica em AUC≈0,50-0,52 (baixíssimo desvio entre folds:
±0,01-0,02), LogisticRegression em ≈0,53. As **três** abordagens
completamente diferentes (fenotipagem+Cox, RandomForest,
LogisticRegression) convergem para essencialmente a mesma conclusão —
evidência forte de que a limitação é do **dado** (o primeiro encontro,
sozinho, não carrega informação suficiente para prever o futuro
específico deste desfecho), não da escolha de método. Um único método
fraco poderia ser explicado por uma escolha de modelo ruim; três
métodos completamente diferentes convergindo no mesmo teto é evidência
mais forte que qualquer um isoladamente.

Ver `tests/test_prediction.py` (7 testes: 6 de infraestrutura com dado
fabricado, 1 com o achado real triangulado contra a UCI completa).

## 13. `risk` e `latent`: mais dois módulos achados numa auditoria, sem teste nem aplicação

Mesmo padrão de `prediction/`, `early_warning/`: dois módulos
arquiteturalmente completos, zero testes, zero aplicação.

**`biospace.risk` (`RiskOperator`/`LinearRiskOperator`)**: generaliza
uma fórmula ad-hoc que existia hardcoded num script legado do plugin
sleep (`07_clusterizacao.py`, já não existe mais no projeto — score =
soma ponderada de Features nomeadas, de qualquer domínio) como pesos
explícitos e auditáveis. 8 testes (`tests/test_risk.py`), incluindo um
real: um score de risco (ido + spo2_minima invertido + carga
hipóxica, pesos iguais) aplicado ao SAOS real associa
significativamente (p=0,02) com status de tratamento (CPAP/AAM) —
pacientes tratados têm score mais alto, consistente com a expectativa
clínica de que tratamento é direcionado a casos mais graves.
**Verificação de sinal feita antes de confiar no resultado**: o
z-score de `spo2_minima` para um paciente específico deu negativo
(-0,471) para um valor bruto de 88 — checagem inicial sugeria erro de
convenção, mas a média populacional real desta coorte é 86,74 (raw=88
está *acima* da média, ou seja, melhor que o típico nesta população
referida para exame de sono) — o sinal estava correto, era minha
leitura inicial do número bruto que estava errada.

**`biospace.latent.FactorAnalysisLatentDomain`**: extração de fator
latente por Análise Fatorial de verdade (não combinação linear
arbitrária), conectado à mesma base `LatentDomain` já usada por
`InsulinResistanceProxyDomain`. Validado primeiro contra um fator
latente sintético CONHECIDO (correlação recuperada >0,85, não um
artefato do método) antes de qualquer aplicação real. Aplicado ao
NHANES real: um fator "carga metabólica" extraído de
glycemic+cardiovascular+anthropometric é **dominado por adiposidade**
(carga de circunferência abdominal +0,81, IMC +0,74 — muito maiores
que HbA1c +0,19 ou pressão diastólica +0,16), não distribuído
igualmente entre os 3 domínios como se poderia assumir ingenuamente —
a Análise Fatorial encontra o eixo de variância COMPARTILHADA mais
forte, que aqui é predominantemente adiposidade. Mesmo assim, mostra
gradiente forte e sensato com status de diabetes (não usado na
construção do fator): normal < pré-diabetes < diabetes,
Kruskal-Wallis p=4,64e-162 — validação externa real, não circular.

Ver `tests/test_risk.py` (8 testes) e
`tests/test_latent_factor_analysis.py` (6 testes: fator latente
sintético conhecido, casos de borda, achado real contra o NHANES).

## 14. `biospace.longitudinal`: sexta vez, mesmo padrão

`SurvivalAnalyzer`/`SurvivalOperator`, `TransitionAnalyzer`/
`TransitionOperator`, `TrajectoryUpdater` — os 3 arquivos do pacote
inteiro, sem NENHUM teste antes desta rodada. Mesmo padrão de
`prediction/`, `early_warning/`, `risk/`, `latent/`.

**`TrajectoryUpdater`** (5 testes): formaliza `Cohort.update()` +
validação de ordem monotônica de tempo — recusa uma nova observação
com timestamp anterior ao último exame já registrado, a menos que
`enforce_monotonic_time=False` seja passado explicitamente.

**`TransitionAnalyzer`** (6 testes): matriz de transição entre
fenótipos com filtro de intervalo de tempo (`min_gap`/`max_gap`) e
`time_to_transition()`/`summary()` — quanto tempo cada tipo de
transição observada de fato levou, não só se ocorreu. Testado com
verdade conhecida (1 paciente muda de fenótipo em 10 dias, 1
permanece estável — P(alto|baixo)=0,5 exatamente, gap=10 dias
exatamente).

**`SurvivalAnalyzer`** (5 testes) — o achado mais valioso desta
seção: é uma implementação PRÓPRIA de Kaplan-Meier (sem `lifelines`,
"consistente com a filosofia do projeto de manter o núcleo
autocontido"), nunca antes validada contra nada. **Validação cruzada
contra `lifelines`** (já usado em `biospace.survival`) sobre o mesmo
dado de duration/event: concordância exata até 1e-9 em 5 pontos de
tempo diferentes — confirmação forte de que a implementação própria
está matematicamente correta, não coincidência.

Ver `tests/test_trajectory_updater.py`, `tests/test_transition_analyzer.py`,
`tests/test_longitudinal_survival.py`.

## 15. Coortes automáticas por proximidade geométrica — a aplicação que a própria lista de IA chamava de "mais original"

`Geometry.neighborhood(space, system_id, radius)` (§7.6 da teoria
formal) já existia, mas sem NENHUM teste — mesmo padrão de todas as
seções anteriores desta auditoria. Adicionado `Geometry.k_nearest()`
(complementar: k fixo sempre devolve k vizinhos, raio fixo devolve um
número variável dependendo da densidade local) e um módulo novo,
`biospace.geometry.cohort_query` (`cohort_around`/`GeometricCohort`) —
a peça que faltava para "coortes deixam de ser consultas SQL e passam
a ser subconjuntos geométricos": consulta por um PONTO ARBITRÁRIO
(não precisa ser um paciente real — ex.: o centroide de um fenótipo,
um perfil clínico hipotético), não só por um paciente existente.

**Achado real, contraintuitivo, na UCI**: uma coorte geométrica
definida como "k mais próximos do CENTROIDE do fenótipo de alto risco"
(kmeans_3, achado publicado) — com o MESMO tamanho k — recupera só
~20% da mesma população que a associação de cluster original
(Jaccard≈0,11). Faz sentido matematicamente, investigado antes de
aceitar como surpresa incômoda: K-Means particiona por Voronoi entre
os 4 centroides SIMULTANEAMENTE (mais perto DESTE centroide que de
qualquer outro), enquanto "k mais próximos de um centroide" só
considera distância a ESSE ponto de referência, ignorando os outros
3. São mecanismos de definição de coorte genuinamente diferentes, não
aproximações um do outro — achado que dá conteúdo real à alegação da
lista de que "coortes geométricas" seriam algo distinto de clustering,
não apenas outro nome para a mesma coisa.

Mesmo com a divergência grande, a coorte geométrica ainda captura
sinal real: taxa de readmissão ~1,5x a linha de base populacional
(4,51%), mais fraca que o cluster completo (~1,9x), mas claramente não
nula.

Ver `tests/test_geometric_neighborhood.py` (10 testes: `neighborhood()`
e `k_nearest()`, fronteira exata calculada à mão) e
`tests/test_geometric_cohort.py` (7 testes: consulta por paciente
existente e por vetor arbitrário, tratamento de erro, e o achado real
contra a UCI completa).

## 16. Explicabilidade (SHAP) — mais uma peça da triangulação, vista por outro ângulo

Módulo novo, `biospace.explainability` (`explain_predictor`) —
envelope de SHAP sobre `SklearnPredictor` já ajustado, traduzindo
índice de coluna numérico pra `domínio.feature` legível (usa
`TreeExplainer` exato para modelos de árvore, `KernelExplainer`
aproximado para os demais). Validado primeiro contra um cenário com
verdade conhecida (uma Feature determina o rótulo por construção,
outra é ruído puro) — a Feature verdadeira recebe |SHAP| médio >5x
maior que o ruído, antes de qualquer aplicação real.

**Achado real, que completa a triangulação das Seções 10/11/12**:
aplicado ao RandomForest real já treinado pra prever readmissão
futura na UCI (AUC≈0,50-0,52, achado documentado como perto do
acaso), o SHAP mostra a importância **difusa**, não concentrada — a
maior |SHAP| médio (`num_procedures`, 0,0096) e a menor
(`max_glu_serum_ordinal`, 0,0006) ficam numa faixa estreita e baixa,
sem nenhuma Feature dominando as outras (razão <20x, contra >5x no
cenário sintético com verdade conhecida). Isso responde uma pergunta
que o AUC baixo sozinho não respondia: não é que existe sinal forte
numa Feature específica que o modelo está deixando passar — a
ausência de sinal está distribuída por igual entre as 13 Features
disponíveis no primeiro encontro. Três métodos diferentes (Cox,
RandomForest, LogisticRegression) já convergiam pro mesmo teto; agora
uma quarta lente (explicabilidade, não desempenho agregado) mostra
por que, num nível mais fino.

Ver `tests/test_explainability.py` (6 testes: identificação correta
da Feature verdadeiramente importante em cenário sintético, tratamento
de erro, KernelExplainer para modelos não-árvore, e o achado real de
importância difusa contra a UCI completa).
