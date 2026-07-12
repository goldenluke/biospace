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
os mesmos 6 domínios físiológicos, nenhum deles sabendo o que é
diabetes. `plugins/diabetes/` virou uma camada fina: reexporta tudo de
`metabolic` sob os nomes antigos (`DiabetesSystem is MetabolicSystem`,
literalmente a mesma classe, testado) e contribui só o que é
genuinamente específico — o critério clínico de diagnóstico e o
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
(dinâmica) — os achados desta seção viraram testes de regressão nos
arquivos por-fonte já existentes, não um arquivo consolidado à parte.

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
- **Dado real ainda não extraído**: creatinina real do NHANES (arquivo
  `BIOPRO_J`, fecharia o `RenalDomain` com dado de verdade); perfil
  lipídico do NHANES (`TCHOL_J`/`HDL_J`/`TRIGLY_J`, fecharia os 5
  critérios completos de síndrome metabólica); raça/gênero da UCI
  (análise de equidade — o fenótipo de alto risco de readmissão
  distribui igual entre grupos demográficos?). Códigos de diagnóstico
  ICD-9 da UCI e medicação de diabetes do NHANES (`DIQ050`/insulina)
  foram extraídos e analisados — ver Seção 10.
- **Autoencoder vs. PCA e GNN semi-supervisionado** nas duas fontes
  novas — testados em SAOS, ainda não em NHANES/UCI.
