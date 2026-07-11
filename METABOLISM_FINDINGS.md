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
  autorreferido: sensibilidade 66,6%, especificidade 95,6%, acurácia
  91,0% — consistente com subdiagnóstico de diabetes, fenômeno
  documentado na literatura.
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

## 6. Onde os achados foram publicados

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
  `tests/test_derived_variable.py`, `tests/test_process_coherence.py`.

## 7. O que ficou de fora, deliberadamente

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
