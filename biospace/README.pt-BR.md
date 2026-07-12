# BioSpace

*[Read in English](README.md)*

Infraestrutura computacional para representação de sistemas biológicos —
não uma biblioteca de algoritmos de medicina. O papel que o NumPy tem
para computação numérica, ou o scikit-learn para aprendizado de máquina:
uma camada fundamental sobre a qual plugins de doença específicos (hoje,
só SAOS) são construídos.

Implementa o meta-modelo M = (B, O, D, R, X, G, Γ, F, C):

| Símbolo | Entidade | Classe |
|---|---|---|
| B | Sistema biológico | `BiologicalSystem` |
| O | Observação | `Observation`, `Observable`, `Measurement` |
| D | Domínio semântico | `SemanticDomain`, `LatentDomain` |
| R | Representação | `Representation`, `RepresentationVector` |
| X | Espaço de representação | `RepresentationSpace` |
| G | Geometria | `Geometry`, `TrajectoryGeometry`, `DynamicGeometry` |
| Γ | Trajetória | `Trajectory` |
| F | Fenótipo | `Phenotype` |
| C | Coorte | `Cohort` |

**Para o histórico completo de como cada peça foi construída — achados
reais, bugs encontrados e corrigidos, decisões de design e por quê —
veja [HISTORY.md](HISTORY.md).** Este arquivo é só referência de uso.

## Camada opcional: PhysiologicalProcess

Estendendo o meta-modelo com uma camada ADITIVA, nunca obrigatória,
levantada em revisão externa: "diabetes não é um conjunto de exames, é
um conjunto de processos biológicos — as observações medem esses
processos". `Observable.process` (opcional, default `None`) nomeia o
mecanismo biológico que uma grandeza mede — ex.: HbA1c mede
`"glucose_homeostasis"` — independente de qual `SemanticDomain` a
consome. `SemanticDomain.processes()` e `Representation.processes()`/
`features_by_process()` são consultas COMPUTADAS a partir dessa
anotação, nunca exigem que um domínio existente mude uma linha.

Retrocompatibilidade testada, não apenas assumida: o plugin sleep
(real, validado) nunca foi tocado por esta mudança —
`SleepRepresentation().processes()` continua vazio
(`tests/test_physiological_process.py::test_sleep_plugin_has_no_processes_declared_proving_layer_is_optional`).
O plugin metabolic anota 5 processos reais como exemplo concreto
(`glucose_homeostasis`, `body_composition`, `cardiovascular_regulation`,
`renal_filtration`, `lipid_metabolism`), e o teste decisivo
(`test_features_by_process_crosses_domain_boundary`) prova que a
agregação funciona cruzando fronteira de domínio de verdade — duas
Features de domínios DIFERENTES declarando o mesmo processo aparecem
juntas em `features_by_process()`, não só quando processo e domínio
coincidem 1:1 (o cenário motivador: uma medida de colesterol pode
alimentar tanto um domínio metabólico quanto um cardiovascular).

```python
from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, exam

representation = MetabolicRepresentation()
system = MetabolicSystem()
system.observe(exam({"hba1c_pct": 7.5, "glicemia_jejum_mg_dl": 145.0, ...}))
vector = representation.transform(system)

representation.processes()               # {'glucose_homeostasis', 'body_composition', ...}
representation.features_by_process(vector)  # {'glucose_homeostasis': [Feature(hba1c_pct), Feature(glicemia_jejum)], ...}
```

Ver `examples/12_physiological_processes.py` para o exemplo completo.
Variáveis Derivadas e validação de coerência de processo — nomeadas
aqui como pendentes numa rodada anterior — foram implementadas depois;
ver seção seguinte.

## O que ficou de fora na rodada anterior, implementado agora

Três peças nomeadas explicitamente como pendentes, fechadas em rodada
dedicada — todas aditivas, testadas com caso decisivo antes de confiar.

### `DerivedVariable` (`core/derived_variable.py`) — Observação → Trajetória → Variável Derivada

Distinta de Feature (que vem de UM instante): uma variável derivada
precisa da TRAJETÓRIA inteira — não cabe dentro de `SemanticDomain.encode()`,
que só enxerga um instante. Entidade PARALELA, não subclasse.

```python
from biospace.core import augment_with_derived_variables
from biospace.plugins.metabolic import HbA1cSlopeVariable, HbA1cVariabilityVariable, GlycemicBurdenVariable

vetor_aumentado = augment_with_derived_variables(
    trajetoria.latest(), trajetoria, [HbA1cSlopeVariable(), HbA1cVariabilityVariable(), GlycemicBurdenVariable()]
)
# vetor_aumentado.components["derived"] = [Feature(hba1c_slope_per_year=...), ...]
```

3 variáveis concretas no plugin metabolic — `GlycemicBurdenVariable`
reaproveita EXATAMENTE o mecanismo já usado (e validado) em
`plugins/diabetes/synthetic.py` para o declínio renal, não um novo
mecanismo inventado. Testado contra trajetória com resultado calculado
à mão antes de confiar (slope e desvio-padrão de uma progressão linear
conhecida batem com precisão de 1e-3). Casos de borda testados:
trajetória com 1 ponto devolve `None` para slope/variabilidade (nunca
um valor inventado), mas ainda computa burden (que faz sentido com 1
ponto). `augment_with_derived_variables` nunca modifica o vetor
original — devolve cópia.

### `check_process_coherence` (`core/process.py`) — valida a alegação, não só declara

Testa se Features do mesmo `PhysiologicalProcess` correlacionam mais
entre si, através da população, que Features de processos diferentes
(Mann-Whitney). Validado em dois cenários sintéticos com verdade
conhecida antes de confiar: correlação real dentro do processo →
confirma (p=4,9e-06); ruído puro independente → não confirma
(contraprova).

**Achado real ao aplicar no plugin metabolic**: a coerência **não se
confirma** nos dados sintéticos de diabetes já existentes (|r|
mesmo-processo=0,42 vs. diferente-processo=0,43, p=0,65) — investigado,
não escondido: `plugins/diabetes/synthetic.py` sorteia cada variável
INDEPENDENTEMENTE dentro da classe de severidade (uma chamada de
`rng.normal()` por chave, sem fator latente compartilhado) — a
correlação populacional existente vem do rótulo de classe e de efeitos
de tratamento, não de um mecanismo fisiológico real ligando HbA1c e
glicemia além disso. É uma limitação real do GERADOR, não um bug no
contrato novo (que se comporta corretamente nos dois testes sintéticos
de verdade conhecida). Documentado como teste de regressão
(`test_metabolic_synthetic_data_does_not_confirm_coherence`).

### `check_domain_update_independence` (`core/contracts.py`) — conecta com o bug real do `as_of`

"Domínios independentes não sofrem alteração quando recebem observação
não relacionada" — proposto em revisão externa, conectado diretamente
ao bug real já documentado no parâmetro `as_of` (uma teoria de
representação que produz testes formais capazes de detectar erros
reais de implementação, não hipotéticos).

```python
resultado = check_domain_update_independence(
    system, representation, nova_observacao, protected_domain_names=["glycemic", "cardiovascular"]
)
# resultado["glycemic"].is_independent -> True se nenhuma Feature desse domínio mudou
```

Checagem PASSIVA por desenho (o sistema já não deveria mudar Features
de domínios não relacionados, dado como `latest_measurement()` já
funciona) — o valor é DETECTAR REGRESSÃO futura, não corrigir bug
conhecido hoje. Teste decisivo: confirma tanto que domínios não
relacionados ficam intocados quanto que o domínio DE FATO afetado muda
— provando que o contrato discrimina de verdade, e uma varredura
sistemática tocando cada um dos 7 domínios do plugin metabolic, um de
cada vez.

**Ainda deliberadamente não implementado**: `DomainPackage` como
interface abstrata para outros sistemas do corpo, e o roteiro
NHANES/UCI/MIMIC — o primeiro muda a interface de todo domínio já
existente (grande demais para uma rodada), o segundo precisa de dados
externos não disponíveis neste ambiente.

## `biospace.datasets.nhanes` — dados reais chegaram, dois bugs reais encontrados e corrigidos

O usuário enviou os 6 arquivos `.XPT` diretamente (contornando a
restrição de rede). Dois problemas reais apareceram ao testar contra
eles — nenhum dos dois seria pego pelos testes com dado fabricado, que
mockavam a leitura de arquivo inteira:

1. **`pandas.read_sas` exige `format="xport"`, não `format="xpt"`** —
   a versão anterior deste módulo usava o valor errado; todos os 6
   arquivos falhavam com `ValueError: unknown SAS format` até a correção.
2. **O ciclo enviado é "Pre-pandemic" (ago/2017–mar/2020 combinado,
   prefixo `P_`), não 2017-2018 isolado** — a metodologia de pressão
   arterial mudou para oscilométrica: o arquivo é `P_BPXO`, não `P_BPX`,
   e as variáveis são `BPXOSY1`/`BPXODI1` (com "O" extra), não
   `BPXSY1`/`BPXDI1` como o mapeamento anterior assumia. Descoberto
   inspecionando as colunas reais dos arquivos antes de assumir que a
   documentação do ciclo isolado bateria.

**Lição registrada como teste**: `test_read_xpt_works_against_real_files`
(`tests/test_nhanes_real_data.py`) exercita `pandas.read_sas` de
verdade — os testes anteriores (`test_nhanes_loader.py`) só testavam a
lógica de merge/renomeação com DataFrames já carregados, nunca a
leitura de arquivo em si, que é exatamente onde os dois bugs viviam.

### Dois achados reais, validados contra ~9.232 adultos (idade ≥20)

**`classify_diabetes_status` contra diagnóstico autorreferido real**
(critério ADA sobre HbA1c/glicemia vs. resposta à pergunta "um médico
já disse que você tem diabetes?"): sensibilidade 66,6%, especificidade
95,6%, acurácia 91,0%. Clinicamente plausível — subdiagnóstico de
diabetes é um fenômeno real e bem documentado (parte de quem tem
diabetes por critério laboratorial não sabe ou não relata). A maior
parte dos casos classificados como `pre_diabetes` (2.874 de 3.141, 91%)
não tem autorrelato de diabetes — também condizente com a literatura:
pré-diabetes é amplamente subdiagnosticada.

**`check_process_coherence` confirma em dado real** (|r|=0,782
mesmo-processo vs. 0,151 processos-diferentes, p=0,0022) — o oposto
exato do achado no gerador sintético de diabetes (`is_coherent=False`,
Seção "O que ficou de fora..."). Isso não é uma contradição — é a
validação que faltava: o gerador sintético tem uma limitação real
(variáveis sorteadas independentemente, sem processo compartilhado);
população real tem a estrutura fisiológica que o gerador não
reproduziu. O mesmo contrato, aplicado a duas fontes com propriedades
diferentes, deu duas respostas diferentes e corretas — evidência de
que ele discrimina de verdade, não apenas em cenários de brinquedo.

**Achado adicional, sem surpresa mas registrado**: a idade mínima na
amostra é próxima de 0 — NHANES amostra toda a população, inclusive
crianças, por desenho (é uma pesquisa de saúde populacional, não uma
coorte de diabetes). Filtramos para ≥20 anos antes de qualquer
interpretação clínica de diabetes — aplicar critério ADA em bebê não
faz sentido clínico, e isso está documentado explicitamente onde a
filtragem acontece.

Ver `tests/test_nhanes_real_data.py` (6 testes, todos contra os
arquivos reais, `skipif` quando ausentes).

## `biospace.datasets.uci_diabetes` — segunda fonte real, estrutura genuinamente diferente

O usuário enviou `diabetic_data.csv` (Strack et al., 2014 — 101.766
encontros hospitalares, 71.518 pacientes). Inspecionado antes de
qualquer mapeamento (não assumido por analogia ao NHANES): **essa base
NÃO tem HbA1c/glicemia contínuas** (só categorias — `A1Cresult`
83,3% ausente, `max_glu_serum` 94,8% ausente), **não tem IMC,
circunferência abdominal, pressão arterial nem creatinina**. Idade vem
em faixas de 10 anos, não contínua.

**Consequência honesta**: isto não é "a mesma `MetabolicRepresentation`
aplicada a uma segunda fonte" — a estrutura é incompatível com aquele
conjunto de domínios. `biospace/datasets/uci_diabetes.py` define uma
representação GENUINAMENTE DIFERENTE, apropriada ao que esta fonte
realmente mede: `HospitalUtilizationDomain` (8 variáveis de
utilização, 100% completas — o oposto do padrão de completude do
NHANES), `GlycemicTestingDomain` (A1Cresult/max_glu_serum
recodificados ordinalmente, esparsos por desenho), `MedicationIntensityDomain`
(status de insulina, mudança de medicação). **Conexão deliberada com o
NHANES**: `GlycemicTestingDomain` declara `process="glucose_homeostasis"`
— o MESMO nome de processo do NHANES — porque medem o mesmo mecanismo
biológico, só que categorizado e muito mais esparso; registrado
formalmente via `PhysiologicalProcess`, não apenas comentado.

**Achado real não previsto originalmente**: 23% dos pacientes
(16.773 de 71.518) têm MÚLTIPLOS encontros — até 39 no máximo
observado — algo que o NHANES (transversal) não tinha. `encounter_id`
não é uma data real (não existe data no dataset), mas cresce
monotonicamente — usado como proxy de ORDEM cronológica (não
intervalo real, documentado explicitamente) para construir
trajetórias multi-ponto de verdade. `DerivedVariable` foi testada
pela primeira vez numa trajetória real multi-ponto (não sintética,
não transversal) — funciona.

```python
from biospace.datasets.uci_diabetes import load_uci_diabetes_cohort

cohort, representation = load_uci_diabetes_cohort("/caminho/diabetic_data.csv")
```

### O achado mais forte: fenótipo sem idade nem diagnóstico associa com readmissão real

Fenotipagem K-Means (K=4, só 3 clusters não-vazios emergiram — achado
honesto, não escondido) sobre utilização + testes glicêmicos +
medicação — **sem idade, sem código de diagnóstico** — produz um
fenótipo (6.091 pacientes) com quase o dobro da taxa de readmissão em
30 dias (8,75% vs. 3,97%/4,64% dos outros dois). Caracterizado: não é
o grupo com mais insulina/mudança de medicação (esse é outro
fenótipo, com risco intermediário) — é o grupo com utilização PRÉVIA
alta (outpatient 2,24 vs. 0,13–0,23; emergência 0,74 vs. 0,07–0,15;
internação prévia 1,85 vs. 0,30–0,48) — consistente com a literatura
de predição de readmissão, onde utilização prévia é um dos
preditores mais fortes conhecidos. Documentado como teste de
regressão (`test_phenotype_associates_with_real_readmission_outcome`).

### O que não foi testado ainda

`check_process_coherence` não pôde ser aplicado de forma significativa
nesta representação — só um processo (`glucose_homeostasis`) foi
declarado, em apenas 2 Features, sem nenhum outro processo para
comparar (o teste precisa de pares "mesmo processo" E "processo
diferente" para funcionar). Anotar processo também em
`utilization`/`medication_intensity` habilitaria esse teste — não
feito ainda, para não forçar rótulos de processo sem confiança
mecanística real por trás deles.

Ver `tests/test_uci_diabetes_domains.py` (6 testes, dado fabricado,
roda em CI) e `tests/test_uci_diabetes_real_data.py` (6 testes,
`skipif` quando o arquivo real está ausente).

## Instalação

```bash
pip install -r requirements-dev.txt   # numpy, scipy, scikit-learn, pandas, POT, pytest
```

Sem `setup.py`/`pyproject.toml` ainda — use diretamente a partir da raiz
do repositório (`import biospace`) ou adicione ao `PYTHONPATH`.

## CI

`.github/workflows/ci.yml` roda a cada push/PR: suíte de testes,
`examples/*.py`, `check_install.py`, `demo_sleep.py`, e
`scripts/check_dashboard.py` (verifica TODAS as páginas do dashboard via
`AppTest` com dados sintéticos — pega exatamente a classe de erro que já
quebrou um deploy real no Streamlit Cloud: um arquivo desatualizado em
relação a outro). Rode `python3 scripts/check_dashboard.py` localmente
antes de subir mudanças no dashboard.

## Quickstart

```python
from datetime import datetime
from biospace.plugins.sleep import SleepSystem, SleepRepresentation, exam
from biospace.core import Cohort

representation = SleepRepresentation()
cohort = Cohort()

system = SleepSystem(identifier="paciente_1")
system.observe(exam({"ido": 22.0, "idade": 55, "imc": 31.0}, timestamp=datetime(2024, 1, 1)))
cohort.update(system, representation, timestamp=datetime(2024, 1, 1))

space = cohort.snapshot()
matrix, ids = space.matrix()   # pronto para qualquer Geometry/Phenotyper/Predictor
```

Campos omitidos do dict de `exam(...)` viram ausência estrutural
(imputados por z=0, ponderados por completude — não um erro). Veja
`examples/` para casos completos com todos os campos e múltiplos exames.

## Estrutura do pacote

```
biospace/
├── core/            entidades fundamentais (nunca conhece doença nem algoritmo)
│   ├── biological_system.py, observation.py, measurement.py, feature.py
│   ├── domain.py, latent_domain.py, representation.py, composite_representation.py
│   ├── representation_space.py, trajectory.py, cohort.py, phenotype.py
│   ├── geometry.py (interface), operator.py (interface), distribution.py
│   └── contracts.py            verificação empírica dos contratos formais (ver tabela abaixo)
│
├── geometry/        implementações concretas de Geometry/TrajectoryGeometry/DynamicGeometry
│   ├── euclidean.py, mahalanobis.py, wasserstein.py, information.py, cosine.py
│   ├── dtw.py, gromov_wasserstein.py       (TrajectoryGeometry)
│   ├── riemannian.py                        (geodésica aproximada via grafo k-NN)
│   ├── learned.py                            (NCA — métrica aprendida)
│   └── dynamic.py                             (d(x,y,t) — métrica condicionada por fenótipo)
│
├── phenotyping/     Operator[list[Phenotype]]
│   ├── kmeans.py, clinical_kmeans.py, hdbscan.py, gaussian.py, spectral.py
│   └── contracts.py             check_phenotype_stability (Contrato 8.5)
│
├── prediction/      Predictor (wrapper de qualquer estimador sklearn)
├── risk/            RiskOperator (LinearRiskOperator)
├── intervention/    InterventionOperator — τ: X -> X (FeatureShiftIntervention)
├── early_warning/   EarlyWarningOperator — CriticalSlowingDownDetector
├── longitudinal/    SurvivalAnalyzer (Kaplan-Meier), TransitionAnalyzer, TrajectoryUpdater
├── dynamics/        EvolutionOperator, StabilityOperator, DynamicSystem
│                    (dinâmica espontânea aprendida — Trajectory -> Future State)
├── causal/          check_baseline_balance, ObservationalEffectEstimator,
│                    estimate_propensity/match_on_propensity/estimate_matched_effect,
│                    DigitalTwin, Scenario
├── latent/          FactorAnalysisLatentDomain (Análise Fatorial genérica)
├── ontology/        Ontology (dicionário de dados), run_contract_suite (todos os contratos)
│
└── plugins/
    ├── sleep/       plugin de doença #1 — SAOS, validado com dados reais
    │   ├── domains.py                8 domínios (Anthropometric, Apnea, Hypoxia, SleepArchitecture,
    │   │                              Cardiovascular, Comorbidity, Symptoms, Treatment)
    │   ├── latent.py                  InflammationProxyDomain, FrailtyProxyDomain, AutonomicBalanceProxyDomain
    │   ├── hierarchical.py             HierarchicalSleepRepresentation (agrupamento por sistema fisiológico)
    │   ├── loader.py                   load_from_excel/load_from_dataframe (agrupa por paciente)
    │   └── representation.py, system.py, builders.py, clinical_maps.py
    │
    └── diabetes/    plugin de doença #2 — Tipo 2, inteiramente sintético (rigor arquitetural, não clínico)
        ├── domains.py                7 domínios (Glycemic, Anthropometric, Cardiovascular, Renal, Lipid, Comorbidity, Treatment) -- reexportados de metabolic, Lipid adicionado depois da refatoração
        ├── latent.py                  InsulinResistanceProxyDomain
        ├── synthetic.py                gerador longitudinal (declínio renal por exposição glicêmica acumulada)
        └── loader.py, representation.py, system.py, builders.py, reference.py
```

## Referência por módulo

### `core` — nunca conhece doença nem algoritmo

- `BiologicalSystem(identifier)` — acumula `Observation`s; nunca recriado, só atualizado.
- `Observation(timestamp, source, values)` / `Observable` (subclasse por grandeza) / `Measurement` (com proveniência + incerteza opcional via `Distribution`).
- `SemanticDomain(observables)` — `encode(measurements) -> list[Feature]`; `name` obrigatório (validado na construção).
- `LatentDomain(source_domains)` — domínio sem Observables próprios; exige `hypothesis` declarada e `is_validated=False` por padrão.
- `Representation(domains)` — `transform(system, timestamp=None) -> RepresentationVector`; rejeita domínios com nomes colidentes.
- `CompositeRepresentation(name, children)` — agrupa domínios (ou outros grupos) sob um nome, comporta-se como um domínio comum.
- `RepresentationSpace` / `Cohort` / `Trajectory` — armazenamento; `Cohort.snapshot()` produz um `RepresentationSpace` transversal.
- `Phenotype(name, membership_fn, interpretation)` — uma região de X, nunca um algoritmo.
- `Operator[TOutput]` / `LongitudinalOperator[TOutput]` — interfaces-marcador (só `describe()` é universal).
- `Distribution` / `Normal` / `PointMass` — observações probabilísticas (`std < 0` rejeitado na construção).
- `contracts.py` — ver tabela de contratos formais abaixo.

### `geometry`

| Classe | O que mede |
|---|---|
| `Euclidean`, `Mahalanobis`, `Wasserstein`, `InformationGeometry`, `Cosine` | Distância entre pontos de X |
| `DTW`, `GromovWasserstein` | Distância entre trajetórias inteiras (`TrajectoryGeometry`) |
| `RiemannianGeometry` | Geodésica aproximada por grafo k-NN — espaço não-plano |
| `LearnedGeometry` | Métrica aprendida via NCA a partir de rótulos |
| `PhenotypeConditionedGeometry` | `d(x,y,t)` — métrica local por fenótipo (Ledoit-Wolf) |

### `dynamics` — Trajectory → EvolutionOperator → Future State

`MeanRevertingEvolutionOperator.fit(cohort)` ajusta um processo de
Ornstein-Uhlenbeck discreto por Feature; `.predict(x, delta_t_days)`
extrapola. `StabilityOperator` resume quantas Features divergem (φ≥1).
`DynamicSystem` combina uma `Trajectory` com um `EvolutionOperator`
ajustado para `predict()`/`simulate()`.

### `causal` — associação observacional ajustada, NUNCA prova causal

`do()` (em `DigitalTwin`) aplica uma transformação no espaço de
representação — não é inferência causal identificada no sentido de
Pearl. Sequência recomendada:

```python
from biospace.causal import check_baseline_balance, estimate_propensity, match_on_propensity, estimate_matched_effect

balance = check_baseline_balance(cohort, "treatment", "aam", order=order)   # SEMPRE primeiro
model = estimate_propensity(cohort, "treatment", "aam", order=order)         # regressão logística L2
match_result = match_on_propensity(cohort, "treatment", "aam", order=order)  # pareamento com caliper
effect = estimate_matched_effect(cohort, match_result, order=order)          # diferença-em-diferenças
```

### `plugins/sleep`

Único plugin de doença implementado. `load_from_excel(path)` /
`load_from_dataframe(df)` agrupam por paciente (`id_column="paciente"`)
e ordenam por data (`order_column="inicio"`), produzindo um
`SleepSystem` por paciente com trajetória completa — nunca um sistema
por linha.

## Contratos formais (verificação empírica, não prova matemática)

| # | Contrato | Função |
|---|---|---|
| 5.2 | Preservação Semântica | `core.contracts.check_semantic_preservation` |
| 5.3 | Composicionalidade | `core.contracts.check_compositionality` |
| 5.4 | Continuidade | `core.contracts.check_lipschitz_continuity` |
| 5.5 | Extensibilidade | `core.contracts.check_extensibility` |
| 5.6 | Independência Algorítmica | `core.contracts.check_algorithmic_independence` |
| 5.7 | Temporalidade | `core.contracts.check_temporality` |
| 5.8 | Reprodutibilidade | `core.contracts.check_reproducibility` |
| 5.9 | Versionabilidade | `core.contracts.check_representation_schema_compatibility` |
| 5.10 | Interoperabilidade | `core.contracts.check_interoperability` |
| 8.5 | Estabilidade Fenotípica | `phenotyping.contracts.check_phenotype_stability` |
| — | Injetividade (populacional) | `core.contracts.check_injectivity` |

Todos os 11 rodam juntos via `ontology.run_contract_suite(...)` — cada um
só é checado se os dados necessários forem fornecidos.

**Pendente de verdade** (não implementável como verificador automático,
por natureza): 5.1 Rastreabilidade — hoje só implícita via
`Feature.provenance` (cada Feature carrega de onde veio), mas não existe
um "teste" formal para rastreabilidade além de inspecionar esse campo
manualmente.

**5.3, 5.6 e 5.10 foram fechados numa rodada de investigação dedicada**
(ver seção "Lacunas científicas fechadas" abaixo) — 5.9 já estava
coberta por `check_representation_schema_compatibility`, só não estava
rotulada como tal.

## Lacunas científicas fechadas (investigação dedicada)

### `hypoxia.tempo_total_em_hipoxemia_min`: a "divergência genuína" era 1 outlier, não sinal real

Ficou registrado por várias sessões como a única Feature com divergência
genuína (φ=1,0039) detectada pelo `EvolutionOperator` nos dados reais —
"mereceria investigação dedicada". Investigado: a Feature tem
distribuição fortemente assimétrica (mediana=0, 75º percentil=0,
máximo=135, contra a maioria dos pacientes em zero). Removendo **1 único
paciente** (`sleep_ls_000035`, o outlier de valor 135, de ~296 pacientes
com trajetória) φ cai para 0,984 — **estável**. A divergência nunca foi
progressão populacional real, era sensibilidade a um outlier isolado.

Formalizado como diagnóstico reutilizável:
`biospace.dynamics.check_feature_stability_robustness(cohort, feature_name, order)`
— leave-one-patient-out sobre uma Feature específica, reporta se a
conclusão de estabilidade depende de algum paciente individual. Testado
tanto no caso positivo (recupera o achado real acima) quanto num caso
sintético contraprova (Feature genuinamente estável permanece robusta à
remoção de qualquer paciente).

### Estabilidade fenotípica: varredura de 4 algoritmos confirma que a instabilidade é real, não escolha de K

ARI=0,42 (K=4, `ClinicalKMeansPhenotyper`) ficou registrado como "vale
revisitar com outro K ou algoritmo". Feito: varredura de ~28
configurações — KMeans (K=2..8), GaussianMixture (K=2..6), Spectral
(K=2..6), HDBSCAN (min_cluster_size=5..30) — **nenhuma cruza o limiar de
estabilidade** (ARI≥0,7, Hennig 2007). A mais próxima foi
GaussianMixture K=2 (ARI=0,583). Spectral clustering ficou pior que
aleatório em K pequeno (ARI≈0 ou levemente negativo). Conclusão: a
população de SAOS forma um CONTÍNUO de severidade, não clusters
nitidamente separados — não é um artefato de K ou de algoritmo. Ver
`tests/test_phenotype_stability_sweep.py`.

## Testes

```bash
pytest              # tests/ (pytest.ini já aponta pra lá)
```

Cobre os 11 contratos formais, um teste por bug real encontrado durante o
desenvolvimento (ver `tests/test_core_hardening.py` e
`tests/test_regressions.py`-equivalentes espalhados por módulo), e uma
prova de que o núcleo é genérico de verdade
(`tests/test_core_disease_agnostic.py` — constrói uma doença nova do
zero, sem nenhuma dependência do plugin sleep). A maioria dos testes não
depende do Excel real (dados de paciente não pertencem ao repositório);
os poucos que dependem (achados específicos dos dados reais, como os
dois acima) usam `pytest.mark.skipif` e são pulados quando o arquivo não
está presente (ex.: em CI).

Veja também `examples/` — scripts standalone, comentados, pensados para
serem lidos por humanos (não travar regressão como os testes).

## Dashboard

`biospace_dashboard/` — Streamlit, 15 páginas, construído inteiramente
sobre `biospace` (nunca recalcula representação/fenótipo). Ver seu
próprio README para detalhes. Roda com dados reais (upload de .xlsx) ou
uma coorte sintética longitudinal gerada localmente.

## Limitações conhecidas (resumo — detalhes em HISTORY.md)

- Dois plugins de doença: SAOS (`plugins/sleep/`, com dados reais
  validados) e Diabetes Tipo 2 (`plugins/diabetes/`, inteiramente
  sintético — ver seção abaixo). A genericidade do núcleo é testada por
  dois exemplos independentes, mas nenhuma terceira doença com dados
  reais ainda exercitou isso em produção.
- Sem ajuste por confundidores não medidos em `causal/` — pareamento por
  propensão reduz, nunca elimina, confundimento por indicação.
- `RiemannianGeometry` e `GromovWasserstein` são caros
  computacionalmente (não escalam para populações grandes sem otimização).
- Domínios latentes (`InflammationProxyDomain` etc.) são hipóteses
  estatísticas explicitamente não validadas (`is_validated=False`) —
  sem biomarcador independente nesta planilha para confirmá-las.

## Fase 5 — Representation Learning (`representation_learning/`)

```
Hoje:   Sistema -> Representação
Depois: Sistema -> Representação -> Representation Learning
```

Diferença arquitetural central: o aprendizado acontece **em cima do
`RepresentationSpace` já computado pelos domínios fisiológicos** — nunca
a partir de `Observation`/`Measurement` brutos. `RepresentationLearner.fit(space)`
é o contrato central da interface (não um detalhe de implementação):
isso significa que o aprendizado nunca pode redescobrir uma estrutura
que os domínios já não tenham exposto como eixo — ele reorganiza/comprime
o que os domínios definiram, não substitui essa definição.

`AutoencoderRepresentationLearner` — autoencoder NÃO LINEAR via
`sklearn.neural_network.MLPRegressor` (entrada = saída = X, com uma
camada intermediária estreita = o embedding), com extração manual da
ativação no gargalo. **Por que não PyTorch/TensorFlow**: testado —
`pip install torch` baixa ~1GB de dependências CUDA mesmo para uso
somente em CPU, incompatível com um framework que deve continuar leve e
testável em CI.

```python
from biospace.representation_learning import AutoencoderRepresentationLearner, compare_reconstruction_error

ae = AutoencoderRepresentationLearner(embedding_dim=5, hidden_dim=16)
ae.fit(space)  # nunca cohort/dados brutos — sempre o RepresentationSpace já computado
embedding = ae.transform(space.get(algum_id).as_vector(order))

resultado = compare_reconstruction_error(space, embedding_dim=5)  # compara com PCA automaticamente
```

### Achado real #1: em estrutura latente não linear CONHECIDA, o autoencoder vence claramente

Construí um cenário sintético controlado — 8 Features observadas a
partir de um latente 2D, com mistura de combinações lineares e NÃO
lineares (seno, produto, diferença de quadrados). Erro de reconstrução:
PCA (linear) = 0,562; Autoencoder (não linear) = 0,125 — **~4,5x melhor**.
Confirma que a não linearidade captura algo real quando ela existe.

### Achado real #2 (o mais importante): nos dados reais de SAOS, PCA venceu em TODA configuração testada

Nos 355 pacientes reais (52 dimensões), testei o autoencoder contra PCA
em várias dimensões de embedding (2, 5, 10) e várias combinações de
hiperparâmetros (`hidden_dim`, `alpha`, `max_iter`) — **PCA venceu em
todas**. Não é um problema de ajuste: é uma lição real sobre a
diferença entre métodos lineares (solução analítica, poucos parâmetros)
e não lineares (treino por gradiente, muito mais parâmetros) em uma
amostra pequena — 355 pacientes é pouco para uma rede neural encontrar
uma solução melhor que a solução ótima linear do PCA, mesmo que a
estrutura real tivesse alguma não linearidade.

**Por isso `compare_reconstruction_error()` existe e roda os dois
automaticamente**: a disciplina de "nunca confiar no não linear sem
comparar contra o linear" está embutida na própria API, não é deixada
como responsabilidade do usuário lembrar.

### Achado real #3 (ao validar o teste): comparar por RAZÃO contra um PCA quase-perfeito é instável

Num terceiro cenário (dados EXATAMENTE lineares), PCA atingiu erro
quase zero (0,0018 — solução analítica). O autoencoder, treinado por
gradiente, teve uma folga de otimização (0,04) — pequena em termos
absolutos, mas 22x pior em RAZÃO, só porque o denominador (erro do PCA)
é quase zero. Ajustei o teste correspondente para comparar erro
absoluto, não razão, exatamente por essa instabilidade.

## Fase 10 — Foundation Model: protótipo de arquitetura (`foundation/`)

```
Milhões de pacientes -> BioSpace -> Foundation Model
(não treinado em texto, treinado em estados fisiológicos)
```

Marcada explicitamente como "**no futuro**" no pedido original — tratada
como tal. Com 355 pacientes reais e coortes sintéticas, este projeto
está a ~4 ordens de grandeza de distância de "milhões de pacientes": a
diferença não é de grau, é de escala. Nada aqui é ou finge ser um
foundation model.

O que foi construído: `MaskedFeaturePredictor` — uma prova de conceito
ARQUITETURAL de que a representação deste projeto (Features com
proveniência semântica, construídas por `SemanticDomain`) é um
substrato válido para pré-treino auto-supervisionado, no mesmo padrão
do Masked Language Modeling do BERT (Devlin et al., 2019) — mas
mascarando Features fisiológicas em vez de palavras.

```python
from biospace.foundation import MaskedFeaturePredictor

modelo = MaskedFeaturePredictor(hidden_dim=32, mask_fraction=0.15)
modelo.fit(space)  # RepresentationSpace de UMA doença — nunca dados brutos
resultado = modelo.masked_reconstruction_error(space)  # honesto: erro só nas posições mascaradas
```

### Validação decisiva: estrutura de correlação sintética conhecida

Testado ANTES de confiar em dados reais — Features com correlação
conhecida (f2=2·f1, f3=-f1) devem reconstruir muito melhor que o
baseline ingênuo (prever a média); Features de ruído puro (f4, f5, sem
relação com nada) NÃO devem "fingir" reconstruir melhor que esse mesmo
baseline:

| Feature | MSE reconstrução | Variância (baseline) | Razão |
|---|---|---|---|
| f1, f2, f3 (correlacionadas) | 0,016 – 0,040 | 1,04 – 4,22 | **0,01 – 0,04** |
| f4, f5 (ruído puro) | 1,17 – 1,37 | 0,98 – 1,06 | **1,20 – 1,29** |

O modelo aprende estrutura real onde ela existe (razão «1) e não
inventa estrutura onde não há nenhuma (razão ≈1, honestamente não
melhor que o baseline).

### Nos dados reais de SAOS: achado honesto sobre redundância fisiológica

Comorbidades/tratamentos reconstroem muito bem a partir do resto do
perfil (razão MSE/variância 0,02–0,06 para `irritabilidade`, `cancer`,
`doenca_coronaria`, `cpap`) — plausível: comorbidades tendem a coexistir
em pacientes mais graves. Idade (`anthropometric.idade`) e alguns
extremos de hipoxemia (`hypoxia.tempo_spo2_90`) ficam com razão perto de
ou acima de 1,0 — o modelo não bate o baseline — plausível também: idade
carrega informação relativamente INDEPENDENTE do estado fisiológico do
momento. Genericidade confirmada: o mesmo `MaskedFeaturePredictor`, sem
nenhuma alteração, roda igual sobre o plugin de diabetes.

### O que fica de fora, deliberadamente

Treino com perda mascarada "pura" (só nas posições mascaradas — o MLM
de verdade) não foi implementado; `MLPRegressor` do scikit-learn não
aceita perda com máscara por amostra, então o treino reconstrói o vetor
inteiro (simplificação documentada). A AVALIAÇÃO, porém, sempre mede
erro só nas posições efetivamente mascaradas — a métrica reportada é
honesta mesmo com essa simplificação no treino. Combinar múltiplas
doenças num único modelo (o passo que tornaria isto mais parecido com
um foundation model de verdade) não foi tentado — os espaços de
Features de cada plugin têm dimensões diferentes, e resolver isso
direito é um problema de design em si, não uma linha de código.

## Fase 9 — Simulação: `twin.simulate_ensemble()` (múltiplos futuros, não um só)

```
twin = patient.clone()
twin.simulate(...)
```

`DigitalTwin.clone_from(trajectory)` (= `patient.clone()`) e
`DigitalTwin.simulate()` (determinística) já existiam, construídos
durante a Inferência Causal. O que faltava: simulação em CONJUNTO —
um gêmeo digital de verdade relata incerteza sobre o futuro, não um
único ponto determinístico.

```python
resultado = twin.simulate_ensemble(evolution_operator, horizon_days=2000, step_days=50, n_samples=500)
resultado["mean"]   # média da distribuição preditiva, por instante
resultado["std"]    # desvio padrão da distribuição preditiva, por instante
resultado["paths"]  # todas as n_samples trajetórias simuladas
```

Usa `EvolutionOperator.sample()` (a versão ESTOCÁSTICA de `predict()`,
que já existia com a fórmula teoricamente correta de variância de um
processo de Ornstein-Uhlenbeck discreto) — `simulate_ensemble()` só
precisou encadear chamadas a `sample()` corretamente.

### Achado real e correção: escala de ruído errada inflava a variância em ~7x

Ao validar `simulate_ensemble()` contra a variância ESTACIONÁRIA
TEÓRICA CONHECIDA de um processo OU sintético (target_var=4,0), a
variância simulada convergiu para **29,2** — quase 7,5x o valor real.

**Causa raiz**: `residual_std` (o desvio padrão do resíduo já ajustado
por `MeanRevertingEvolutionOperator`) está na escala do Δt MÉDIO dos
pares usados no ajuste (em dados clínicos reais, tipicamente dezenas a
centenas de dias) — mas a fórmula de variância estacionária do
processo OU precisa do ruído na escala de **1 dia**, consistente com
`phi_per_day`. `sample()` estava usando `residual_std` diretamente como
se já fosse essa escala de 1 dia.

**Correção**: adicionado `FeatureDynamics.mean_dt_days` (Δt médio dos
pares do ajuste) e `FeatureDynamics.sigma_eps_per_day`, que inverte a
relação de variância estacionária:

```
residual_std² = σ_dia² · (1 - φ^(2·mean_dt)) / (1 - φ²)
   =>  σ_dia = residual_std / sqrt((1 - φ^(2·mean_dt)) / (1 - φ²))
```

Depois da correção, a mesma validação convergiu para **3,60** — a 10%
do valor teórico verdadeiro (4,0), dentro do esperado para ruído de
amostragem com 500 simulações. Formalizado como teste de regressão
permanente (`tests/test_simulation.py::test_simulate_ensemble_variance_converges_to_known_stationary_variance`)
— se a escala de ruído regredir para o bug original, este teste falha.

## Fase 8 — Geometria: Variedade, Curvatura, Estabilidade, Metaestabilidade

```
Hoje:   Paciente -> Representação
Depois: Paciente -> Representação -> Variedade -> Trajetória -> Curvatura -> Estabilidade
```

A maior parte desta cadeia já existia antes deste item: `RiemannianGeometry`
(Variedade — geodésica aproximada via grafo k-NN), `Trajectory`,
`StabilityOperator`/`EvolutionOperator` (Estabilidade). Encontrei também
um trabalho já substancialmente avançado (de uma sessão anterior
interrompida) cobrindo Curvatura e Metaestabilidade — verifiquei tudo,
adicionei uma terceira perspectiva complementar, e documento as três
juntas aqui pela primeira vez.

### Três formas INDEPENDENTES de estimar curvatura

| Fonte | Função | O que mede |
|---|---|---|
| Temporal | `FeatureDynamics.curvature` (`biospace.dynamics`) | k = -ln(φ) — direto do φ já ajustado por `MeanRevertingEvolutionOperator`; curvatura ALTA = poço estreito e profundo = recuperação rápida = alta resiliência |
| Densidade populacional | `estimate_density_curvature`, `detect_metastability` (`geometry/curvature.py`) | U''(x) no modo de um potencial efetivo U(x)=-log(densidade(x)) reconstruído por KDE sobre a população transversal |
| Estrutural (grafo) | `ollivier_ricci_curvature`, `graph_curvature_summary` (`geometry/graph_curvature.py`) | Curvatura de Ollivier-Ricci (transporte ótimo entre vizinhanças) sobre o grafo k-NN da VARIEDADE — a única das três que usa a Variedade propriamente dita |

São três medidas de fontes diferentes (dinâmica de 1 paciente ao longo
do tempo; forma da distribuição de 1 Feature na população; estrutura
relacional do espaço inteiro) — não se espera que coincidam
numericamente, mas concordância de DIREÇÃO é evidência de que capturam
algo real.

**Validação da curvatura temporal**: processo de Ornstein-Uhlenbeck
sintético com k verdadeiro conhecido — `FeatureDynamics.curvature`
recuperou k quase exatamente (ρ=0,99+ com o k real, erro relativo <25%).

**Achado real (limitação) da curvatura por densidade**: diferenciar uma
curva de KDE duas vezes amplifica muito a imprecisão — testado com 4
Features de curvatura verdadeira IDÊNTICA por construção, o valor
retornado variou quase 5x (0,14 a 0,71). Documentado como indicador
QUALITATIVO, não para comparação numérica precisa.

**Validação da curvatura estrutural (minha contribuição)** — 3 casos
sintéticos com resultado conhecido da literatura de Ollivier-Ricci,
testados ANTES de confiar em dados reais:
- Grafo completo → +0,625 (fortemente positivo, esperado).
- Ciclo grande → exatamente 0,000 (resultado clássico e bem citado).
- Árvore binária → média global +0,033, inicialmente surpreendente
  (esperava negativo); investigando por tipo de aresta: internas
  (backbone) = -0,310 (negativo, como esperado — gargalo estrutural),
  folhas = +0,333 (positivo — artefato conhecido da caminhada
  preguiçosa em nós de grau 1). A média global só mistura as duas.

**Nos dados reais de SAOS** (grafo de similaridade, 355 pacientes):
arestas que cruzam fenótipos diferentes têm curvatura mais negativa
(-0,146) que arestas dentro do mesmo fenótipo (-0,076) — Mann-Whitney
p=5,7e-19. Fronteiras entre fenótipos são estruturalmente mais frágeis
("gargalos"), exatamente a interpretação da literatura de Ollivier-Ricci
como sinal de perda de resiliência.

### Metaestabilidade

`detect_metastability(space, feature_name)` conta quantos POÇOS
distintos existem na paisagem de potencial de uma Feature — mais de um
poço = múltiplos estados estáveis genuínos (não apenas "há uma
clusterização", mas "existe uma barreira de energia real entre os
grupos", quantificada por `Well.escape_barrier`). Validado: população
unimodal → 1 poço; população bimodal bem separada → 2 poços com
barreira substancial (>1,0); ruído dentro de uma única gaussiana não
fragmenta espuriamente em vários poços.

### Conectando ao que já existia: Early Warning Signals, Resiliência, Critical Slowing Down

Os quatro conceitos citados já tinham (ou passaram a ter, com este
item) um lugar concreto no projeto:

| Conceito | Onde vive |
|---|---|
| Early Warning Signals | `early_warning.CriticalSlowingDownDetector` |
| Critical Slowing Down | idem — variância/autocorrelação/assimetria crescentes numa janela deslizante |
| Resiliência | `FeatureDynamics.resilience_score` (alias direto de `curvature`) |
| Metaestabilidade | `detect_metastability` (poços de potencial) + curvatura estrutural do grafo (fronteiras entre fenótipos) |

## Fase 7 (parte 2) — GNN: `SimpleGCN` em NumPy puro (`gnn/`)

Continuação direta do grafo (`graph/`) — o "depois GNN" que tínhamos
deixado explicitamente pendente. `SimpleGCN`: uma Graph Convolutional
Network de verdade (Kipf & Welling, 2017), forward E backward
(gradiente) derivados e escritos à mão, em NumPy puro — mesma decisão de
não usar PyTorch/TensorFlow já tomada em `representation_learning/` e
`graph/` (~1GB de CUDA mesmo para CPU, e o índice CPU-only do PyTorch
nem está na lista de domínios de rede permitidos aqui).

**Passo crítico, feito ANTES de confiar em qualquer resultado**: o
gradiente analítico (backward manual) foi conferido contra diferenças
finitas — erro relativo de ~1e-8 em todos os parâmetros. Backprop manual
escrito errado não gera nenhum erro em tempo de execução; o modelo
"treina" para a direção errada silenciosamente. Sem essa checagem, nada
do resto teria valor.

```python
from biospace.graph import build_cohort_similarity_graph
from biospace.gnn import SimpleGCN, prepare_node_classification_data
from biospace.geometry import Euclidean

G = build_cohort_similarity_graph(space, Euclidean(), k=8)
dados = prepare_node_classification_data(space, G, labels=fenotipos, labeled_ids=treino_ids)

gcn = SimpleGCN(hidden_dim=16)
gcn.fit(dados["A"], dados["X"], dados["y"], dados["labeled_mask"], epochs=300)
predicoes = gcn.predict(dados["A"], dados["X"])
```

### Achado real: o padrão de cruzamento clássico da literatura de GCN semi-supervisionada

Testado nos 355 pacientes reais, prevendo fenótipo (transdutivo,
semi-supervisionado — Kipf & Welling 2017), comparando a GCN (usa o
grafo) contra o MESMO modelo sem propagação de mensagens (`A=identidade`
— isola o que o grafo contribui):

| Fração rotulada | Com grafo | Sem grafo | Diferença |
|---|---|---|---|
| 50% (177 pacientes) | 0,882 | 0,927 | **-0,045** (grafo atrapalha) |
| 20% (71 pacientes) | 0,817 | 0,845 | -0,028 |
| 10% (35 pacientes) | 0,794 | 0,750 | +0,044 |
| 5% (17 pacientes) | 0,757 | 0,580 | **+0,178** (grafo ajuda muito) |

**Por que isso faz sentido, não é ruído**: os fenótipos vêm de
`ClinicalKMeansPhenotyper` sobre o MESMO espaço X usado como Feature —
um classificador direto em X já reconstrói quase perfeitamente essas
fronteiras quando há dado suficiente, e a suavização por vizinhos do
grafo pode borrar a fronteira exata perto da divisa entre clusters. Mas
quando os rótulos ficam escassos (17 de 355 pacientes), "emprestar"
informação da estrutura do grafo (pacientes parecidos tendem a
compartilhar fenótipo — validado antes em `graph/`: 74,9% vs. 27,4% de
acaso) compensa bastante a falta de exemplos rotulados — exatamente a
motivação original de Kipf & Welling para propor GCN semi-supervisionada.

## Fase 7 (parte 1) — Knowledge Graph: o paciente deixa de ser vetor (`graph/`)

```
Hoje:   patient.vector() -> x ∈ X (um ponto)
Depois: patient.graph()  -> G = (V, E) (uma rede)
```

Escopo deliberadamente limitado ao GRAFO — a GNN em si ("depois GNN")
foi explicitamente colocada como próximo passo pelo pedido original, não
implementada aqui. Dois níveis:

**`build_patient_graph(system, representation, feature_correlations=None)`**
— a rede INTERNA de um paciente: nós para o paciente, cada domínio, cada
Feature, e cada comorbidade/tratamento PRESENTE; arestas
`OBSERVED`/`BELONGS_TO`/`HAS`, e `CORRELATES_WITH` entre Features cuja
correlação POPULACIONAL REAL (`compute_feature_correlations`, não uma
ontologia inventada) excede um limiar. Testado nos dados reais: a maior
correlação encontrada foi `apnea.ido` × `apnea.no_de_dessaturacoes`
(+0,971) — faz sentido clínico completo, ambas medem frequência de
eventos obstrutivos.

**`build_cohort_similarity_graph(space, geometry, k=5)`** — a rede de
PACIENTES (não a rede interna de um só) — nós = pacientes, arestas = k
vizinhos mais próximos sob qualquer `Geometry` já existente no projeto.
Esta é a estrutura que uma GNN futura consumiria (message passing entre
pacientes parecidos). **Validado nos dados reais**: vizinhos no grafo
compartilham fenótipo em 74,9% dos casos, contra 27,4% de um baseline
aleatório — quase 3x melhor, confirma que a rede captura estrutura real.

**`to_pyg_arrays(graph)`** — exporta `node_features`/`edge_index` no
formato cru que PyTorch Geometric/DGL esperam, SEM depender de nenhum
framework de GNN (`pip install torch` baixa ~1GB de CUDA mesmo para
CPU — mesma decisão já tomada em `representation_learning/`) — deixa o
próximo passo pronto para conectar, sem pretender já ser esse passo.

```python
from biospace.graph import build_patient_graph, build_cohort_similarity_graph, compute_feature_correlations, to_pyg_arrays
from biospace.geometry import Euclidean

correlations = compute_feature_correlations(space)
G_patient = build_patient_graph(system, representation, feature_correlations=correlations)
G_cohort = build_cohort_similarity_graph(space, Euclidean(), k=5)
arrays = to_pyg_arrays(G_patient)  # node_features, edge_index -- pronto para uma GNN, quando ela vier
```

**Achado ao validar o próprio teste**: numa população sintética com 2
grupos bem separados, duas Features geradas de forma INDEPENDENTE
dentro de cada grupo ainda correlacionaram >0,9 na população inteira —
correlação "ecológica" (Simpson-like: as duas saltam juntas entre
grupos, mesmo sendo independentes dentro de cada um). Não é bug — é uma
lição real sobre interpretar correlação populacional; o teste foi
ajustado para isolar o que realmente queria checar.

## Refatoração: `plugins/metabolic/` (genérico) + `plugins/diabetes/` (interpretação)

Diabetes deixou de ser um pacote de representação próprio. É agora uma
**interpretação clínica** aplicada sobre uma representação genérica do
sistema endócrino-metabólico — `plugins/metabolic/`, que não sabe o que
é diabetes. Motivação (levantada em revisão externa): o núcleo já
organiza domínios por significado fisiológico, não por doença
(`GlycemicDomain`, `RenalDomain`, ...) — mas o PACOTE continuava
nomeado e estruturado em torno de "diabetes" como conceito
organizador, o que contradizia o próprio princípio.

**`plugins/metabolic/`** — mesmos 7 domínios (6 na refatoração original, Lipid adicionado depois), mesmo domínio latente
(`InsulinResistanceProxyDomain`), mesmo `loader.py`, agora com nomes
disease-agnostic: `MetabolicSystem`, `MetabolicRepresentation`. Nada
aqui pergunta ou assume qual doença o paciente tem.

**`plugins/metabolic/interpretations.py`** — a peça nova. Interpretação
clínica como função pura sobre um vetor JÁ representado — N(R(B)),
nunca R. Duas interpretações independentes, sobre a MESMA representação:

```python
from biospace.plugins.metabolic import MetabolicRepresentation, MetabolicSystem, classify_diabetes_status, classify_metabolic_syndrome_risk, exam

representation = MetabolicRepresentation()
system = MetabolicSystem()
system.observe(exam({"hba1c_pct": 7.5, "glicemia_jejum_mg_dl": 145.0, ...}))
vector = representation.transform(system)

classify_diabetes_status(vector)             # "diabetes" | "pre_diabetes" | "normal"
classify_metabolic_syndrome_risk(vector)     # critérios NCEP ATP III adaptados, individualmente auditáveis
```

**Prova de genericidade, não só afirmação** — o teste decisivo
(`tests/test_metabolic_package_genericity.py`) constrói 4 perfis
clínicos com resultado conhecido (diabetes sem síndrome; síndrome sem
diabetes; as duas; nenhuma) e confirma que as duas interpretações
discordam exatamente onde deveriam discordar. Se sempre concordassem,
"síndrome metabólica" seria só "diabetes" com outro nome — o teste
mostra que não é.

**`plugins/diabetes/`** agora é uma camada fina: reexporta tudo de
`metabolic` sob os nomes antigos (`DiabetesSystem`, `DiabetesRepresentation`
— literalmente `is` a mesma classe, não cópia, verificado em teste) para
não quebrar nada existente, e contribui apenas o que É de fato
específico de diabetes: o critério clínico (`classify_diabetes_status`)
e o gerador sintético (`generate_synthetic_dataframe` — um CENÁRIO de
diabetes: progressão glicêmica, adoção de metformina/insulina, declínio
renal por exposição acumulada; o cenário é específico, a representação
que o consome não é).

```python
from biospace.plugins.diabetes import generate_synthetic_dataframe, load_from_dataframe

df = generate_synthetic_dataframe(n_per_group=30, seed=42)
cohort, representation = load_from_dataframe(df)  # e' metabolic.load_from_dataframe, reexportado
```

**Achado de rigor deliberado no gerador**: declínio renal (eGFR/creatinina)
correlacionado com EXPOSIÇÃO GLICÊMICA ACUMULADA ao longo do tempo, não
apenas severidade basal — hiperglicemia crônica danifica os rins, um
mecanismo real. Testado: correlação positiva entre HbA1c médio e queda
de eGFR (ρ≈0,21 — positiva como esperado, modesta por causa do ruído
individual também injetado deliberadamente).

Validado sem NENHUMA alteração no núcleo ou no resto do ferramental:
contratos formais (Reprodutibilidade, Preservação Semântica,
Temporalidade), `KMeansPhenotyper`, e `check_baseline_balance` (detecta
corretamente confundimento por indicação na adoção de insulina — 13/16
Features desequilibradas). Ver `tests/test_diabetes_plugin.py` (11
testes, todos passando após a refatoração via API pública),
`tests/test_metabolic_package_genericity.py` (4 testes, a prova de
genericidade), e `examples/05_diabetes_toy_disease.py`.

**Implementado depois, em rodada dedicada**: a camada de "Processos
Fisiológicos" entre Observação e Domínio — ver seção "Camada opcional:
PhysiologicalProcess" no topo deste documento.

**Ainda deliberadamente não implementado** (escopo maior, levantado na
mesma revisão): `DomainPackage` como interface abstrata para replicar
o padrão de domínio-por-significado-fisiológico em outros sistemas do
corpo (respiratório, neurológico, hematológico, ...). Mudaria a
organização de todo o repositório de plugins — não implementado
especulativamente sem antes validar a camada de processos isoladamente
(feito acima) e ver o que ela ensina antes de generalizar mais.
