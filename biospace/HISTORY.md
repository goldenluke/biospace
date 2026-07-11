# HISTORY — Diário de Desenvolvimento do BioSpace

Este arquivo é o registro cronológico completo de como o BioSpace foi
construído: cada achado real, cada bug encontrado e corrigido, cada
decisão de design e por quê, sessão após sessão. Preservado
integralmente porque tem valor — é a evidência de que as afirmações do
projeto foram verificadas, não apenas declaradas.

**Se você quer saber COMO USAR o BioSpace, veja [README.pt-BR.md](README.pt-BR.md)
(ou [README.md](README.md) para a versão em inglês).**
Este arquivo é para quem quer entender COMO ele foi construído e
POR QUE cada decisão foi tomada — incluindo os becos sem saída.

---

# BioSpace

Infraestrutura geral para **representação computacional de sistemas biológicos**,
implementando o meta-modelo `M = (B, O, D, R, X, G, Γ, F, C)`.

O plugin `sleep` foi migrado da pipeline real de análise de SAOS
(`Exames realizados (SD).xlsx` + scripts `01_exploracao.py` ... `07_clusterizacao.py`
+ dashboard Streamlit), substituindo feature engineering ad-hoc em pandas por
domínios semânticos determinísticos e um operador de fenotipagem formal.

## Núcleo dividido por entidade (não mais um entities.py monolítico)

```
biospace/core/
├── biological_system.py    BiologicalSystem                (B)
├── observation.py          Observation, Observable          (O)
├── measurement.py          Measurement                      — valor + proveniência
├── feature.py               Feature, features_to_array       — coordenada auditável
├── domain.py                 SemanticDomain                   (D)
├── representation.py         Representation, RepresentationVector (R)
├── representation_space.py   RepresentationSpace              (X)
├── geometry.py                Geometry (interface)             (G)
├── trajectory.py               Trajectory                       (Γ)
├── phenotype.py                 Phenotype                        (F)
├── cohort.py                     Cohort                            (C)
├── operator.py                   Operator (interface p/ algoritmos — fora da lista original, mas indispensável)
├── plugin.py                      Plugin (interface p/ módulos de doença — idem)
├── contracts.py                    verificação empírica dos contratos formais (Seção 5)
└── entities.py                      DEPRECIADO — shim de compatibilidade, reexporta tudo acima
```

`biospace.geometry.base` (implementações concretas) agora reexporta
`Geometry` de `biospace.core.geometry`, em vez de duplicar a definição.

### Measurement e Feature: rastreabilidade de verdade, não por convenção

Antes desta divisão, `Observable.extract()` existia mas nunca era
chamado — `SemanticDomain.collect()` lia direto de um dict achatado
(`system.latest_values()`), perdendo fonte e timestamp de cada valor. E
`SemanticDomain.encode()` devolvia um `np.ndarray` opaco: um float sem
contexto.

Agora:

- `BiologicalSystem.latest_measurement(key)` retorna uma `Measurement`
  (valor + fonte + timestamp) percorrendo as `Observation`s em ordem
  cronológica reversa.
- `Observable.extract(system)` de fato chama isso — não é mais código morto.
- `SemanticDomain.encode(measurements: dict[str, Measurement]) -> list[Feature]`
  — cada `Feature` carrega `value` (o que entra na geometria), `raw_value`,
  `z_score`, `weight` (peso de completude), `is_missing`, `is_excluded` e
  `provenance` (de quais Measurements derivou). Cada coordenada do espaço
  de representação agora é auditável individualmente, não só o domínio
  como um todo.

`RepresentationVector.as_vector()` continua devolvendo um `np.ndarray`
plano (via `features_to_array`) para quem só precisa da geometria — nada
a jusante (Geometry, RepresentationSpace.matrix(), PhenotypingOperator)
precisou mudar.

## Camadas adicionadas: ontology/, geometry/ (Wasserstein+Information), phenotyping/ (HDBSCAN+GMM+Spectral), longitudinal/

```
biospace/
├── ontology/                  catálogo de domínios/observables — NÃO redefine Observable/SemanticDomain
│   ├── observable.py           ObservableRegistry — detecta conflito de definição (mesma key, semântica diferente)
│   ├── semantic_domain.py       DomainRegistry
│   └── ontology.py               Ontology.from_representation() + to_markdown() (dicionário de dados automático)
│
├── geometry/
│   ├── wasserstein.py           Wasserstein — EMD, tratando o vetor como distribuição sobre os eixos
│   └── information.py            InformationGeometry — distância de Fisher-Rao exata (não aproximada)
│
├── phenotyping/
│   ├── hdbscan.py                HDBSCANPhenotyper — ruído (-1) nunca vira um "fenótipo falso"
│   ├── gaussian.py                GaussianMixturePhenotyper — nº de componentes automático via BIC
│   └── spectral.py                 SpectralPhenotyper — clusters não-convexos via grafo de afinidade
│
└── longitudinal/
    ├── updater.py                 TrajectoryUpdater — valida ordem temporal antes de aceitar um novo exame
    └── transition.py                TransitionAnalyzer — matriz de transição com filtro de gap de tempo
                                     + time_to_transition()/summary() (quanto tempo cada transição levou)
```

### Achados reais ao testar estas camadas nos 355 pacientes

- **HDBSCAN degrada totalmente em alta dimensão**: com `min_cluster_size`
  padrão antigo (10) em 52 dimensões, 100% dos pontos viravam "ruído".
  Ajustamos o padrão para 5 e adicionamos um aviso explícito
  (`RuntimeWarning`) quando isso acontece — maldição da dimensionalidade
  é um fenômeno real de métodos baseados em densidade, não um bug.
- **TransitionAnalyzer revelou tempos de transição concretos**: no Excel
  real, "Fenótipo Leve → Hipoxêmico Grave" levou em média 448 dias
  (mediana 206) — informação que a matriz de transição sozinha (sem
  filtro de tempo, sem duração) não mostra.
- **`Ontology.from_representation(representation)`** gera automaticamente
  um dicionário de dados em Markdown a partir da `SleepRepresentation` —
  ver `data_dictionary_sleep.md`, produzido por `run_real_cohort.py`.

### Geometrias de distribuição (Wasserstein/InformationGeometry): use com cautela

Ambas tratam o `RepresentationVector` como uma distribuição sobre os
próprios eixos do domínio (valores deslocados para não-negativos,
normalizados para somar 1). Isso é literalmente correto para domínios que
já são histogramas por natureza (ex.: `tempo_em_ronco_baixo/_medio/_alto`),
mas uma interpretação mais fraca ("ênfase relativa entre eixos") para
domínios z-scored genéricos. `Euclidean`/`Mahalanobis` continuam sendo a
escolha padrão razoável; use as geometrias de distribuição de propósito,
não por padrão.

## Correção crítica: agrupamento por paciente (não por linha)

Ao rodar sobre o Excel real completo, descobrimos que 1557 linhas
correspondem a apenas **355 pacientes únicos** — 296 deles com **mais de
um exame** (um paciente chegou a ter 17 exames entre 2019 e 2024). A
primeira versão do loader criava um `SleepSystem` por LINHA, violando
diretamente o Princípio da Seção 9.3 da teoria ("cada nova observação
atualiza a trajetória, nunca cria um novo paciente").

`load_from_dataframe()` agora agrupa as linhas por `paciente`, ordena por
`inicio` (data do exame) e constrói **um único `SleepSystem` por paciente**,
com sua trajetória completa. `cohort.snapshot()` usa o exame mais recente de
cada paciente para a fenotipagem transversal — a trajetória completa fica
disponível para análise longitudinal (página "Paciente" do dashboard).

Isso também corrigiu um efeito colateral no relatório de qualidade de
dados: `domain.missing_counts` é incrementado por EXAME (chamado a cada
`cohort.update()`), então seu percentual deve ser calculado sobre o total
de exames, não sobre o total de pacientes — os scripts de relatório
(`run_real_cohort.py`, página "Qualidade de Dados") foram ajustados.

## Ponderação automática por completude (dados reais têm ausência estrutural)

Rodar o pipeline sobre a planilha real revelou ausência bem heterogênea
entre domínios (ex.: `no_de_eventos_de_hipoxemia`/`tempo_total_em_hipoxemia_min`
ausentes em 70% dos exames; `ido_sono` em 45%; os 4 campos de arquitetura
do sono em 26%). Tratar esses eixos como se tivessem a mesma confiabilidade
de um campo presente em 99,9% dos casos distorce a geometria do espaço.

`fit_reference()` agora calcula, além de (média, desvio), a **completude**
de cada campo (fração da população com valor não-ausente), retornada como
`FieldStats(mean, std, completeness)`. `_zscore_features()` usa essa
completude para ponderar cada coordenada:

- peso = completude (contribuição proporcional aos dados disponíveis);
- peso = 0 (exclusão total) se completude < `exclude_below` (padrão 5%).

Isso é auditável por Feature (`feature.weight`, `feature.is_excluded`) e
por domínio, via `domain.feature_weights()` — usado por
`run_real_cohort.py` no relatório "QUALIDADE DE DADOS POR DOMÍNIO". Nos
dados reais desta planilha, isso melhorou o silhouette em K=4 de 0.069
para 0.081 (~18%), mantendo a coerência da validação cruzada com
`classificar_apneia()`.

## Estrutura completa

```
biospace/
├── core/                   núcleo: NUNCA conhece doença, algoritmo ou modalidade (ver acima)
│
├── geometry/                camada matemática: distâncias sobre RepresentationSpace
│   ├── base.py               reexporta Geometry de biospace.core.geometry
│   ├── euclidean.py           Euclidean
│   └── mahalanobis.py          Mahalanobis
│
├── phenotyping/              operadores de estimação (algoritmos == Operator)
│   ├── base.py                 PhenotypingOperator (ABC) — nunca define o fenótipo, só estima
│   ├── kmeans.py                KMeansPhenotyper (K fixo)
│   └── clinical_kmeans.py        ClinicalKMeansPhenotyper — K automático via silhouette + rotulagem clínica
│
└── plugins/
    └── sleep/                  plugin migrado da pipeline real de SAOS
        ├── clinical_maps.py     MAPA_DOENCAS / MAPA_SINTOMAS / MAPA_TRATAMENTOS (06_features_texto.py)
        ├── observables.py       ~35 observables = colunas normalizadas da planilha real
        ├── domains.py            8 domínios + classificar_apneia()/classificar_hipoxemia()
        │                        (regras clínicas migradas literalmente de 04_preprocessamento.py)
        ├── loader.py             normalize_column() + load_from_dataframe/load_from_excel
        │                        (agrupa por paciente, ordena por data — ver correção acima)
        ├── system.py             SleepSystem(BiologicalSystem)
        ├── representation.py     SleepRepresentation(Representation)
        ├── builders.py           exam() — construção manual de uma Observation avulsa
        └── plugin.py             SleepPlugin(Plugin)
```

## Domínios reais (vs. a versão sintética inicial)

Este dataset é de um **exame de oximetria/actigrafia** (sem canal de
fluxo aéreo), então **IDO (Índice de Dessaturação de Oxigênio)** funciona
como proxy de apneia — não há AHI verdadeiro.

- `AnthropometricDomain` — idade, peso, altura, IMC
- `ApneaDomain` — IDO, IDO do sono, nº de dessaturações, ronco
- `HypoxiaDomain` — SpO2 mín/média/máx, T90, carga hipóxica, eventos de hipoxemia
- `SleepArchitectureDomain` — latência, duração total, tempo acordado, eficiência
- `CardiovascularDomain` — FC mín/média/máx, amplitude
- `ComorbidityDomain`, `SymptomsDomain`, `TreatmentDomain` — texto livre estruturado

Todos os domínios numéricos orientam seus eixos para que **"maior valor =
mais grave"** (SpO2 e eficiência do sono são invertidas no `encode()`),
replicando a inversão manual feita em `07_clusterizacao.py` — mas agora
como parte determinística e documentada do φ_i.

## Uso

```python
import pandas as pd
from biospace.plugins.sleep import load_from_excel
from biospace.phenotyping import ClinicalKMeansPhenotyper

cohort, representation = load_from_excel("Exames realizados (SD).xlsx", header=1)

space = cohort.snapshot()
phenotyper = ClinicalKMeansPhenotyper()  # K escolhido automaticamente via silhouette
phenotypes = phenotyper.fit(space)

for ph in phenotypes:
    print(ph.name, ph.interpretation)

print("K escolhido:", phenotyper.best_k)
for row in phenotyper.elbow_table:
    print(row.k, row.inertia, row.silhouette)

# Auditar uma coordenada individual (Feature) de um paciente:
system_id = cohort.ids() if hasattr(cohort, "ids") else list(cohort.systems.keys())[0]
vetor = space.get(list(space.ids())[0])
for feature in vetor.components["hypoxia"]:
    print(feature)  # Feature(spo2_minima=-0.625), com .raw_value, .z_score, .weight, .provenance
```

Para forçar K=4 (nomenclatura clínica original — Bradicárdico / Leve /
Hiperadrenérgico / Hipoxêmico Grave):

```python
phenotyper = ClinicalKMeansPhenotyper(k_range=[4])
```

## Limitação conhecida (herdada da pipeline original, não introduzida na migração)

A nomenclatura de K=4 é **posicional**: assume que a ordem crescente de
severidade dos centróides sempre corresponde a essa sequência específica de
4 padrões fisiológicos (bradicárdico → leve → hiperadrenérgico → hipoxêmico
grave), sem verificar se o cluster realmente apresenta bradicardia ou
hiperadrenergia. Isso já existia em `07_clusterizacao.py`; a migração apenas
tornou a suposição explícita e isolada em `_DEFAULT_NAMES_K4`.

## Rodando o demo

```bash
pip install numpy scipy scikit-learn pandas openpyxl
python demo_sleep.py
```

O demo gera uma planilha sintética com as mesmas colunas e regras clínicas
reais (já que não temos acesso ao Excel original neste ambiente), roda a
ingestão completa (incluindo múltiplos exames por paciente), compara a
fenotipagem automática (K por silhouette) com K=4 forçado, e valida
cruzadamente contra `classificar_apneia()`.

## Dashboard Streamlit

Ver `biospace_dashboard/` (pasta irmã deste pacote) — 8 páginas construídas
diretamente sobre `Cohort`/`RepresentationSpace`, sem reimplementar nenhuma
lógica clínica em paralelo.

## Próximas camadas sugeridas

- Portar `08_pacientes.py`...`13_associacoes.py` para dentro do dashboard
  (associações estatísticas entre domínios).
- `biospace.validation` — estabilidade fenotípica, reprodutibilidade,
  capacidade prognóstica (Seção 8.10 da teoria).
- Mais geometrias (`Wasserstein`, `InformationGeometry`) e mais
  phenotypers (`HDBSCAN`, `GMM`, `Spectral`).
- Domínios latentes / operador de harmonização multimodal (Contrato 5.10).
- Novos plugins de doença: `diabetes/`, `heart_failure/`, `copd/`.
- Testes com `pytest` + `hypothesis`, cobertura mínima de 90%.
- Repensar se a referência estatística (`fit_reference`) deveria ponderar
  por paciente em vez de por exame (hoje, pacientes com mais exames pesam
  mais na referência populacional — ver docstring de `load_from_dataframe`).

## longitudinal/survival.py: análise de sobrevivência (Kaplan-Meier)

`SurvivalAnalyzer` estima curvas de Kaplan-Meier (implementação própria,
sem dependência de `lifelines`) para "tempo até um evento de interesse
ocorrer pela primeira vez na trajetória de um paciente" — com censura à
direita tratada corretamente para quem nunca atinge o evento.

```python
from biospace.longitudinal import SurvivalAnalyzer

# tempo até entrar em um fenótipo específico
analyzer = SurvivalAnalyzer.for_phenotype(fenotipo_grave, order=space.order())
km = analyzer.kaplan_meier(cohort)
print(km.median_survival_time())  # em dias

# tempo até uma Feature binária de tratamento virar 1.0
analyzer2 = SurvivalAnalyzer.for_feature_threshold("treatment", "aam")
km2 = analyzer2.kaplan_meier(cohort)
```

Testado nos dados reais: mediana de 12 dias até início de AAM; 97/355
pacientes entraram no fenótipo "Hipoxêmico Grave" em algum momento do
acompanhamento (muitos já no primeiro exame).

## Dashboard atualizado (10 páginas)

Duas páginas novas: **Sobrevivência** (curvas de Kaplan-Meier) e
**Ontologia** (dicionário de dados automático, com download em Markdown).
A página **Perfis** ganhou um seletor de algoritmo de fenotipagem
(K-Means/HDBSCAN/GMM/Spectral, todos operando sobre o mesmo
RepresentationSpace) e uma seção de transição longitudinal com filtro de
intervalo mínimo entre exames.

## Dois novos contratos formais: Injetividade e Compatibilidade entre Versões

`check_injectivity()` varre TODOS os pares de uma população (não apenas
um par isolado, como `check_semantic_preservation`) verificando que
sistemas fisiologicamente distintos nunca colapsam no mesmo ponto do
espaço. Testado nos 355 pacientes reais: **62.835 pares checados, zero
violações** no `ApneaDomain` (custo O(n²), ~1.8s nesta escala — para
coortes muito maiores, amostre um subconjunto).

`check_representation_compatibility()` / `check_representation_schema_compatibility()`
verificam que duas VERSÕES do mesmo domínio (ex.: `Reference` reajustada
sobre uma nova população) preservam o mesmo ESQUEMA — os mesmos nomes de
Feature, na mesma ordem — mesmo que os valores numéricos mudem porque a
referência estatística mudou. Testado comparando `Reference` ajustadas
em duas metades da população real: esquema idêntico, como esperado.

```python
from biospace.core.contracts import check_injectivity, check_representation_compatibility
from biospace.core.feature import features_to_array

report = check_injectivity(
    transform=lambda s: features_to_array(domain.transform(s)),
    systems=list(cohort.systems.values()),
)
print(report.is_injective, report.n_pairs_checked, len(report.violations))

compat = check_representation_compatibility(domain_v1, domain_v2, algum_sistema)
print(compat.is_compatible, compat.only_in_v1, compat.only_in_v2)
```

## Hierarquia de Operator

```
Operator (transversal: RepresentationSpace)          LongitudinalOperator (Cohort)
├── PhenotypingOperator  (phenotyping/base.py)        ├── SurvivalOperator = SurvivalAnalyzer
├── Predictor            (prediction/base.py)         ├── TransitionOperator = TransitionAnalyzer
├── RiskOperator         (risk/base.py)                └── EarlyWarningOperator (interface only — sem
└── InterventionOperator (intervention/base.py)            implementação concreta ainda, de propósito)
```

O único contrato universal é `describe()` — forçar todos em
`fit(space) -> TOutput` seria estruturalmente falso: operadores
longitudinais (Survival/Transition/EarlyWarning) precisam de uma `Cohort`
inteira (trajetórias + tempo), não de um `RepresentationSpace` estático;
`InterventionOperator` transforma um único vetor (`apply()`), não ajusta
um modelo; `RiskOperator` tipicamente não precisa de rótulos (`score()`
transparente), enquanto `Predictor` exige (`fit(space, labels)`).

**Implementações concretas, uma por família (testadas nos dados reais):**

- `SklearnPredictor(estimator)` — envelopa QUALQUER estimador
  sklearn-compatível (RandomForest, XGBoost via wrapper, etc.). Trocar de
  algoritmo é só trocar o `estimator`.
  ⚠️ No teste de fumaça (prever `classificar_apneia()` a partir do
  resto), a acurácia in-sample foi 100% — isso é **vazamento de dado
  esperado** (o rótulo deriva do IDO, que também é um input), não uma
  validação preditiva real. Sirva-se de dados de desfecho genuinamente
  independentes antes de tirar qualquer conclusão clínica.
- `LinearRiskOperator(weights)` — score = soma ponderada de Features
  nomeadas de qualquer domínio. Generaliza a fórmula `score_risco`
  ad-hoc da pipeline legada como pesos explícitos e auditáveis.
- `FeatureShiftIntervention(shifts)` — τ(x) = x + shift nas Features
  nomeadas. Formaliza, como Operator reutilizável, o padrão usado
  manualmente em `demo_sleep.py` para simular uma segunda visita
  pós-tratamento.
- `EarlyWarningOperator` — apenas a interface (herda `LongitudinalOperator`).
  Implementação concreta (janela deslizante → variância → autocorrelação
  → critical slowing down) fica para quando esse item for priorizado
  explicitamente — é o mais inovador e o que mais exige rigor.

`SurvivalAnalyzer`/`TransitionAnalyzer` (já existentes) agora também são
`LongitudinalOperator` de verdade — `fit(cohort)` foi adicionado como
alias de `kaplan_meier(cohort)`/`matrix(cohort)`, e `SurvivalOperator`/
`TransitionOperator` são aliases dos mesmos nomes de classe (nada quebrou
no dashboard, que já usava os nomes antigos).

## Early Warning Signals: CriticalSlowingDownDetector (com teste de significância por substitutos)

`EarlyWarningOperator` ganhou sua implementação concreta:
`CriticalSlowingDownDetector`, aplicando o pipeline clássico de detecção
de "critical slowing down" (Scheffer et al. 2009, *Nature*; Dakos et al.
2012, *PLOS ONE*) sobre trajetórias de uma `Cohort` — **incluindo** a
variante de teste de significância por dados substitutos que Dakos et
al. descrevem (não apenas o p-value paramétrico ingênuo do τ de Kendall):

```
Trajectory -> indicador univariado -> detrend linear -> janela deslizante
    -> variância + autocorrelação lag-1 por janela -> τ de Kendall
    -> significância por SUBSTITUTOS AR(1) (fração de séries nulas com τ >= observado)
```

```python
from biospace.early_warning import CriticalSlowingDownDetector

detector = CriticalSlowingDownDetector.for_distance_from_baseline(n_surrogates=200)
# ou: CriticalSlowingDownDetector.for_feature("apnea", "ido", n_surrogates=200)

results = detector.fit(cohort)  # dict[system_id, EWSResult]; ~10.7s para 355 pacientes (200 substitutos)
for sid, r in results.items():
    if r.sufficient_data:
        print(sid, r.tau_variance, r.p_variance_surrogate, r.warning)
```

### Por que substitutos, não só o p-value do Kendall

O p-value paramétrico do teste de Kendall assume observações
independentes — falso aqui, já que janelas consecutivas se sobrepõem
(são seriadamente correlacionadas por construção) e há poucas janelas
por trajetória. A alternativa implementada: ajustar um processo AR(1)
aos MESMOS resíduos detrendizados (preserva variância/autocorrelação de
base, remove qualquer tendência real), gerar `n_surrogates` séries desse
processo nulo, rodar o MESMO pipeline em cada uma, e comparar o τ
observado à distribuição nula resultante. É a variante paramétrica
(substitutos AR(1)) do método — mais simples que os substitutos
IAAFT/Fourier que Dakos et al. também descrevem (que exigem séries
longas e regularmente amostradas, incompatíveis com a amostragem esparsa
e irregular de dados clínicos reais).

### Resultado: o teste de substitutos é mais conservador, como esperado

| Indicador | Pacientes elegíveis | Warnings (p paramétrico) | Warnings (p substitutos) |
|---|---|---|---|
| Distância multivariada à baseline | 53 | 1 | **0** |
| IDO diretamente (`apnea.ido`) | 53 | 7 | **4** |

Os valores de τ (estimativa pontual) são idênticos entre os dois
métodos — só a significância muda, e caiu em ambos os casos, confirmando
que o p paramétrico estava de fato inflando artificialmente a confiança
(exatamente o problema que a literatura aponta).

### Validação empírica: ainda mista, ainda não conclusiva

A correlação entre τ e progressão real de severidade (fenótipo do
primeiro para o último exame, K=4) **não muda** com o teste de
substitutos (ele afeta o p-value de significância por paciente, não a
correlação populacional entre τ e desfecho):

| Indicador | ρ (Spearman) τ_variância x progressão | p |
|---|---|---|
| Distância multivariada à baseline | -0,29 | 0,035 |
| IDO diretamente | +0,16 | 0,243 |

Continua invertendo de sinal e perdendo significância só de trocar o
indicador, com n=53. O teste de substitutos torna cada paciente
individual mais confiável (menos falsos positivos), mas não resolve a
limitação de fundo: poucos pacientes elegíveis e nenhuma correção para
múltiplas comparações. **Ainda não é uma conclusão clínica** — é uma
ferramenta implementada corretamente e testada com rigor, aguardando
mais dados para validação real.

## Geometrias: Cosine + DTW (TrajectoryGeometry — nova interface)

`Cosine(Geometry)` — compara direção, ignorando magnitude. Documentado
com o mesmo rigor das demais: a distância de cosseno convencional
(1 - similaridade) **não é uma métrica formal** (viola desigualdade
triangular em geral); implementei a convenção usual de ML/ciência de
dados de qualquer forma, por ser a mais reconhecível, com o aviso
explícito no docstring.

`DTW` comparа **trajetórias inteiras**, não pontos — por isso não cabe
na interface `Geometry` (que opera sobre `np.ndarray`). Criei uma
interface irmã, `TrajectoryGeometry` (`core/geometry.py`), com a mesma
lógica que já separou `Operator` de `LongitudinalOperator`: forçar tudo
no mesmo contrato seria estruturalmente falso.

```python
from biospace.geometry import DTW

dtw = DTW()  # point_geometry=Euclidean() por padrão
distancia = dtw.distance(cohort.trajectories["A"], cohort.trajectories["B"])
distancia, caminho = dtw.align(cohort.trajectories["A"], cohort.trajectories["B"])

matrix, ids = dtw.distance_matrix(cohort, min_points=5)  # par-a-par sobre toda a coorte
```

Suporta `time_penalty_weight` (penaliza alinhar pontos muito distantes
no tempo — o DTW clássico ignora isso, o que pode ser fisiologicamente
estranho para trajetórias clínicas) e `normalize` (divide pelo
comprimento do caminho, para trajetórias longas não parecerem
sistematicamente "mais distantes" só por terem mais pontos).

### Validação real: DTW captura similaridade fisiológica genuína

Testado num cenário sintético controlado primeiro (trajetórias de mesma
forma com números de pontos diferentes → distância baixa; formas opostas
→ distância alta; simetria confirmada). Depois, nos 149 pacientes reais
com ≥5 exames: o **vizinho mais próximo por DTW compartilha o mesmo
fenótipo final em 54,4% dos casos**, contra 23,5% em um baseline
aleatório — mais que o dobro, evidência de que a distância captura
estrutura fisiológica real, não ruído.

## Gromov-Wasserstein e Geometria Aprendida (as duas mais arriscadas — tratadas com o devido cuidado)

### GromovWasserstein — compara trajetórias por estrutura relacional, não ponto-a-ponto

Diferente de DTW (que exige correspondência entre pontos do MESMO espaço
de representação), GW compara a matriz de distâncias INTERNAS de cada
trajetória — em princípio permitindo comparar a forma da evolução de um
paciente de SAOS com a de um paciente de outra doença, mesmo com
Representations diferentes (a promessa "cross-disease" da Seção 10 da
teoria). Implementado via `POT` (Sinkhorn entrópico), já que a solução
exata é NP-difícil.

**Dois achados honestos, não favoráveis, documentados de propósito:**

1. **GW é invariante à direção temporal.** Num teste sintético controlado,
   uma trajetória crescente e sua espelhada decrescente (mesma
   variabilidade interna, direção oposta) tiveram distância GW=0,127 —
   quase tão baixa quanto duas trajetórias genuinamente parecidas
   (GW=0,161). Isso é uma propriedade matemática correta do método (GW só
   enxerga estrutura relacional, não ordem), não um bug — mas significa
   que GW não é a ferramenta certa para "esse paciente está piorando ou
   melhorando", só para "a forma de variabilidade é parecida".
2. **Validação real: sem sinal para prever fenótipo final.** No mesmo
   teste de vizinho-mais-próximo usado para validar o DTW (54,4% vs.
   23,5% de baseline), o GW deu **exatamente igual ao acaso** — 30,0% vs.
   30,0% de baseline aleatório, no mesmo subconjunto de 40 pacientes.
   Consistente com o achado 1: a informação que mais importa para prever
   progressão (direção) é exatamente o que o GW descarta.
3. **Custo computacional**: ~0,1-0,2s por par (Sinkhorn), tornando a
   matriz completa da coorte real (149 pacientes) inviável em escala
   interativa (~37 min estimados) — testado apenas em subconjuntos de até
   40 pacientes. O dashboard só calcula GW sob demanda para 1 par por vez.

**Conclusão honesta**: a implementação está correta (testada em cenário
controlado com resultado matematicamente esperado), mas para a tarefa de
prever progressão de severidade nesta coorte, GW não agrega valor sobre
o acaso — diferente do DTW, que tem sinal real. Isso não invalida a
promessa teórica de comparação cross-disease (que exigiria um segundo
plugin de doença para ser testada de verdade), só mostra que, PARA ESTA
tarefa específica, DTW é a ferramenta certa e GW não é.

### LearnedGeometry — métrica aprendida via NCA (Goldberger et al. 2004)

Usa `sklearn.neighbors.NeighborhoodComponentsAnalysis` (implementação de
produção, não uma reimplementação própria) para aprender uma
transformação linear que maximiza a separação entre classes rotuladas.

**Validação com rótulo genuinamente independente** (`classificar_apneia()`,
que vem direto do IDO bruto — não da clusterização sobre a mesma
Representation, evitando o problema de "aprender a enxergar o que o
K-Means já viu"): silhouette score subiu de **0,033** (Euclidiana) para
**0,186** (espaço aprendido) — melhora de ~465%. Este é o achado mais
consistentemente positivo entre as extensões de geometria testadas até
agora.

```python
from biospace.geometry import LearnedGeometry

geo = LearnedGeometry()
geo.fit(space, labels={sid: classificar_apneia(...) for sid in space.ids()})
distancia = geo.distance(x, y)  # x, y = np.ndarray brutos do espaço original
```

## Contratos formais expandidos: Estabilidade + Suíte Consolidada

Dos 7 contratos que vocês listaram, 6 já existiam (`check_reproducibility`,
`check_semantic_preservation`, `check_lipschitz_continuity` = continuidade,
`check_extensibility`, `check_injectivity`, `check_representation_compatibility`).
Faltava **Estabilidade** (Seção 8.5 da teoria) — implementada agora em
`biospace/phenotyping/contracts.py` (não no núcleo: exige um
`PhenotypingOperator` de verdade, que é conceito da camada de fenotipagem).

```python
from biospace.phenotyping.contracts import check_phenotype_stability
from biospace.phenotyping import ClinicalKMeansPhenotyper

report = check_phenotype_stability(
    operator_factory=lambda: ClinicalKMeansPhenotyper(k_range=[4]),
    space=space,
)
print(report.adjusted_rand_index, report.is_stable)  # ARI >= 0.7 (Ben-Hur et al. 2002) = estável
```

### Achado real: os fenótipos de SAOS NÃO são estáveis sob reamostragem

Divide a população em duas metades independentes, ajusta o MESMO
algoritmo separadamente em cada uma, e mede a concordância (Adjusted
Rand Index) quando ambos os conjuntos de fenótipos resultantes são
aplicados a TODA a população. Resultado: **ARI = 0,420** — abaixo do
limiar convencional de 0,7 (Ben-Hur, Elisseeff & Guyon, 2002). Isso é
**coerente com tudo que já havíamos encontrado**: o silhouette de K=4
sempre foi baixo (0,069-0,081) ao longo deste projeto — múltiplas
ferramentas independentes (silhouette, estabilidade por reamostragem)
convergem para a mesma conclusão: a estrutura fenotípica é real (a
validação cruzada com `classificar_apneia()` mostrou gradiente
monotônico de severidade), mas fracamente separada — não uma
clusterização "nítida".

### ContractSuite: os 7 contratos rodados de uma vez, com relatório único

`biospace.ontology.verification.run_contract_suite()` consolida tudo —
cada contrato só roda se os dados necessários forem informados (nenhum
argumento obrigatório força a suíte inteira a falhar por falta de um
único dado).

```python
from biospace.ontology import run_contract_suite

report = run_contract_suite(
    representation=representation,
    domain=apnea_domain,
    systems=systems[:60],
    pairs_for_continuity=pairs,
    raw_distance_fn=raw_distance,
    new_domain_for_extensibility=meu_novo_dominio,
    representation_v2=representation_ajustada_em_outra_populacao,
    space=space,
    phenotyping_operator_factory=lambda: ClinicalKMeansPhenotyper(k_range=[4]),
)
print(report.summary())
```

Rodado nos dados reais (355 pacientes): **6 de 7 contratos OK**
(Reprodutibilidade, Preservação Semântica, Continuidade L=2,748,
Extensibilidade, Injetividade — 1770 pares, zero colisões —,
Compatibilidade de Versões), e 1 identificando corretamente a
instabilidade fenotípica documentada acima. Achado colateral do próprio
processo de testar isso: o teste de Extensibilidade "falhou" na primeira
tentativa porque usei um novo domínio com o MESMO NOME de um já
existente na Representation (colisão de chave no dict de componentes)
— o contrato pegou esse erro de configuração corretamente; ao usar um
nome genuinamente novo, passou.

## Domínios Latentes: LatentDomain + InflammationProxyDomain (o item mais arriscado, tratado com o devido cuidado)

`LatentDomain` (`biospace/core/latent_domain.py`) — um `SemanticDomain`
sem Observables próprios (Seção 6.6 da teoria): em vez de Measurements
diretas, `infer()` reconstrói um estado a partir das Features JÁ
COMPUTADAS de outros domínios (`source_domains`).

**Disciplina obrigatória contra "índice inventado vestido de teoria"**:
toda subclasse concreta DEVE declarar `hypothesis` (justificativa
teórica/literatura — a classe recusa instanciar sem isso) e
`is_validated` (`False` por padrão). A regra mais importante:
**correlação com severidade/fenótipo da mesma doença NÃO CONTA como
validação** — os mesmos domínios-fonte já alimentam a fenotipagem, então
essa correlação é esperada por construção, não uma confirmação
independente.

`FactorAnalysisLatentDomain` (`biospace/latent/`) — implementação
concreta e genérica via `sklearn.decomposition.FactorAnalysis` (modelo
estatístico de variável latente de verdade: observado = cargas @ fator +
ruído — não uma combinação linear arbitrária). Precisa de `fit(cohort)`
sobre uma população antes de transformar qualquer sistema individual
(análogo ao `Reference` do plugin sleep).

### InflammationProxyDomain: exemplo concreto, com um achado real que muda a conclusão

Infere um fator a partir de `HypoxiaDomain` + `CardiovascularDomain` +
`AnthropometricDomain`, com a hipótese de que os três mecanismos
(hipóxia intermitente, obesidade, ativação simpática) compartilham
variância ligada a inflamação sistêmica em SAOS.

```python
from biospace.plugins.sleep.latent import InflammationProxyDomain

inflamacao = InflammationProxyDomain(hipoxia_domain, cardio_domain, antropo_domain, n_factors=1)
inflamacao.fit(cohort)  # necessário antes de usar
print(inflamacao.top_loadings(n=5))  # quais Features mais pesam no fator
```

**Achado real que mudou minha conclusão durante o próprio teste**: com
`n_factors=1` (padrão inicial), o fator extraído foi **dominado pela
hipoxemia** (spo2_media/spo2_minima/tempo_spo2_90, cargas > 0,6) — a
contribuição cardiovascular foi quase nula (|carga| < 0,2). Testei com
`n_factors=2` antes de aceitar isso: os dois mecanismos se **separam de
forma limpa** — um fator dominado por FC, outro por hipoxemia+idade.

**Conclusão honesta**: "inflamação" como um ÚNICO fator latente é uma
simplificação questionável nestes dados — o modelo de 1 fator estava,
na prática, redescobrindo o domínio de hipoxemia, não sintetizando os
três mecanismos hipotetizados. `n_factors` agora é parâmetro do
construtor (não exige subclasse) exatamente por causa deste achado.
`is_validated` permanece `False`: mesmo com 2 fatores separando os
mecanismos de forma sensata, não há biomarcador inflamatório real nesta
planilha contra o qual confirmar que qualquer um dos dois fatores mede
"inflamação" de fato — apenas que há duas fontes de variação
compartilhada plausíveis, com nomes ainda hipotéticos.

## Representações Compostas: CompositeRepresentation (hierarquia, sem quebrar nada)

`CompositeRepresentation` (`biospace/core/composite_representation.py`)
agrupa vários domínios (ou outras composições — aninhamento recursivo)
sob um nome único, viabilizando:

```
patient.representation
├── RespiratoryRepresentation
├── HypoxiaRepresentation
├── CardiovascularRepresentation
└── MetabolicRepresentation
```

**Decisão arquitetural chave**: isso não exigiu NENHUMA mudança no
núcleo. Do ponto de vista de uma `Representation` pai,
`CompositeRepresentation` se comporta exatamente como um domínio comum —
tem `.name` e `.transform(system) -> list[Feature]`.
`Representation.transform()` já funciona por duck typing, então uma
CompositeRepresentation ocupa o lugar de um SemanticDomain em qualquer
Representation existente sem quebrar `RepresentationSpace`, `Cohort`,
`Ontology` ou qualquer coisa que já dependa da API atual.

`HierarchicalSleepRepresentation` (`biospace/plugins/sleep/hierarchical.py`)
é a demonstração concreta — mas é uma representação **alternativa**, não
um substituto de `SleepRepresentation`. O pipeline padrão continua usando
nomes de domínio "planos" (`"apnea"`, `"hypoxia"`, ...); trocar isso
quebraria tudo que já depende desses nomes (dashboard, `Ontology`,
scripts). Testado independentemente, lado a lado.

**Ressalva honesta sobre "MetabolicRepresentation"**: esta planilha não
tem marcadores metabólicos reais (glicemia, HbA1c, perfil lipídico) — o
único proxy disponível é `AnthropometricDomain` (IMC), um indicador
indireto e fraco. Nomeei assim só para seguir literalmente a hierarquia
proposta; um plugin de diabetes teria um `MetabolicRepresentation` de
verdade.

### Capacidade nova, não só cosmética: distância isolada por sistema fisiológico

```python
from biospace.plugins.sleep.hierarchical import HierarchicalSleepRepresentation
from biospace.geometry import Euclidean

rep_h = HierarchicalSleepRepresentation(reference=reference_ja_ajustada)
euclid = Euclidean()
for grupo in [rep_h.respiratory, rep_h.hypoxia_group, rep_h.cardiovascular_group, rep_h.metabolic_group]:
    d = euclid.distance(grupo.sub_vector(sistema_a), grupo.sub_vector(sistema_b))
    print(grupo.name, d)
```

Testado em um par real de pacientes: sistema cardiovascular divergiu mais
(2,07), metabólico menos (1,39) — informação que a distância do vetor
completo (5,37) sozinha não revela. Construí uma Cohort inteira com esta
representação e rodei a fenotipagem normalmente sobre ela (mesmos 4
fenótipos clínicos de sempre — confirma que a reorganização é
estruturalmente neutra: mesmas Features, apenas reagrupadas e
renomeadas com prefixo qualificado, ex.: `"apnea.ido"`).

## Domínios Latentes, parte 2: FrailtyProxyDomain, AutonomicBalanceProxyDomain — e uma recusa deliberada

Dois novos domínios latentes concretos, com o mesmo escrutínio empírico
do `InflammationProxyDomain`:

### FrailtyProxyDomain

Fenótipo de fragilidade clássico (Fried et al., 2001) tem 5 componentes:
perda de peso não intencional, exaustão, baixa atividade física, marcha
lenta, fraqueza de preensão. Esta planilha só sustenta o componente de
**exaustão** — marcha, preensão e perda de peso não existem aqui.

**Achado empírico**: com `n_factors=1`, o fator é dominado por sintomas
de exaustão/sono não reparador (`sonolencia_diurna` +0,31,
`dificuldade_concentracao` +0,29, `sono_nao_reparador` +0,27) — idade
(+0,06) e TODAS as comorbidades individuais (< 0,09) contribuem quase
nada. Ou seja: na prática, isto mede exaustão, não fragilidade
multissistêmica — documentado explicitamente na classe, não escondido.

### AutonomicBalanceProxyDomain

Sem HRV real (SDNN/RMSSD) nesta planilha — só fc_minima/media/maxima
como proxies grosseiros, mais hipóxia intermitente (que ativa o reflexo
quimiorreceptor, gerando surtos simpáticos — mecanismo bem estabelecido
em SAOS). Mesmo padrão do `InflammationProxyDomain`: com `n_factors=1`,
hipoxemia domina completamente. Por isso o padrão aqui é `n_factors=2`:
Fator 1 (cardiovascular — o eixo autonômico genuíno), Fator 2
(hipoxemia — largamente redundante com `HypoxiaDomain`, sem informação
nova).

### Recusa deliberada: "CognitiveReserveDomain" não foi implementado

Reserva cognitiva (Stern, 2002) é operacionalizada por educação, QI
pré-mórbido ou neuroimagem — nada disso existe nesta planilha. Antes de
recusar, testei se um proxy mais modesto ("queixa cognitiva", via
`dificuldade_concentracao` + `perda_memoria`) sobreviveria:

1. Com hipoxia como fonte: os sintomas cognitivos ficam nas posições
   13ª e 19ª de 22 (cargas < 0,04) — hipóxia domina completamente.
2. Sem hipoxia (só sintomas + antropometria): os sintomas cognitivos
   aparecem (~0,2-0,3), mas o fator inteiro é dominado por
   `sonolencia_diurna`/`sono_nao_reparador` — mede sonolência geral, não
   algo especificamente cognitivo.

**Conclusão**: não há, nestes dados, um sinal cognitivo isolável via
Análise Fatorial — nem "reserva" (dado inexistente) nem "queixa
cognitiva" (indistinguível de sonolência geral). Implementar qualquer
classe com "cognitive" no nome aqui seria o "índice inventado vestido de
teoria" que a exigência de `hypothesis` do `LatentDomain` existe para
evitar. Testei antes de recusar — a recusa é baseada em evidência, não
em cautela genérica.

## Observações Probabilísticas: Distribution (incerteza de medição, sem quebrar nada)

Antes: `Observation(values={"ido": 82})` — sempre um valor pontual.
Agora, opcionalmente: `Observation(values={"ido": Normal(82, 2)})` — a
biblioteca passa a representar incerteza de medição explicitamente.

```python
from biospace.core import Normal
from biospace.plugins.sleep.builders import exam

obs = exam({"ido": Normal(82.0, 2.0), "idade": 50})  # mistura livremente com valores pontuais
```

**Como propaga, sem quebrar retrocompatibilidade**:

- `Measurement` ganhou `distribution: Optional[Distribution]` e a
  propriedade `uncertainty` (0.0 se ausente). `.value` continua SEMPRE
  um float (o `mean` da distribuição) — todo código que já espera
  `measurement.value` como float continua funcionando sem alteração.
- `BiologicalSystem.latest_values()`/`values_at()` "desembrulham"
  qualquer `Distribution` para seu `.mean` — esses métodos sempre
  devolvem escalares planos (usados em `classificar_apneia()`,
  dashboard, loader — nada disso precisou mudar).
- `Feature` ganhou `uncertainty: Optional[float]` — `None` quando não há
  incerteza de medição a propagar (o caso comum, todo o dataset real).
- No plugin sleep, `_zscore_features()` propaga a incerteza
  ALGEBRICAMENTE (não por simulação) através da transformação linear
  z = (raw-mean)/std: `sigma_final = (sigma_bruto/std_referencia) * peso`
  — exato, porque a transformação é linear. `CardiovascularDomain`
  propaga a incerteza de `amplitude_fc = fc_max - fc_min` como soma em
  quadratura: `sigma = sqrt(sigma_max² + sigma_min²)`.

**Testado e confirmado**: nos 355 pacientes reais (nenhuma Distribution
usada — só valores pontuais, como sempre), `Feature.uncertainty` é
`None` em 100% dos casos — zero mudança de comportamento. Testei
também os Contratos de Reprodutibilidade e Preservação Semântica com
Observations carregando `Normal(...)` — ambos continuam válidos.

```python
from biospace.core import Normal
from biospace.plugins.sleep import SleepSystem, ApneaDomain
from biospace.plugins.sleep.builders import exam

s = SleepSystem()
s.observe(exam({"ido": Normal(20.0, 3.0), "ido_sono": 18.0, ...}))
features = ApneaDomain().transform(s)
# Feature(ido=0.133 ± 0.200) — incerteza propagada; Feature(ido_sono=0.000) — sem incerteza, como sempre
```

## "Corrigir erros" — nenhum encontrado; e aprimoramento real do Early Warning Signals

Antes de aprimorar, fiz uma varredura completa procurando os erros
relatados: import de todo o pacote `biospace` (todos os submódulos,
recursivamente), `check_install.py`, `demo_sleep.py`,
`run_real_cohort.py`, e as 12 páginas do dashboard (incluindo cada
interação de botão/seletor, não só carregamento passivo). **Nenhum erro
encontrado** — reportado com essa mesma honestidade, em vez de inventar
uma correção para algo que não estava quebrado.

### CriticalSlowingDownDetector: 4 melhorias reais

1. **Terceiro indicador — assimetria (skewness)**: perto de uma
   transição crítica, o sistema tende a passar mais tempo perto de um
   dos dois atratores antes de alternar, distorcendo a distribuição das
   flutuações (Guttal & Jayaprakash, 2008) — um sinal independente de
   variância/autocorrelação.
2. **Peso de evidência em vez de unanimidade**: antes exigia que AMBOS
   variância E autocorrelação concordassem (`warning` exigia os 2/2).
   Agora, com até 3 indicadores disponíveis, exige MAIORIA
   (`ceil(n_disponíveis / 2)`) — mais robusto a um único indicador
   ruidoso, e alinhado com a prática recomendada por Dakos et al. 2012 e
   Kéfi et al. 2013 de "contar quantos indicadores sobem" em vez de
   exigir concordância total entre poucos sinais.
3. **Detrend por kernel gaussiano** (`detrend_method="gaussian"`), como
   alternativa ao linear — captura tendências de longo prazo não
   lineares.
4. **Janela por tempo decorrido** (`window_mode="days"`), além de por
   contagem de pontos — ataca diretamente a limitação de amostragem
   irregular já documentada. Achado ao testar: a cobertura populacional
   varia MUITO com o tamanho da janela escolhida — 90 dias cobre só
   23/53 pacientes elegíveis, 180 dias cobre 34/53, 365 dias cobre
   44/53. Isso é esperado (pacientes têm exames espaçados de meses a
   anos), mas importante saber antes de escolher um valor.

**Retrocompatibilidade total**: `EWSResult` mantém as propriedades
antigas (`tau_variance`, `p_variance_surrogate`, etc.) como atalhos
calculados a partir da nova estrutura `indicators: dict[str,
IndicatorResult]` — o dashboard e qualquer código escrito contra a API
anterior continuam funcionando sem alteração (testado: zero regressões
nas 12 páginas + scripts).

```python
from biospace.early_warning import CriticalSlowingDownDetector

# API antiga continua igual
detector = CriticalSlowingDownDetector.for_distance_from_baseline(n_surrogates=200)

# Novidades, todas opcionais
detector2 = CriticalSlowingDownDetector.for_feature(
    "apnea", "ido",
    detrend_method="gaussian",
    window_mode="days", window_days=180, min_points_per_window=3,
)
results = detector2.fit(cohort)
print(results[algum_id].summary())  # relatório legível com os 3 indicadores + peso de evidência
```

## Contrato 5.7 — Temporalidade (completo: verificador + correção real de bug)

`check_temporality()` (`biospace/core/contracts.py`) verifica o Contrato
5.7 da teoria — "cada observação produz uma ATUALIZAÇÃO da trajetória, e
não uma nova representação independente" — através de 5 propriedades:
comprimento correto (1 ponto por observação), sistema único (nunca cria
um paciente novo), ordem cronológica preservada, timestamps
correspondendo às observações, e **ausência de "espiar o futuro"**.

```python
from biospace.core.contracts import check_temporality

report = check_temporality(
    representation=representation,
    system_factory=lambda: SleepSystem(),  # cria um sistema NOVO e vazio
    observations=lista_de_observations,     # pode estar fora de ordem cronológica
)
print(report.summary())
print(report.is_compliant)
```

### Achado real: um bug genuíno de "espiar o futuro" com observações fora de ordem

Ao testar o contrato com observações deliberadamente fora de ordem
cronológica, encontrei uma falha real (não tautológica — testei antes
com um sistema "quebrado" que não ordena, e ele PASSAVA; foi o sistema
CORRETO que falhava, o oposto do esperado): quando uma observação com
timestamp ANTERIOR chega depois de uma posterior já processada,
`latest_values()` fundia TODAS as observações presentes (incluindo a
"futura"), contaminando o ponto da trajetória que deveria refletir
apenas o que era conhecido até aquele instante.

**Causa raiz**: `Representation.transform(system, timestamp)` usava
`timestamp` só como RÓTULO do ponto resultante, nunca como corte
temporal real sobre quais observações considerar.

**Correção**: adicionei um parâmetro `as_of` propagado por toda a
cadeia — `BiologicalSystem.latest_measurement()/latest_values()` ->
`Observable.extract()` -> `SemanticDomain.collect()/transform()` ->
`LatentDomain`/`CompositeRepresentation` -> `Representation.transform()`.
Quando `timestamp` é informado, ele agora TAMBÉM filtra quais
Observations entram no cálculo (`timestamp <= as_of`), não é mais só uma
etiqueta.

**Retrocompatibilidade total**: como `Cohort.update()` já sempre passava
o timestamp da própria observação (e o loader real sempre observa em
ordem cronológica), o corte `as_of` é um NO-OP para todo o pipeline já
testado — confirmado byte-idêntico nos 355 pacientes reais, nas 12
páginas do dashboard, e em `demo_sleep.py`/`run_real_cohort.py`. A
correção só muda o comportamento no caso que estava genuinamente
quebrado (reprocessamento retroativo fora de ordem).

`check_temporality()` também foi integrado à suíte consolidada
(`run_contract_suite`), completando os 8 contratos formais do projeto
(Reprodutibilidade, Preservação Semântica, Continuidade, Extensibilidade,
Injetividade, Compatibilidade de Versões, Estabilidade Fenotípica,
Temporalidade) em uma única chamada.

## Sistemas Dinâmicos: EvolutionOperator, StabilityOperator, DynamicSystem

Nova camada (`biospace/dynamics/`), completando a evolução natural de
`Trajectory`:

```
Trajectory -> EvolutionOperator -> Future State
                     |
                     v
              StabilityOperator
```

**Nota de nomenclatura**: não se chama "TransitionOperator" — esse nome
já existe (`biospace.longitudinal.TransitionAnalyzer`/`TransitionOperator`,
a matriz de transição entre FENÓTIPOS DISCRETOS). `EvolutionOperator`
opera sobre o ESTADO CONTÍNUO (o vetor de representação inteiro),
aprendendo a dinâmica ESPONTÂNEA (sem intervenção) — diferente também de
`InterventionOperator` (Seção 12.3: efeito de UMA intervenção
específica, aplicado sob demanda).

`MeanRevertingEvolutionOperator` ajusta um processo de Ornstein-Uhlenbeck
discreto POR FEATURE, sobre todos os pares consecutivos (x_t, x_{t+Δt})
de toda a coorte:

```python
from biospace.dynamics import MeanRevertingEvolutionOperator, StabilityOperator, DynamicSystem

evo = MeanRevertingEvolutionOperator(order=representation.domain_names())
evo.fit(cohort)
print(evo.describe())

# Previsao/simulacao para um paciente especifico
ds = DynamicSystem(trajectory=cohort.trajectories[algum_id], evolution_operator=evo, order=order)
estado_futuro = ds.predict(delta_t_days=90)
caminho = ds.simulate(horizon_days=365, step_days=90)

# Estabilidade da dinamica ajustada
report = StabilityOperator(order=order).analyze(cohort)
print(report.summary())
```

### Um viés real, encontrado e corrigido ANTES de aceitar o resultado

A primeira versão estimava φ (taxa de contração) por regressão log-linear
através da origem, restrita a pares com "razão positiva" (mesmo lado da
média). Testei em dados SINTÉTICOS com φ real conhecido (0,9) — e o
resultado veio sistematicamente enviesado para perto de 1 (φ_fit=0,9987
mesmo com 3000 pares, não convergindo com mais dados — descartando viés
de amostra pequena). Causa raiz: o filtro "mesmo lado" descarta
desproporcionalmente os pares com MAIS decaimento, onde o ruído mais
facilmente inverte o sinal do desvio — o que resta na amostra é
enviesado para MENOS decaimento aparente.

**Correção**: troquei para mínimos quadrados não lineares conjuntos sobre
(μ, log φ), sem filtrar nenhum par. Retestado no mesmo cenário sintético:
φ_fit=0,9032 com 3000 pares (real: 0,9) — e exato (0,9000) no caso
determinístico sem ruído.

### Resultado real: majoritariamente estável, com padrões que fazem sentido fisiológico

Nos dados reais (355 pacientes, 800-1200 pares por Feature): **47/52
Features estáveis**. As "instáveis" (φ ≈ 1,0000, não divergência
patológica) são majoritariamente **comorbidades** (hipertensão,
diabetes, câncer, ...) e **idade** — o que é o resultado CORRETO
esperado, não um problema: comorbidades são diagnósticos crônicos que
persistem (não "revertem à média" ao longo dos exames), e idade
simplesmente aumenta monotonicamente — nenhuma das duas deveria mostrar
reversão à média de verdade. A única Feature clinicamente interessante
no topo da lista de instabilidade foi `hypoxia.tempo_total_em_hipoxemia_min`
(φ=1,0039, com 1201 pares) — um sinal real de possível divergência que
mereceria investigação dedicada, distinto do "ruído de fronteira" das
demais.

## Geometria, parte 2: Riemanniana + Geometria da Doença (dinâmica)

### RiemannianGeometry — o espaço deixa de ser plano

Implementada via **geodésica aproximada por grafo k-NN** (Isomap —
Tenenbaum, de Silva & Langford, 2000): constrói um grafo sobre a
população (arestas só entre vizinhos próximos, pesadas pela geometria de
base), e a distância entre dois pontos é o caminho mais curto (Dijkstra)
nesse grafo — não uma variedade Riemanniana contínua resolvida por EDO
(mais caro, sem ganho prático claro aqui).

```python
from biospace.geometry import RiemannianGeometry

riem = RiemannianGeometry(k_neighbors=10)
riem.fit(space)  # necessário antes de distance()
d = riem.distance(x, y)
```

**Validado em cenário sintético clássico (espiral)** antes de confiar
nos dados reais: pontos em voltas ADJACENTES (próximos em linha reta, mas
distantes ao longo da variedade) → geodésica 63% maior que Euclidiana.
Pontos genuinamente vizinhos na mesma volta → geodésica ≈ Euclidiana
(0,28 vs 0,28) — confirma que localmente a aproximação plana é
razoável, exatamente a premissa do método.

**Nos 355 pacientes reais**: geodésica em média **83% maior** que
Euclidiana (razão 1,83x; variando de 1,24x a 2,32x em pares aleatórios)
— a população real ocupa uma região genuinamente não-convexa do espaço
de 52 dimensões.

### DynamicGeometry / PhenotypeConditionedGeometry — d(x, y, t)

Nova interface irmã de `Geometry` (mesma lógica que já separou
`TrajectoryGeometry`): `distance(x, y, t)`, onde `t` é um ESTADO de
referência (aqui, qual fenótipo/estágio de doença) — a métrica em si
muda dependendo de onde no espectro de severidade você está avaliando.

`PhenotypeConditionedGeometry` ajusta uma covariância (Mahalanobis)
LOCAL para cada fenótipo já estimado, usando só seus membros — "cada
doença aprende sua própria geometria" na prática mais direta possível.

```python
from biospace.geometry import PhenotypeConditionedGeometry

geo = PhenotypeConditionedGeometry()
geo.fit(space, phenotypes, order=order)
geo.distance(x, y, "Fenótipo Leve")            # usa a covariância local de pacientes leves
geo.distance(x, y, "Fenótipo Hipoxêmico Grave")  # mesma dupla (x,y), METRICA diferente
```

### Achado real: sem encolhimento de covariância, o método quebra numericamente

Testei primeiro com regularização ingênua (epsilon somado à diagonal) —
e um fenótipo com poucos membros (n=40) relativo à dimensionalidade do
espaço (p=52 — menos amostras que dimensões, covariância amostral
necessariamente quase-singular) deu uma distância de **1288**, ~90x
maior que os demais fenótipos (14-16) — um artefato numérico claro, não
um sinal fisiológico.

**Correção**: troquei para encolhimento de covariância Ledoit-Wolf
(Ledoit & Wolf, 2004 — método padrão para o regime n < p), via
`sklearn.covariance.LedoitWolf`. Resultado depois da correção: todas as
distâncias na mesma faixa razoável (8,2 a 11,5), e o fenótipo pequeno
(n=40) corretamente recebeu o MAIOR encolhimento (0,441 — confia mais na
covariância populacional), contra 0,17-0,26 dos fenótipos maiores —
comportamento estatístico esperado, confirmado.

Ambas as geometrias novas foram integradas à página "Geometrias" do
dashboard (4ª aba), com diagnóstico de shrinkage por fenótipo visível
na interface.

## Inferência Causal: o() de Pearl, Gêmeo Digital e Cenários — o item mais arriscado, tratado com o devido peso

Encontrei um pacote `biospace/causal/` já substancialmente escrito (mesmo
padrão de sessões anteriores interrompidas) — `balance.py`,
`observational_effect.py`, `do_operator.py`, `scenario.py`. Verifiquei
tudo nos dados reais antes de aceitar como correto, e encontrei (e
corrigi) um bug real de usabilidade no processo.

### O aviso central, já embutido no próprio código

`do()` aqui aplica uma TRANSFORMAÇÃO no espaço de representação — não
uma inferência causal IDENTIFICADA no sentido formal de Pearl (sem grafo
causal validado, sem ajuste por confundidores desconhecidos). O nome
`do()` é usado pela clareza do paralelo conceitual, não como alegação de
que a identificação causal foi resolvida.

### `check_baseline_balance` — o primeiro passo obrigatório, testado nos dados reais

Compara a linha de base (antes de qualquer tratamento) entre pacientes
que eventualmente iniciam AAM vs. os que nunca iniciam. Resultado real:
**40 de 52 Features com desequilíbrio relevante** (SMD > 0,1) — os que
iniciam AAM já eram sistematicamente mais graves (IDO, hipoxemia, ronco
maiores) ANTES do tratamento. Confundimento por indicação claro e
esperado, exatamente o que essa checagem existe para expor.

### `ObservationalEffectEstimator` — roda o balanceamento automaticamente, não é opcional por acidente

Estima o Δ médio associado ao início real do AAM (214 transições 0→1
reais na coorte). Resultado real: IDO e hipoxemia melhoram após início
do tratamento — clinicamente plausível — mas o relatório automaticamente
anexa o aviso de desequilíbrio de linha de base, tornando difícil
interpretar o "efeito" sem ver a ressalva.

```python
from biospace.causal import ObservationalEffectEstimator

estimator = ObservationalEffectEstimator(treatment_domain="treatment", treatment_feature="aam", order=order)
report = estimator.estimate(cohort)  # roda check_baseline_balance() automaticamente
print(report.summary())  # inclui o aviso de confundimento, se houver
```

### Bug real encontrado e corrigido: `FeatureShiftIntervention` falhava silenciosamente

Ao testar `DigitalTwin.do()` com uma intervenção hipotética, usei o nome
QUALIFICADO de uma Feature (`"apnea.ido"`, o padrão usado em
`FactorAnalysisLatentDomain` e em outras partes do sistema) — e o shift
**não fez nada, sem erro nem aviso**, porque `FeatureShiftIntervention`
casa por `Feature.name` NÃO qualificado (`"ido"`). Confirmei o problema
comparando os dois casos lado a lado antes de corrigir.

**Correção**: `apply()` agora rastreia quais chaves de `shifts`
realmente casaram com alguma Feature, e levanta `KeyError` claro se
alguma não casou — em vez de aplicar silenciosamente um "do() vazio".
Isso importa especialmente aqui: um gêmeo digital simulado a partir de
um `do()` que não fez nada, sem nenhum aviso, seria um resultado
silenciosamente errado.

### DigitalTwin + Scenario, testados de ponta a ponta

```python
from biospace.causal import DigitalTwin, Scenario
from biospace.dynamics import MeanRevertingEvolutionOperator
from biospace.intervention import FeatureShiftIntervention
from biospace.geometry import Euclidean

evo = MeanRevertingEvolutionOperator(order=order)
evo.fit(cohort)

cenario = Scenario("CPAP vs AAM vs Perda de Peso")
cenario.add_arm("AAM (observacional)", FeatureShiftIntervention(shifts={"ido": -0.269, "spo2_minima": -0.440}))
cenario.add_arm("Perda de peso (hipotética)", FeatureShiftIntervention(shifts={"imc": -1.0}))

resultados = cenario.run(trajectory, evo, horizon_days=180, step_days=60, order=order)
distancias = Scenario.compare_to_control(resultados, Euclidean())
```

Testado com um paciente real (17 exames): 3 braços (controle, AAM
observacional, perda de peso hipotética) simulados corretamente, cada um
com seu `history` de `do()`/`simulate()` auditável, e distâncias ao
controle calculadas coerentemente (AAM: 0,023; perda de peso: 0,989 —
diferença de magnitude reflete os tamanhos de efeito usados, não uma
validação clínica).

### O que ficou de fora, deliberadamente

Nenhum ajuste por escore de propensão (propensity score matching) foi
implementado — o estimador atual compara médias brutas antes/depois,
com o aviso de desequilíbrio anexado, mas sem tentar CORRIGIR o
confundimento estatisticamente. Isso é um próximo passo real, não um
esquecimento: dado que 40/52 Features já mostram desequilíbrio, um
ajuste por propensity score exigiria um modelo de propensão sobre um
espaço de 52 dimensões com apenas 355 pacientes — um problema não
trivial de dimensionalidade, análogo ao que já encontramos e corrigimos
em `PhenotypeConditionedGeometry` (Ledoit-Wolf) e mereceria o mesmo
cuidado antes de ser adicionado.

## Suíte de Testes Automatizada (pytest)

Consolidação pedida explicitamente: parar de adicionar funcionalidades
por um momento e travar como regressão os contratos formais e cada bug
real encontrado ao longo do projeto. Encontrei uma suíte substancial já
escrita (mesmo padrão de sessões anteriores interrompidas — ver
`core/operator.py`, `core/latent_domain.py`, `biospace/causal/`) — 9
arquivos, 49 testes. Rodei tudo, conferi contra os módulos mais recentes,
e adicionei os 3 pontos que faltavam (10 novos testes, 59 → 61 no total
final).

```
tests/
├── conftest.py                       fixtures sintéticos (NUNCA dependem do Excel real)
├── test_core_contracts.py             os 8 contratos formais + run_contract_suite (integração)
├── test_temporality.py                 Contrato 5.7 — inclui o bug de "espiar o futuro" já corrigido
├── test_loader.py                       agrupamento por paciente (bug histórico: 1557 linhas -> 355 pacientes)
├── test_missing_data.py                  imputação + peso de completude + denominador do relatório
├── test_probabilistic_observations.py     Distribution/Measurement/Feature.uncertainty
├── test_geometry.py                        Euclidean/Cosine/DTW/GromovWasserstein/PhenotypeConditioned
├── test_riemannian_and_composite.py         NOVO — RiemannianGeometry + uncertainty em CompositeRepresentation
├── test_phenotyping.py                       HDBSCAN (colapso em alta dimensão) + ClinicalKMeans
├── test_dynamics.py                           viés do estimador de reversão à média (corrigido)
├── test_intervention_and_causal.py             FeatureShiftIntervention (falha silenciosa corrigida) + DigitalTwin/Scenario
└── test_causal_balance.py                       NOVO — check_baseline_balance/ObservationalEffectEstimator
```

**O que ficou de fora de propósito**: nenhum teste depende do Excel real
(dados de paciente não pertencem ao repositório) — todos os fixtures são
sintéticos, com nomes de campo e faixas de valor plausíveis. Isso
significa que os testes rodam em qualquer ambiente, mas não substituem a
validação manual que já fizemos nos 355 pacientes reais ao longo deste
README — são coisas complementares, não a mesma garantia.

```bash
pip install -r requirements-dev.txt
pytest              # roda tudo (pytest.ini já aponta para tests/)
pytest -v           # com nomes de cada teste
pytest tests/test_causal_balance.py -v   # só um arquivo
```

## Fortalecendo a Fundação: auditoria do núcleo + prova de genericidade

Pedido explícito: parar de adicionar funcionalidades e reforçar o
`core/` — a forma como grandes frameworks consolidam no início. Duas
frentes:

### 1. Auditoria linha a linha de todo `core/`

Sete endurecimentos reais, cada um motivado por algo concreto encontrado
na leitura (não hipotético):

| Arquivo | Problema encontrado | Correção |
|---|---|---|
| `domain.py` | `SemanticDomain.name` é obrigatório, mas nada validava que uma subclasse o definisse — falharia confuso, bem mais tarde | Valida na construção (mesmo padrão já usado em `LatentDomain.hypothesis`) |
| `representation.py` | **Dois domínios com o mesmo `.name` colidem SILENCIOSAMENTE** no dict de componentes — exatamente o bug que encontrei manualmente ao testar `check_extensibility()` há várias sessões — mas o núcleo nunca foi corrigido na RAIZ, só reagimos ao sintoma | `Representation.__init__` (e portanto `extend()`, que o usa por baixo) agora recusa nomes colidentes imediatamente |
| `representation.py` | `as_vector()`/`features_flat()` com `domain_order` incluindo um nome ausente dava `KeyError` cru | Erro claro, listando o que está disponível |
| `representation_space.py` | `.matrix()` num espaço vazio quebrava com erro genérico do numpy | `ValueError` claro |
| `representation_space.py` | `.get(id_inexistente)` dava `KeyError` sem contexto | Mensagem mostra quantos pontos existem e uma amostra dos IDs válidos |
| `trajectory.py` | `.latest()` numa trajetória vazia dava `IndexError` genérico | Mensagem aponta a causa (trajetória vazia, nenhum `Cohort.update()` bem-sucedido) |
| `distribution.py` | `Normal(mean, std=-5)` (desvio padrão negativo, sem sentido estatístico) era aceito silenciosamente | Validado na construção |
| `composite_representation.py` | Mesma falta de validação de `name` do `SemanticDomain`, mais `children` vazio (não agrupa nada) | Ambos validados |

Todos os 7 têm teste de regressão dedicado em `tests/test_core_hardening.py`.

### 2. Prova de genericidade: uma doença nova, do zero, só com `biospace.core`

`tests/test_core_disease_agnostic.py` — não usa NADA do plugin sleep.
Constrói um domínio de glicemia/diabetes inteiramente novo
(`GlycemicDomain`, `GlucoseObservable`, `HbA1cObservable`) e confirma que
funciona de ponta a ponta:

```python
system = BiologicalSystem(identifier="paciente_1")  # SEM precisar de subclasse como SleepSystem
system.observe(Observation(timestamp=..., source="exame_laboratorial", values={"glicemia_mg_dl": 190.0, "hba1c_pct": 9.0}))
representation = Representation([GlycemicDomain()])
cohort.update(system, representation, timestamp=...)
```

Testado: `Cohort`/`RepresentationSpace` funcionam sem alteração;
`Euclidean` separa corretamente "normais" de "diabéticos"; os contratos
`check_reproducibility`/`check_semantic_preservation` valem para este
domínio novo; `KMeansPhenotyper` (genérico) separa os 2 grupos sem
NENHUM código específico de doença; `Representation.extend()` compõe
domínios não relacionados (glicemia + pressão arterial) corretamente.

**Isso é uma alegação verificável, não uma afirmação de marketing**: se
algum dia alguém colocar sem querer uma suposição específica de SAOS
dentro de `biospace.core`, é este arquivo de teste que vai quebrar
primeiro — ele existe justamente para isso.

### Suíte final: 77 testes (61 → 77)

```
tests/
├── ... (os 11 arquivos já existentes)
├── test_core_hardening.py           NOVO — 1 teste por endurecimento desta auditoria
└── test_core_disease_agnostic.py     NOVO — prova de genericidade (glicemia, sem nenhuma dependência de sleep)
```

Regressão completa confirmada em tudo: 77/77 testes, `check_install.py`,
`demo_sleep.py`, `run_real_cohort.py` (dados reais), e as 15 páginas do
dashboard — zero quebras, apesar de `Representation` agora recusar
ativamente uma classe de erro que antes passava silenciosamente.

## Inferência Causal, parte 2: Pareamento por Escore de Propensão

O "próximo passo real" deixado pendente da primeira versão de
`biospace.causal` — encontrei `propensity.py` já substancialmente
escrito (mesmo padrão de sessões interrompidas anteriores), testei tudo,
encontrei e corrigi 2 problemas reais, e apliquei nos dados reais com o
mesmo rigor de sempre.

### Validação sintética com efeito verdadeiro CONHECIDO (o passo mais importante)

Antes de confiar em qualquer resultado real, construí um cenário
sintético com confundimento por indicação FORTE e de magnitude
conhecida: um confundidor Z afeta tanto a probabilidade de tratamento
quanto o desfecho (efeito verdadeiro = -5,0; confundimento = +4,0·Z).

- **Efeito ingênuo** (antes/depois, sem ajuste): **-2,04** — enviesado
  para perto de zero, como esperado (tratados têm Z médio mais alto,
  que por si só já aumenta o desfecho, mascarando parte do efeito real).
- **Efeito pareado** (escore de propensão + diferença-em-diferenças):
  **-4,18** — muito mais perto da verdade (-5,0).

### Achado real #1: a contagem binária de desequilíbrio pode esconder uma melhora enorme

Ao validar, encontrei: o SMD de "confounder.z" caiu de **1,50 para
0,14** após o pareamento — uma redução de mais de 90% — mas como 0,14
ainda estava (por pouco) acima do limiar convencional de 0,1, a
contagem binária "Features desequilibradas" ficou em 1/3 tanto antes
quanto depois, fazendo `improved_balance` reportar `False` mesmo com
uma melhora real e substancial. **Corrigido**: a métrica de referência
agora é o `|SMD| médio` (contínuo), não a contagem binária.

### Achado real #2 (bônus): warning de depreciação do sklearn

`penalty="l2"` era passado explicitamente para `LogisticRegression`,
mas já é o padrão — gerava `FutureWarning` a cada chamada. Removido.

### Nos dados reais (AAM): balanceamento melhora, mas a amostra pareada fica pequena demais para confiar no efeito

```python
from biospace.causal import estimate_propensity, match_on_propensity, estimate_matched_effect

model = estimate_propensity(cohort, "treatment", "aam", order=order)  # AUC=0.761
match_result = match_on_propensity(cohort, "treatment", "aam", order=order, caliper=0.25)
matched_effect = estimate_matched_effect(cohort, match_result, order=order)
```

- Só **45 de 294** pacientes tratados (15%) conseguiram par dentro do
  caliper — taxa baixa, honesta (não força maus pareamentos), mas
  significa que os pareados não representam bem todos os que recebem AAM.
- `|SMD|` médio caiu de 0,236 para 0,113 (**52% de redução**) — melhora
  real, mas ainda 23/52 Features desequilibradas (contra 40/52 antes).
- Dos 45 pares, **22 tiveram que ser descartados** (controle com só 1
  exame, sem follow-up para medir Δ) — sobrando 23 pares utilizáveis.
- Com n=23, a razão efeito/desvio-padrão ficou em **0,2-0,3** para quase
  toda Feature — ruído, não sinal. **Conclusão honesta**: o pareamento
  melhora o balanceamento de forma real e mensurável, mas nesta base
  específica não sobra amostra suficiente para que o efeito pareado seja
  mais confiável que o ingênuo — é mais defensável metodologicamente,
  mas não necessariamente mais preciso aqui. Reportado assim no
  dashboard, não escondido atrás de um número que pareça mais sólido do
  que é.

### Suíte: +5 testes (77 → 82)

`tests/test_causal_propensity.py` — validação sintética completa (efeito
ingênuo enviesado, efeito pareado mais perto da verdade, |SMD| médio
melhora mesmo quando a contagem binária não muda, `check_baseline_balance`
detecta o confundimento embutido de propósito).

Dashboard: nova aba "Pareamento por Propensão" na página de Inferência
Causal, com os avisos de taxa de pareamento baixa e razão efeito/desvio-
padrão visíveis diretamente na interface — não só no código.
