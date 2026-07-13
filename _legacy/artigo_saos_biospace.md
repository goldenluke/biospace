# Representação Computacional da Síndrome da Apneia Obstrutiva do Sono: Um Estudo de Caso da Teoria de Representação Computacional de Sistemas Biológicos

## Resumo

Aplicamos um meta-modelo formal de representação computacional de
sistemas biológicos — que separa explicitamente sistema biológico,
observação, domínio semântico, representação, geometria, trajetória,
fenótipo e coorte — a uma coorte real de 355 pacientes com Síndrome da
Apneia Obstrutiva do Sono (SAOS), totalizando 1.556 exames de
oximetria/actigrafia entre 2019 e 2024. Implementamos a arquitetura como
uma biblioteca computacional (BioSpace) e a testamos exaustivamente
contra os dados reais, incluindo oito contratos formais de verificação,
cinco geometrias distintas do espaço de representação, seis operadores
de fenotipagem, análise de trajetória sensível e insensível à direção
temporal, sinais de alerta precoce (*early warning signals*) com teste
de significância por dados substitutos, e três domínios latentes
inferidos por Análise Fatorial. Encontramos uma estrutura de severidade
real, porém fracamente separada (estabilidade por reamostragem
ARI = 0,42), validada de forma independente por um gradiente monotônico
de severidade (`classificar_apneia`, baseado no índice de dessaturação
de oxigênio bruto). A distância que respeita a ordem temporal das
trajetórias (*Dynamic Time Warping*) previu o fenótipo final com mais
que o dobro de acerto do acaso (54,4% vs. 23,5%), enquanto uma distância
matematicamente correta, porém invariante à direção (Gromov-Wasserstein),
não superou o acaso (30,0% vs. 30,0%) na mesma tarefa. Sinais de alerta
precoce mostraram-se inconclusivos, com a correlação entre a tendência
dos indicadores e a progressão real de severidade invertendo de sinal
dependendo da escolha do indicador univariado. Tentativas de inferir três
domínios latentes (inflamação sistêmica, fragilidade, balanço autonômico)
revelaram que, com um único fator, a hipoxemia domina completamente a
variância compartilhada — um segundo fator é necessário para isolar
qualquer contribuição cardiovascular —, e uma quarta tentativa (reserva
cognitiva) foi abandonada após evidência de que nenhum sinal cognitivo
isolável sobrevive nestes dados. Um contrato formal de temporalidade,
desenvolvido durante este trabalho, revelou e permitiu corrigir uma falha
genuína de "vazamento temporal" no tratamento de observações retroativas
fora de ordem cronológica. Concluímos que a escolha de método —
geometria, algoritmo, indicador — não é neutra e frequentemente determina
a conclusão obtida a partir dos mesmos dados subjacentes.

---

## 1. Introdução

A medicina computacional contemporânea tende a tratar a representação do
paciente como uma etapa implícita de pré-processamento, subordinada ao
algoritmo de aprendizado que será aplicado em seguida. Um trabalho
teórico precedente propôs inverter essa relação: definir formalmente a
representação computacional como uma camada independente — um
meta-modelo `M = (B, O, D, R, X, G, Γ, F, C)` que separa o sistema
biológico (`B`) de suas observações (`O`), dos domínios semânticos que
organizam essas observações (`D`), da representação matemática
resultante (`R`, produzindo pontos em um espaço `X`), da geometria que
mede similaridade nesse espaço (`G`), das trajetórias que capturam
evolução temporal (`Γ`), dos fenótipos entendidos como regiões do espaço
(`F`), e das coortes como coleções vivas de trajetórias (`C`).

Este artigo relata a implementação computacional integral dessa teoria —
uma biblioteca que batizamos de BioSpace — e sua aplicação a uma coorte
real de pacientes com Síndrome da Apneia Obstrutiva do Sono (SAOS). O
objetivo não foi apenas demonstrar que o meta-modelo é implementável, mas
submetê-lo a um teste mais severo: usá-lo para investigar se ele produz
conclusões clínicas genuínas, honestas e replicáveis a partir de dados
reais — incluindo quando esses dados são esparsos, irregulares e
incompletos, como costuma ser o caso na prática clínica.

Diferente da maior parte dos trabalhos em fenotipagem computacional de
SAOS, que reportam o desempenho de um único algoritmo sobre um único
conjunto de atributos previamente definido, este trabalho testou
sistematicamente **múltiplas escolhas metodológicas** — diferentes
geometrias, diferentes algoritmos de fenotipagem, diferentes indicadores
univariados para detecção de alerta precoce — sobre a **mesma** coorte e
a **mesma** representação subjacente, permitindo isolar quando uma
conclusão reflete a doença e quando reflete apenas a ferramenta escolhida
para observá-la.

---

## 2. Dados e Coorte

### 2.1 Origem e Natureza dos Dados

Os dados consistem em registros de exames de oximetria/actigrafia
domiciliar (não polissonografia completa — não há canal de fluxo aéreo
registrado). Por essa razão, o **Índice de Dessaturação de Oxigênio
(IDO)** funciona, neste conjunto de dados, como um *proxy* do Índice de
Apneia-Hipopneia (IAH) tradicional, e não deve ser interpretado como
equivalente a ele.

A planilha original continha 1.557 linhas, cada uma correspondendo a um
**exame**, não a um paciente. Um exame de identificação e correção
inicial revelou que essas linhas correspondiam a apenas **355 pacientes
únicos**, dos quais **296 (83,4%)** possuíam mais de um exame ao longo do
tempo — um único paciente chegou a acumular 17 exames entre novembro de
2019 e novembro de 2024. A distribuição do número de exames por paciente
foi fortemente assimétrica à direita (mediana = 4 exames), com 53
pacientes (14,9%) apresentando 8 ou mais exames — o subconjunto usado nas
análises longitudinais mais exigentes deste trabalho (Seções 4.4 e 4.6).

### 2.2 Ausência de Dados

A ausência de dados mostrou-se estrutural e heterogênea entre domínios,
não aleatória:

| Campo | % ausente (sobre exames) |
|---|---|
| `no_de_eventos_de_hipoxemia`, `tempo_total_em_hipoxemia_min` | ~70,3% |
| `ido_sono` | ~44,7% |
| Arquitetura do sono (4 campos: latência, duração, tempo acordado, eficiência) | ~26,3% |
| Demais campos numéricos | < 0,1% |

Essa estrutura sugere que um subconjunto dos exames foi realizado sob um
protocolo mais simples, que não computa determinados índices — uma
característica do serviço, não um defeito da planilha.

### 2.3 Tratamento da Ausência

Em vez de descartar casos incompletos (o que reduziria drasticamente a
amostra dado o padrão acima) ou imputar silenciosamente, adotamos uma
estratégia de **ponderação por completude**: cada eixo numérico da
representação recebe um peso proporcional à fração da população que de
fato possui aquele dado (peso = completude; peso = 0, exclusão total,
abaixo de 5% de completude). Essa abordagem melhorou modestamente a
separação de fenótipos (silhouette em K=4 subiu de 0,069 para 0,081,
~18% de ganho relativo) sem alterar a validação clínica cruzada (Seção
4.2).

---

## 3. Métodos

### 3.1 Arquitetura Computacional

Implementamos as nove entidades do meta-modelo como classes
independentes, cada uma isolada em seu próprio módulo: `BiologicalSystem`
(nunca chamado de "paciente" no núcleo, permitindo generalização a
outros sistemas biológicos), `Observation` e `Observable`
(operadores de extração com proveniência explícita), `Measurement`
(resultado rastreável de uma extração, incluindo suporte a incerteza de
medição via distribuições de probabilidade — Seção 3.7), `Feature` (uma
coordenada auditável da representação, carregando valor bruto, z-score,
peso de completude e proveniência), `SemanticDomain`, `Representation` e
`RepresentationVector`, `RepresentationSpace`, `Geometry`, `Trajectory`,
`Phenotype` e `Cohort`.

O núcleo (`biospace.core`) foi deliberadamente mantido livre de qualquer
conhecimento sobre SAOS, sobre algoritmos específicos, ou sobre
modalidades de dados — toda especificidade médica reside em um *plugin*
de doença, testando empiricamente a alegação teórica de que a
representação computacional pode ser desacoplada tanto do algoritmo
quanto da patologia.

### 3.2 Domínios Semânticos

A representação de cada paciente compõe oito domínios: `Anthropometric`
(idade, IMC), `Apnea` (IDO e ronco), `Hypoxia` (SpO2, T90, carga
hipóxica), `SleepArchitecture` (latência, duração, eficiência),
`Cardiovascular` (frequência cardíaca e sua amplitude), `Comorbidity`,
`Symptoms` e `Treatment` (os três últimos derivados de texto livre
estruturado por mapas clínicos). Cada domínio numérico é codificado por
z-score contra uma referência ajustada sobre a própria população, com
sinal orientado de modo que "maior valor" sempre signifique "mais grave"
(SpO2 e eficiência do sono são invertidas).

### 3.3 Fenotipagem

Implementamos seis operadores de fenotipagem — K-Means com escolha
automática de K por silhueta, K-Means com rotulagem clínica automática
por severidade dos centroides, HDBSCAN, Gaussian Mixture Models (seleção
de componentes por BIC) e Spectral Clustering — todos operando
exclusivamente sobre o espaço de representação (nunca sobre dados
brutos), tratando o algoritmo como um detalhe intercambiável, conforme a
teoria propõe.

### 3.4 Geometrias

Implementamos sete geometrias distintas: Euclidiana, Mahalanobis
(covariância populacional), Cosine (direção, ignorando magnitude),
Wasserstein e Geometria da Informação (distância de Fisher-Rao exata,
tratando o vetor como distribuição sobre os próprios eixos), e duas
geometrias específicas para **trajetórias inteiras** (não pontos
isolados): *Dynamic Time Warping* (DTW) e Gromov-Wasserstein (GW). Esta
distinção — geometria de ponto vs. geometria de trajetória — exigiu uma
interface própria (`TrajectoryGeometry`), já que DTW e GW comparam
sequências de tamanhos e amostragens distintas, uma operação
estruturalmente diferente de comparar dois pontos do espaço `X`.

Adicionalmente, implementamos uma geometria **aprendida**, via
*Neighbourhood Components Analysis* (NCA; Goldberger et al., 2004),
otimizando uma transformação linear que maximiza a separação entre
classes rotuladas.

### 3.5 Análise Longitudinal

Sobre a coleção de trajetórias de uma coorte, implementamos: (i) um
`TransitionAnalyzer`, calculando matrizes de transição fenotípica com
filtro de intervalo mínimo de tempo entre exames consecutivos e o tempo
efetivamente decorrido em cada transição observada; (ii) um
`SurvivalAnalyzer`, estimando curvas de Kaplan-Meier (implementação
própria) para tempo até um evento de interesse (entrada em um fenótipo
específico, início de um tratamento), com censura à direita tratada
corretamente para pacientes que nunca atingem o evento.

### 3.6 Sinais de Alerta Precoce (*Early Warning Signals*)

Implementamos o pipeline clássico de detecção de *critical slowing
down* (Scheffer et al., 2009; Dakos et al., 2012): para um indicador
univariado extraído de cada ponto da trajetória, aplicamos detrend
(linear ou por kernel gaussiano), calculamos variância, autocorrelação
lag-1 e assimetria (Guttal & Jayaprakash, 2008) em janela deslizante
(por número de pontos ou por tempo decorrido), e testamos a
significância da tendência de cada indicador por dados substitutos
(processos AR(1) ajustados aos mesmos resíduos, não pelo p-value
paramétrico do teste de Kendall, que assume observações independentes —
falso aqui, dado que janelas consecutivas se sobrepõem). Um critério de
"peso de evidência" — maioria dos indicadores disponíveis concordando,
não unanimidade — determina o sinal de alerta final.

### 3.7 Domínios Latentes

Implementamos `LatentDomain`, uma especialização de `SemanticDomain` sem
observáveis próprios, cujo valor é inferido a partir das *Features* já
computadas de outros domínios via Análise Fatorial
(`sklearn.decomposition.FactorAnalysis`). Toda subclasse concreta é
obrigada a declarar uma `hypothesis` (justificativa teórica explícita) e
um sinalizador `is_validated` (`False` por padrão), com a regra de que
correlação com o desfecho da mesma doença que os próprios domínios-fonte
alimentam **não constitui validação independente**.

### 3.8 Observações Probabilísticas

Estendemos `Observation` para aceitar, opcionalmente, uma distribuição
de probabilidade (`Normal(média, desvio)`) em vez de um valor pontual,
representando incerteza de medição explicitamente. A incerteza propaga
algebricamente através das transformações lineares do pipeline (z-score,
diferenças) até a *Feature* final, sem alterar o comportamento para
dados sem incerteza declarada (o caso de toda a coorte real analisada).

### 3.9 Contratos Formais

Implementamos oito contratos formais de verificação empírica,
correspondentes a propriedades exigidas pela teoria: Reprodutibilidade
(determinismo), Preservação Semântica (estados distintos produzem
representações distintas), Continuidade (estimativa da constante de
Lipschitz), Extensibilidade (adicionar um domínio preserva os
anteriores), Injetividade (nenhum par de sistemas distintos colide no
mesmo ponto, verificado sobre toda uma população, não apenas pares
isolados), Compatibilidade entre Versões (o esquema de uma representação
permanece estável quando sua referência estatística é reajustada),
Estabilidade Fenotípica (duas amostras independentes da mesma população
devem produzir fenótipos semelhantes, medido por Índice de Rand
Ajustado — Ben-Hur, Elisseeff & Guyon, 2002) e Temporalidade (uma
sequência de observações deve produzir uma única trajetória ordenada
cronologicamente, sem que informação futura contamine pontos passados).

---

## 4. Resultados

### 4.1 Estrutura Fenotípica: Real, porém Fracamente Separada

A fenotipagem por K-Means com escolha automática de K (por silhueta)
convergiu consistentemente para **K=2**, não K=4, tanto na coorte real
quanto em subconjuntos sintéticos de controle — o algoritmo não força
uma granularidade que os dados não sustentam. Ao forçar K=4 (para
comparação com a nomenclatura clínica de quatro fenótipos previamente
em uso — "Bradicárdico", "Leve", "Hiperadrenérgico", "Hipoxêmico
Grave"), obtivemos silhueta entre 0,069 e 0,081 — uma separação fraca
segundo os critérios usuais.

Não obstante, uma validação cruzada **independente** — a classificação
de severidade `classificar_apneia()`, derivada diretamente do IDO bruto,
sem qualquer relação com a clusterização multivariada — revelou um
gradiente monotônico e clinicamente coerente entre os quatro fenótipos:

| Fenótipo (K=4) | n | Distribuição por classe (IDO bruto) |
|---|---|---|
| Bradicárdico | 103 | 81% Normal, 19% Leve |
| Leve | 114 | 56% Normal, 40% Leve, 4% Moderada |
| Hiperadrenérgico | 98 | 53% Normal, 39% Leve, 7% Moderada, 1% Grave |
| Hipoxêmico Grave | 40 | 52% Moderada, 25% Leve, 20% Grave |

Um terceiro teste, de **estabilidade por reamostragem** (dividir a
população em duas metades independentes, ajustar o mesmo algoritmo
separadamente em cada uma, e medir a concordância via Índice de Rand
Ajustado quando ambos os conjuntos de fenótipos são aplicados a toda a
população), retornou **ARI = 0,420** — abaixo do limiar de 0,7
convencionalmente adotado como evidência de estrutura genuína e
reprodutível (Ben-Hur, Elisseeff & Guyon, 2002).

A convergência desses três resultados independentes — silhueta baixa,
validação cruzada com gradiente coerente, estabilidade abaixo do limiar
convencional — sustenta uma leitura específica: a estrutura de
severidade em SAOS, nesta coorte, é **real, porém melhor descrita como
um contínuo fracamente estruturado do que como fenótipos discretos e
nitidamente separados**.

### 4.2 Geometria Aprendida Melhora Substancialmente a Separação

Ajustamos uma geometria NCA usando `classificar_apneia()` como rótulo de
treinamento — um rótulo genuinamente independente da representação
multivariada usada na clusterização, portanto não sujeito à
circularidade de "aprender a enxergar o que o próprio K-Means já viu".
O silhueta da separação por classe subiu de **0,033** (geometria
Euclidiana no espaço original) para **0,186** (espaço transformado pela
NCA) — um ganho relativo de aproximadamente 465%. Isso indica que o
espaço de representação z-scored padrão não pondera os eixos da forma
mais informativa para separar severidade, e que uma métrica aprendida
consegue reorganizar essa informação de modo substancialmente mais
eficaz.

### 4.3 A Direção Temporal da Trajetória Carrega Informação Prognóstica Real

Comparamos duas geometrias de trajetória em uma tarefa idêntica: prever
se o vizinho mais próximo de um paciente (por forma de trajetória)
compartilha o mesmo fenótipo no último exame.

- **Dynamic Time Warping** (alinha trajetórias respeitando a ordem
  temporal): o vizinho mais próximo compartilhou o mesmo fenótipo final
  em **54,4%** dos casos (149 pacientes com ≥5 exames), contra 23,5% de
  uma base aleatória — mais que o dobro.
- **Gromov-Wasserstein** (compara apenas a estrutura relacional interna
  da trajetória, *ignorando* a direção temporal): no mesmo teste, sobre
  um subconjunto de 40 pacientes (o custo computacional do método —
  ordens de grandeza maior que o DTW — impediu testar a coorte inteira
  em tempo hábil), a taxa de acerto foi **exatamente igual** à base
  aleatória (30,0% vs. 30,0%).

Um experimento sintético controlado esclareceu a causa: Gromov-Wasserstein
é matematicamente invariante a reflexões temporais — uma trajetória
crescente e sua espelhada decrescente, com a mesma variabilidade
interna, produziram distância de 0,127, quase idêntica à distância entre
duas trajetórias genuinamente similares (0,161). Esse comportamento é
uma propriedade correta do método, não um defeito de implementação — mas
demonstra que a informação mais relevante para prognóstico em SAOS
(se o paciente está piorando ou melhorando) é exatamente o que uma
geometria invariante à direção descarta.

### 4.4 Sobrevivência e Transição Fenotípica

O tempo mediano até o início de tratamento com Aparelho de Avanço
Mandibular (AAM), estimado por Kaplan-Meier sobre toda a coorte
(97 eventos observados, 258 casos censurados), foi de **12 dias**. O
tempo mediano até a primeira entrada no fenótipo "Hipoxêmico Grave" não
pôde ser estimado com precisão (mediana não alcançada dentro do período
de acompanhamento), refletindo que apenas 97 dos 355 pacientes
atingiram esse fenótipo em algum momento — muitos já no primeiro exame.

A análise de transição fenotípica (filtrando pares de exames
consecutivos com intervalo ≥30 dias, para não confundir reexames de
controle com verdadeiras transições de acompanhamento) revelou tempos
médios de transição específicos por par origem-destino — por exemplo,
"Leve → Hipoxêmico Grave" levou em média 448 dias (mediana 206 dias;
n=27 transições observadas), enquanto "Hipoxêmico Grave → Hipoxêmico
Grave" (persistência no mesmo fenótipo) teve tempo médio de
permanência de 154 dias entre reavaliações (mediana 92; n=50).

### 4.5 Sinais de Alerta Precoce: Resultado Misto, Não Conclusivo

Aplicamos o detector de *critical slowing down* a 53 pacientes com ≥8
exames (o subconjunto mínimo necessário para uma janela deslizante
minimamente informativa). Testamos dois indicadores univariados
distintos — distância multivariada à linha de base da própria
trajetória, e o valor bruto do IDO diretamente — e correlacionamos a
tendência de variância (`τ` de Kendall) com a progressão real de
severidade (diferença de posição de severidade entre o fenótipo do
primeiro e do último exame):

| Indicador | ρ (Spearman) τ_variância × progressão | p |
|---|---|---|
| Distância multivariada à linha de base | −0,29 | 0,035 |
| IDO diretamente | +0,16 | 0,243 |

A correlação **inverteu de sinal** e perdeu significância estatística
apenas ao trocar o indicador univariado utilizado, com uma amostra de 53
pacientes e sem correção para múltiplas comparações. Adicionalmente,
substituímos o teste de significância paramétrico (que assume
observações independentes, uma suposição falsa dado que janelas
consecutivas se sobrepõem) por um teste de dados substitutos AR(1) —
mais rigoroso, e por consequência mais conservador: o número de
pacientes sinalizados com alerta caiu de 1 para 0 (indicador de
distância) e de 7 para 4 (indicador de IDO) ao trocar o critério de
significância.

Concluímos que a ferramenta está corretamente implementada — testada em
cenários sintéticos controlados e submetida a um teste de significância
rigoroso — mas que **o sinal de alerta precoce não é robusto, nesta
escala de dados, à escolha do indicador univariado**, e não deve ser
tratado como conclusão clínica validada.

### 4.6 Domínios Latentes: Hipoxemia Domina a Variância Compartilhada

Tentamos inferir três estados fisiológicos latentes — inflamação
sistêmica, fragilidade e balanço autonômico — por Análise Fatorial sobre
combinações de domínios já existentes, na ausência de qualquer
biomarcador direto (não há PCR, IL-6, HRV real, marcha ou preensão nesta
planilha).

Em todos os três casos, com um único fator extraído, a hipoxemia
dominou completamente a variância compartilhada (cargas fatoriais de
SpO2 média/mínima e T90 sistematicamente acima de 0,6), com contribuição
cardiovascular ou de comorbidade próxima de zero. Apenas ao forçar dois
fatores os mecanismos se separaram de forma limpa — um eixo dominado por
frequência cardíaca (o componente cardiovascular genuíno), outro por
hipoxemia e idade (amplamente redundante com o domínio de hipoxemia já
existente). Para o proxy de fragilidade, mesmo com essa separação, idade
e comorbidades individuais contribuíram quase nada (cargas < 0,09) — na
prática, o fator mediu exaustão sintomática, não fragilidade
multissistêmica no sentido de Fried et al. (2001).

Uma quarta tentativa — um proxy de "reserva cognitiva" — foi
deliberadamente **abandonada** após teste empírico: com hipoxemia como
domínio-fonte, os únicos dois sintomas cognitivos disponíveis
(dificuldade de concentração, perda de memória) ficaram nas posições 13ª
e 19ª de 22 variáveis em carga fatorial; sem hipoxemia como fonte, esses
sintomas emergiram, mas como parte indistinguível de um fator geral de
sonolência diurna e sono não reparador, não de um construto
especificamente cognitivo. Concluímos que não havia, nestes dados, sinal
suficiente para justificar a existência de um domínio latente cognitivo
sob qualquer nome.

### 4.7 Um Bug de Vazamento Temporal Descoberto pelo Próprio Contrato Formal

Ao implementar o contrato formal de Temporalidade — que exige que uma
sequência de observações produza uma única trajetória corretamente
ordenada, sem que informações futuras contaminem pontos passados —
construímos um teste com observações deliberadamente fora de ordem
cronológica, esperando validar apenas o comportamento correto do
sistema. Em vez disso, o teste revelou uma falha genuína: quando uma
observação com timestamp anterior era adicionada depois de uma posterior
já processada, a fusão de valores do sistema (`latest_values()`)
incorporava a observação futura no cálculo do ponto que deveria refletir
apenas o passado — um vazamento temporal real, embora restrito ao
cenário de reprocessamento retroativo fora de ordem (o pipeline de
carregamento em produção sempre processa observações em ordem
cronológica, e não foi afetado). Corrigimos a causa raiz — introduzindo
um parâmetro de corte temporal propagado por toda a cadeia de cálculo —
e confirmamos que a correção é numericamente neutra (idêntica byte a
byte) para todo o pipeline já validado sobre a coorte real.

---

## 5. Discussão

### 5.1 A Representação Não É Neutra

O achado mais recorrente deste trabalho, atravessando praticamente todas
as seções de resultados, é que **a escolha de método frequentemente
determina a conclusão obtida a partir dos mesmos dados subjacentes**: a
mesma coorte produziu evidência de estrutura fenotípica real (validação
cruzada) e evidência de fraca separação (silhueta, estabilidade) a
depender de qual pergunta se fez à mesma representação; a mesma
trajetória produziu sinal prognóstico forte (DTW) ou nulo
(Gromov-Wasserstein) a depender exclusivamente de a geometria escolhida
preservar ou descartar a direção temporal; e o mesmo pipeline de
detecção de alerta precoce produziu correlações de sinais opostos a
depender apenas do indicador univariado selecionado. Essa observação
corrobora diretamente a motivação teórica original: a representação
computacional de um sistema biológico é um objeto de estudo próprio, não
um subproduto neutro do algoritmo aplicado sobre ela.

### 5.2 O Valor de Contratos Formais Verificáveis

O episódio do bug de vazamento temporal (Seção 4.7) ilustra
concretamente o valor de tratar propriedades teóricas como testes
executáveis, não apenas como afirmações em texto. O contrato não foi
escrito para confirmar uma expectativa — foi escrito para testá-la, e
revelou uma falha real que nenhuma das validações anteriores (rodadas
extensivamente sobre 355 pacientes reais) havia exposto, precisamente
porque o pipeline de produção nunca exercitava o cenário de
reprocessamento fora de ordem.

### 5.3 Sobre Domínios Latentes e a Tentação do "Índice Vestido de Teoria"

A exigência arquitetural de que todo domínio latente declare uma
hipótese teórica explícita antes de ser instanciado, e que permaneça
não-validado até que exista um desfecho independente contra o qual
compará-lo, mostrou-se mais do que uma formalidade: foi o mecanismo que
nos levou a abandonar o domínio de reserva cognitiva após teste, e a
descobrir que os outros três domínios latentes precisavam de dois
fatores, não um, para não colapsar trivialmente em uma redescrição do
domínio de hipoxemia já existente.

---

## 6. Limitações

Este trabalho tem limitações importantes, várias delas já discutidas em
detalhe nas seções correspondentes:

1. **Ausência de biomarcadores independentes.** Nenhum dos três domínios
   latentes pôde ser validado contra um desfecho medido
   independentemente (biomarcador inflamatório, HRV real, avaliação
   funcional de fragilidade), permanecendo hipóteses não-confirmadas.
2. **Amostra pequena para análises longitudinais exigentes.** Apenas 53
   pacientes possuíam dados suficientes para a análise de sinais de
   alerta precoce, e o teste de correlação com progressão de severidade
   não teve correção para múltiplas comparações.
3. **Custo computacional do Gromov-Wasserstein** impediu testá-lo sobre
   a coorte completa, limitando a comparação com DTW a um subconjunto de
   40 pacientes.
4. **Ausência de canal de fluxo aéreo.** O IDO como proxy do IAH é uma
   limitação do próprio dispositivo de coleta, não da modelagem.
5. **Um único plugin de doença.** A alegação teórica de comparabilidade
   *entre* doenças diferentes (motivação central do Gromov-Wasserstein)
   não pôde ser testada empiricamente, por não dispormos de uma segunda
   coorte de outra patologia neste trabalho.

---

## 7. Conclusão

A implementação computacional integral do meta-modelo de representação
de sistemas biológicos, aplicada a uma coorte real de SAOS, produziu
tanto confirmações quanto achados negativos ou mistos — e tratamos
ambos como resultado legítimo. Existe uma estrutura de severidade real
em SAOS nesta coorte, mas ela é mais bem descrita como um contínuo
fracamente estruturado do que como fenótipos discretos. A direção
temporal de uma trajetória clínica carrega informação prognóstica que
uma comparação puramente estrutural descarta. Sinais de alerta precoce e
domínios latentes multissistêmicos, promissores em teoria, mostraram-se
metodologicamente frágeis quando submetidos a testes de robustez
explícitos — não porque a implementação estivesse incorreta, mas porque
os próprios dados, nesta escala e composição, não sustentam essas
conclusões com a confiança que uma leitura menos cética poderia sugerir.
Se este meta-modelo tem valor além de sua elegância teórica, esse valor
reside precisamente nisso: tornar possível — e, neste trabalho,
obrigatório — perguntar não apenas "o que os dados mostram", mas "o que
a ferramenta escolhida para olhá-los estava, e não estava, apta a
revelar".

---

## Referências

Ben-Hur, A., Elisseeff, A., & Guyon, I. (2002). A stability based method
for discovering structure in clustered data. *Pacific Symposium on
Biocomputing*.

Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V.,
Ives, A. R., Kéfi, S., Livina, V., Seekell, D. A., van Nes, E. H., &
Scheffer, M. (2012). Methods for detecting early warnings of critical
transitions in time series illustrated using simulated ecological data.
*PLOS ONE*.

Flamary, R., Courty, N., Gramfort, A., et al. (2021). POT: Python
Optimal Transport. *Journal of Machine Learning Research*.

Fried, L. P., Tangen, C. M., Walston, J., et al. (2001). Frailty in
older adults: evidence for a phenotype. *The Journals of Gerontology
Series A*.

Goldberger, J., Roweis, S., Hinton, G., & Salakhutdinov, R. (2004).
Neighbourhood components analysis. *Advances in Neural Information
Processing Systems (NeurIPS)*.

Guttal, V., & Jayaprakash, C. (2008). Changing skewness: an early
warning signal of regime shifts in ecosystems. *Ecology Letters*.

Kéfi, S., Guttal, V., Brock, W. A., et al. (2013). Early warning
signals of ecological transitions: methods for spatial patterns.
*PLOS ONE*.

Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S.
R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G.
(2009). Early-warning signals for critical transitions. *Nature*.

Stern, Y. (2002). What is cognitive reserve? Theory and research
application of the reserve concept. *Journal of the International
Neuropsychological Society*.

---

*Nota metodológica: todos os valores numéricos reportados neste artigo
foram obtidos executando o código da biblioteca BioSpace diretamente
sobre a coorte real de 355 pacientes (1.556 exames), não estimados ou
aproximados. O código-fonte completo, incluindo os scripts que geraram
cada resultado citado, está disponível junto a este projeto.*
