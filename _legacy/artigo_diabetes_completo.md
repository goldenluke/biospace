# Diabetes Mellitus Tipo 2 no Framework BioSpace: Síntese Empírica em Duas Fontes de Dados Reais e Independentes

**Análise computacional — framework BioSpace**
**Julho de 2026**

---

## Resumo

Este artigo reúne os achados empíricos obtidos ao aplicar um framework de representação computacional de sistemas biológicos a duas fontes de dados públicas, reais e estruturalmente independentes sobre diabetes mellitus tipo 2: o National Health and Nutrition Examination Survey (NHANES, ciclo combinado agosto/2017–março/2020, 9.232 adultos analisados) e o conjunto UCI Diabetes 130-US Hospitals (Strack et al., 2014; 101.766 encontros hospitalares, 71.518 pacientes). As duas fontes diferem estruturalmente ao ponto de exigirem representações computacionais genuinamente distintas — uma baseada em valores laboratoriais contínuos (NHANES), outra em utilização hospitalar e testagem categórica esparsa (UCI) — e essa diferença foi tratada como oportunidade de teste, não normalizada. Os achados centrais: (1) subdiagnóstico real de diabetes (sensibilidade 66,6%, especificidade 95,6% do critério laboratorial contra autorrelato) e subdiagnóstico ainda mais acentuado de pré-diabetes (91,5% dos casos sem relato); (2) a estrutura fisiológica esperada (HbA1c e glicemia correlacionando mais entre si que com outras variáveis) confirma-se em população real e não se confirma no gerador sintético usado em fases anteriores do projeto; (3) o NHANES mostra o oposto exato de um achado publicado anteriormente para apneia obstrutiva do sono (SAOS) em quase toda análise estrutural comparável — estabilidade fenotípica alta em vez de ausente, curvatura que não discrimina fronteira em vez de discriminar, autoencoder que vence PCA em vez de perder, grafo relacional que atrapalha em vez de ajudar; (4) inferência causal aplicada a adoção de insulina revela confundimento por indicação real e um limite estrutural honesto do método (estimação de efeito exige dado longitudinal, que o NHANES não tem, e o módulo recusa a estimativa em vez de produzir um número inválido); (5) na UCI, um fenótipo construído sem idade nem diagnóstico — só utilização hospitalar, testagem glicêmica esparsa e intensidade de medicação — associa-se a quase o dobro da taxa de readmissão em 30 dias, caracterizado por utilização *prévia*, não por intensidade de tratamento; (6) dinâmica de reversão à média ajustada pela primeira vez em trajetória real multi-encontro (23,4% dos pacientes da UCI) mostra-se globalmente estável, com a Feature mais próxima da instabilidade validada como sinal genuíno, não artefato de amostra. Reportamos também, com o mesmo padrão de rigor aplicado aos achados clínicos, os erros reais de implementação e de metodologia encontrados e corrigidos ao longo do processo — incluindo uma autocorreção de uma alegação estatística própria que não sobreviveu a uma verificação mais ampla.

---

## 1. Introdução

Diabetes mellitus tipo 2 foi usado, ao longo deste projeto, como estudo de caso para testar se um framework de representação computacional construído primeiro para outra doença (apneia obstrutiva do sono, SAOS) generaliza para um sistema fisiológico diferente — e, mais adiante, para duas fontes de dados com naturezas completamente diferentes entre si. Este artigo não introduz o framework nem a arquitetura (documentados em `README.pt-BR.md` e `METABOLISM_FINDINGS.md`); reúne exclusivamente os **achados empíricos** — o que foi encontrado, testado e, em alguns casos, corrigido depois de um exame mais cuidadoso.

O fio condutor que atravessa quase todos os achados é uma comparação, nem sempre buscada deliberadamente, com os resultados publicados anteriormente para SAOS. Em praticamente toda análise estrutural — estabilidade de fenótipo, curvatura de grafo, autoencoder vs. PCA, aprendizado semi-supervisionado em grafo — o NHANES produziu o resultado **oposto** ao de SAOS. Isso não é tratado aqui como uma doença "confirmando" e outra "refutando" o framework; é tratado como o resultado mais interessante do projeto: o mesmo ferramental, aplicado a duas populações com propriedades estruturais genuinamente diferentes, produz conclusões diferentes e corretas — evidência de que os métodos discriminam estrutura real, não a impõem.

---

## 2. Fontes de Dados e Métodos

### 2.1 NHANES (ciclo Pre-pandemic, agosto/2017–março/2020)

Inquérito de saúde populacional do CDC, representativo dos EUA, incluindo todas as idades por desenho. Seis arquivos públicos (`P_DEMO`, `P_GHB`, `P_GLU`, `P_BMX`, `P_BPXO`, `P_DIQ`) fornecem demografia, HbA1c, glicemia de jejum, antropometria, pressão arterial e questionário de diabetes (incluindo uso de insulina). Amostra analisável, restrita a adultos (idade ≥ 20 anos): **9.232 participantes**, 87,5% com HbA1c registrado. Representação computacional: `MetabolicRepresentation` — seis domínios físiológicos (glicêmico, antropométrico, cardiovascular, renal, comorbidade, tratamento), nenhum deles nomeado em torno de "diabetes".

### 2.2 UCI Diabetes 130-US Hospitals

Strack et al. (2014) — **101.766 encontros hospitalares**, **71.518 pacientes distintos**, 130 hospitais dos EUA, 1999–2008. Estruturalmente incompatível com a representação do NHANES: sem HbA1c/glicemia contínuas (só categorias, 83,3%/94,8% ausentes), sem antropometria, pressão ou creatinina. Em compensação, oito variáveis de utilização hospitalar 100% completas e 23 classes de medicação detalhadas. Representação computacional própria (`UCIHospitalRepresentation`): `HospitalUtilizationDomain`, `GlycemicTestingDomain`, `MedicationIntensityDomain` — conectada ao NHANES apenas onde há mecanismo fisiológico real compartilhado (o processo `glucose_homeostasis`, comum a `GlycemicTestingDomain` e ao domínio glicêmico do NHANES).

### 2.3 Métodos analíticos aplicados

Classificação clínica (critério ADA para diabetes/pré-diabetes; critério NCEP ATP III adaptado para síndrome metabólica); teste de coerência de processo fisiológico (correlação intra- vs. inter-processo, Mann-Whitney); varredura de estabilidade fenotípica (K-Means, ARI entre reamostragens, Hennig 2007); curvatura estrutural de Ollivier-Ricci em grafo de similaridade; comparação autoencoder vs. PCA por erro de reconstrução; classificação semi-supervisionada em grafo (GCN) variando fração de rótulo; inferência causal observacional (balanceamento de linha de base, pareamento por escore de propensão, estimativa de efeito); dinâmica de reversão à média (processo de Ornstein-Uhlenbeck por Feature) com diagnóstico de robustez a outlier.

---

## 3. Resultados — NHANES

### 3.1 Subdiagnóstico de diabetes e, mais acentuadamente, de pré-diabetes

Entre 8.965 adultos com HbA1c/glicemia e resposta válida sobre diagnóstico prévio, o critério laboratorial (HbA1c ≥ 6,5% ou glicemia de jejum ≥ 126 mg/dL) identificou diabetes com **sensibilidade de 66,6%** e **especificidade de 95,6%** em relação ao autorrelato (acurácia global 91,0%).

| Classificação laboratorial | Autorrelato: Sim | Autorrelato: Não |
|---|---|---|
| Diabetes | 946 | 331 |
| Pré-diabetes | 267 | 2.874 |
| Normal | 207 | 4.340 |

Entre os classificados como pré-diabetes por laboratório (n=3.141), **91,5%** não têm nenhum diagnóstico autorreferido — subdiagnóstico ainda mais acentuado que o de diabetes propriamente dito, consistente com a literatura epidemiológica.

### 3.2 Coerência de processo fisiológico: confirma em dado real, não em dado sintético

Teste: HbA1c e glicemia (mesmo processo, "homeostase glicêmica") correlacionam mais entre si do que com variáveis de outros processos? Na amostra adulta do NHANES: |r| médio intra-processo = 0,782 (n=3 pares) contra 0,151 inter-processo (n=12 pares), Mann-Whitney p=0,0022 — **confirma**.

O mesmo teste, aplicado anteriormente a uma coorte sintética de diabetes construída para validação arquitetural (não clínica), **não confirmou** — o gerador sintético amostrava cada variável independentemente dentro do grupo de severidade, sem mecanismo latente compartilhado. A mesma ferramenta, duas fontes com propriedades diferentes, duas respostas corretas.

### 3.3 Síndrome metabólica sob critério adaptado

Usando os quatro critérios disponíveis (adiposidade central, IMC, pressão arterial, glicemia — 2 de 4 necessários, adaptação explícita do NCEP ATP III de 5 critérios, que exigiria perfil lipídico não coletado): **43,9%** de risco elevado numa amostra de 2.000 adultos. Não diretamente comparável a taxas publicadas com o critério completo.

### 3.4 Estabilidade fenotípica: o oposto exato de SAOS

Em SAOS (355 pacientes), nenhuma das 28 configurações testadas (K-Means, Misturas Gaussianas, Espectral, HDBSCAN, K=2–8) cruzava o limiar convencional de estabilidade (ARI≥0,7, Hennig 2007). No NHANES, K-Means com K=2 atinge **ARI de até 0,957** e permanece acima de 0,7 até K=7 — quase todas as configurações estáveis.

**Investigado antes de aceitar**: seria idade disfarçada de estrutura metabólica (idade correlaciona com quase toda variável fisiológica numa população real)? Removida a Feature `idade`, testado com amostra ampliada de 20 seeds (uma primeira verificação com apenas 3 seeds sugeriu, incorretamente, que a instabilidade sem idade era universal — corrigido após uma reexecução independente contradizer essa alegação, ver Seção 5). Resultado preciso:

- **K=2, sem idade**: robustamente estável em 20/20 seeds (mínimo 0,871, média 0,916) — há estrutura metabólica genuína além da idade.
- **K=3, com idade**: 0,938±0,018 (quase sem variância, nunca abaixo de 0,7 em 20 seeds).
- **K=3, sem idade**: 0,533±0,195 (desvio-padrão dez vezes maior), 18 de 20 seeds abaixo do limiar de estabilidade.

A idade não "sustenta" a partição em K=3 de forma absoluta — torna-a muito mais **reproduzível**, sem a qual a partição existe mas fica muito mais sensível à amostra exata usada.

### 3.5 Curvatura estrutural: achado negativo, coerente com a alta estabilidade

Em SAOS, arestas que cruzam fronteira de fenótipo têm curvatura de Ollivier-Ricci significativamente mais negativa que arestas internas (p=5,7×10⁻¹⁹). No NHANES, a mesma comparação (K=2, amostra de 1.500) **não é significativa** (dentro=-0,106, entre=-0,103, p=0,36). Interpretação: a assinatura de curvatura parece específica de fronteiras estruturalmente frágeis num contínuo mal separado; com fenótipos já bem separados e estáveis, sobra pouca tensão estrutural para detectar.

### 3.6 Autoencoder vence PCA em baixa dimensão — a primeira vez neste projeto

Em SAOS (n=355), PCA venceu em toda dimensão de embedding testada. No NHANES (n=9.232), o autoencoder vence em dim=2, robusto em 9 configurações (3 tamanhos de camada oculta × 3 seeds): erro de reconstrução 0,102 (PCA) contra 0,059–0,088 (autoencoder, melhorando com mais capacidade). Em dim=5 e dim=8, porém, PCA volta a vencer — o efeito é específico de dimensão baixa. Confirma diretamente a hipótese de tamanho de amostra: com dado suficiente, o método não linear encontra solução melhor que a ótima linear.

### 3.7 Aprendizado semi-supervisionado em grafo: o grafo atrapalha, mesmo com poucos rótulos

Em SAOS, com apenas 5% de rótulos, a estrutura relacional (GCN) melhorava a classificação em +17,8 pontos percentuais sobre features isoladas. No NHANES, o grafo **atrapalha ou não ajuda** de 5% a 50% de rótulos (ex.: em 5%, com_grafo=0,923 vs. sem_grafo=0,951). Coerente com os dois achados anteriores: fenótipos já tão bem separados que a classificação por Features sozinha satura, sobrando pouco espaço para a estrutura relacional contribuir. Em frações extremas (~1,5%, n=22), há sinal de que o grafo passa a ajudar, mas a margem (+0,009) é pequena demais para confirmar.

### 3.8 Inferência causal: confundimento por indicação real, e um limite estrutural honesto

Primeira aplicação do módulo causal fora de SAOS. Restrito a adultos com diagnóstico de diabetes (n=1.420): usuários de insulina (n=413) têm HbA1c de linha de base muito mais alto que não-usuários (n=1.007) — SMD=+0,619, o maior desequilíbrio entre 15 Features — confundimento por indicação clássico. Pareamento por propensão funcionou: 369 de 413 pacientes pareados, desequilíbrio zerado em todas as 15 Features.

**`estimate_matched_effect` recusou corretamente** ao tentar estimar o efeito do tratamento: o método calcula diferença-em-diferença (último exame menos primeiro), exigindo ≥2 exames por paciente; NHANES é transversal (1 exame por pessoa). Documentado como limite estrutural real: as duas etapas do módulo causal (balanceamento/pareamento vs. estimativa de efeito) têm exigências de dado diferentes, e o módulo recusa produzir um número degenerado em vez de apresentá-lo como estimativa válida.

---

## 4. Resultados — UCI Diabetes 130-US Hospitals

### 4.1 Trajetórias reais multi-encontro

16.773 pacientes (23,4%) têm mais de um encontro hospitalar, até 39–40 no máximo observado — a primeira trajetória real e longitudinal do projeto, fora de sleep e de dado sintético. `encounter_id` (não uma data real) usado como proxy de ordem cronológica, documentado explicitamente como tal.

### 4.2 Utilização hospitalar, não medicação, prevê readmissão

Fenotipagem K-Means (K=4, apenas 3 partições não-vazias) sobre utilização, testagem glicêmica esparsa e intensidade de medicação — **sem idade, sem diagnóstico**:

| Fenótipo | n | Readmissão <30d | Utilização prévia (outpatient/emergência/internação) | Medicação |
|---|---|---|---|---|
| A | 51.186 | 3,97% | Baixa (0,13/0,07/0,30) | Baixa |
| B | 14.241 | 4,64% | Baixa (0,23/0,15/0,48) | Alta (100% mudança de dose de insulina) |
| C | 6.091 | **8,75%** | **Alta (2,24/0,74/1,85)** | Moderada |

O fenótipo de maior risco (C) não é o de maior intensidade de tratamento (esse é B, risco só moderadamente elevado) — é o de maior utilização **prévia**, consistente com a literatura estabelecida de predição de readmissão.

### 4.3 Dinâmica de reversão à média em trajetória real

`MeanRevertingEvolutionOperator` ajustado sobre os 16.773 pacientes multi-encontro — pela primeira vez em dado genuinamente longitudinal e real. Resultado: **13/13 Features estáveis** (globalmente estável). A mais próxima do limite de instabilidade (`utilization.number_emergency`, φ=0,98) foi testada com o diagnóstico de robustez (remoção de paciente individual, 40 testados) — conclusão **robusta**, não muda removendo nenhum paciente extremo, consistente com o fenômeno real e documentado de "frequent flyers" em uso de serviços de emergência, não um artefato de amostra pequena.

---

## 5. Discussão

### 5.1 O tema central: real vs. sintético, não apenas NHANES vs. SAOS

Embora a maior parte deste artigo contraste NHANES com SAOS, a comparação metodologicamente mais importante é outra: coerência de processo (Seção 3.2) confirma em dado real e não confirma no gerador sintético do próprio projeto. Isso significa que a distinção relevante não é "duas doenças diferentes", mas **dado real vs. dado simulado** — mesmo um gerador sintético cuidadosamente construído para outros fins não reproduz necessariamente a estrutura de correlação que populações reais têm, e conclusões obtidas exclusivamente em dado sintético não substituem validação em dado real quando a pergunta depende dessa estrutura.

### 5.2 Erros reais, corrigidos como parte do processo, não escondidos

Este projeto tratou erros de implementação e de metodologia como achados a serem reportados com o mesmo rigor dos achados clínicos:

- **`pandas.read_sas` exige `format="xport"`, não `"xpt"`** — descoberto só ao processar os arquivos reais do NHANES; nenhum teste com dado fabricado (que mockava a leitura de arquivo) o teria detectado.
- **Ciclo NHANES errado assumido inicialmente** — arquivos reais eram do ciclo Pre-pandemic (2017–2020, `P_BPXO`/`BPXOSY1`), não do ciclo isolado 2017-2018 (`P_BPX`/`BPXSY1`) que a documentação sugeria antes de inspecionar os arquivos de verdade.
- **Um teste de dashboard com import local mal posicionado** — só quebrava se o segundo de dois botões fosse clicado sem o primeiro ter rodado antes; passava despercebido em testes que só verificavam carregamento de página, não interação completa.
- **Uma alegação estatística própria, incorreta, autocorrigida** — a Seção 3.4 originalmente afirmava que K=3 "sempre desestabiliza sem idade", validada com apenas 3 seeds. Uma reexecução independente, mais tarde, encontrou uma dessas mesmas 3 seeds dando resultado oposto (estável). Investigado com 20 seeds antes de corrigir a alegação para o que os dados realmente sustentam: variabilidade muito maior, não instabilidade universal.
- **Um teste causal com bug de dado de entrada** — passava o código de saída (`"aam"`) em vez da frase clínica de entrada esperada (`"Aparelho de avanço mandibular"`), fazendo `check_baseline_balance` reportar zero pacientes tratados; só apareceu ao rodar a suíte inteira, não isoladamente.

Nenhum desses erros invalida os achados clínicos reportados — todos foram descobertos e corrigidos antes de qualquer achado ser reportado como final — mas registrá-los é parte do padrão de evidência deste projeto, não um adendo.

### 5.3 Duas fontes, duas hipóteses de representação, uma conexão formal mínima

NHANES e UCI não foram comparados diretamente em nenhum momento — a diferença estrutural entre as fontes tornaria essa comparação metodologicamente imprópria. A única conexão formal entre as duas representações é a declaração de que `GlycemicTestingDomain` (UCI) e o domínio glicêmico (NHANES) medem o mesmo processo fisiológico, via a camada `PhysiologicalProcess` — uma afirmação testável, não uma fusão de dado.

---

## 6. Limitações

1. As duas fontes não são comparáveis diretamente entre si.
2. O critério de síndrome metabólica é uma adaptação (4 de 5 critérios do NCEP ATP III original).
3. A ordem de encontros na UCI é um proxy de ordem cronológica, não datas reais — intervalos absolutos de tempo não são conhecidos.
4. A classificação laboratorial de diabetes usada é uma simplificação do critério ADA completo (sem teste oral de tolerância à glicose, sem exigência de confirmação em consultas separadas).
5. Nenhuma das análises foi pré-registrada; todas são exploratórias.
6. A UCI abrange 1999–2008; padrões de utilização podem ter mudado desde então.
7. A estimativa de efeito causal da insulina sobre controle glicêmico não pôde ser obtida no NHANES (Seção 3.8) — permanece uma pergunta em aberto, não respondida neste trabalho.
8. Diagnósticos ICD-9 da UCI (`diag_1/2/3`) e creatinina/perfil lipídico do NHANES não foram incorporados — extensões naturais, não realizadas nesta rodada.

---

## 7. Conclusão

Duas fontes de dados reais, públicas e estruturalmente muito diferentes sobre diabetes tipo 2 produziram um corpo de achados clinicamente interpretáveis, consistentes com a literatura estabelecida onde há literatura para comparar (subdiagnóstico, confundimento por indicação em insulina, utilização prévia como preditor de readmissão), e metodologicamente informativos mesmo onde os resultados foram negativos ou estruturalmente limitados (curvatura sem discriminação, estimativa de efeito causal recusada, teste GNN sem vantagem clara). O achado mais amplo, atravessando quase todas as seções, é que o mesmo ferramental produziu conclusões opostas e corretas quando aplicado a populações com propriedades estruturais genuinamente diferentes — e que a distinção real vs. sintético, mais do que a distinção entre doenças, foi o que mais determinou se uma hipótese se confirmava ou não.

---

## Referências

American Diabetes Association. (2023). Diagnosis and Classification of Diabetes Mellitus. *Diabetes Care*.

Grundy, S. M., et al. (2005). Diagnosis and Management of the Metabolic Syndrome. *Circulation*.

Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Computational Statistics & Data Analysis*, 52(1), 258–271.

National Center for Health Statistics. National Health and Nutrition Examination Survey, 2017–March 2020 Pre-pandemic Data Files.

Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis*, 256(3), 810–864.

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*.
